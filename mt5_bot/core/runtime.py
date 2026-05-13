from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from alerts.telegram import TelegramNotifier
from brokers.backtest import BacktestGateway
from brokers.base import BrokerGateway
from brokers.mt5_live import MT5LiveGateway
from core.auto_tuning import ParameterAutoTuningLoop
from core.bar_gate import ClosedBarGate
from core.config import load_config
from core.control import (
    DESIRED_STATE_RUN,
    DESIRED_STATE_STOP,
    RuntimeControlChannel,
    write_desired_state,
)
from core.lifecycle import LifecycleController
from core.models import BotMode, DecisionAction, OrderResult, Position, Side, StrategyDecision, StrategyEvaluationContext, StrategyState
from core.no_trade_guard import NoTradeBiasGuard
from core.validation import check_live_readiness
from execution.daily_bleed_guard import DailyBleedGuard
from execution.execution_churn_guard import ExecutionChurnGuard
from execution.cost_edge_guard import CostEdgeGuard
from execution.entry_quality_guard import EntryQualityGuard
from execution.exit_engine import ExitEngine
from execution.exit_quality_guard import ExitQualityGuard
from execution.exit_retry_guard import ExitRetryGuard
from execution.manual_position_guard import ManualPositionGuard
from execution.mtf_confirm import MtfDirectionConfirm
from execution.order_manager import OrderManager
from execution.profit_lock import ProfitLockTrailingManager
from execution.risk_manager import RiskEngine
from execution.trade_journal import TradeJournal
from reports.active_opportunities import write_active_opportunity_reports
from reports.no_trade_report import build_no_trade_report_json, build_no_trade_report_markdown
from execution.trailing_guard import DynamicTrailingProfitGuard
from storage.json_store import JsonStore
from strategies.entry_filter import FeeAwareEntryFilter
from strategies.factory import build_strategies
from strategies.opportunity_scanner import TradeOpportunityScanner
from utils.liquidity import find_daily_levels


LOGGER = logging.getLogger(__name__)
REQUIRED_ACTIVE_SYMBOLS = ("BTCUSD", "ETHUSD", "GOLD")
DEFAULT_NASDAQ_UNIVERSE = ()


class TradingRuntime:
    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None) -> None:
        self.config = self._normalize_runtime_universe(config or {})
        self.config_path = Path(config_path).resolve() if config_path else None
        self._config_mtime_ns = self._safe_mtime_ns(self.config_path)

        general = self.config.get("general", {})
        self.mode = BotMode(str(general.get("mode", BotMode.LIVE.value)))
        self._assert_live_readiness_before_startup()
        execution_cfg = self.config.get("execution", {}) if isinstance(self.config.get("execution", {}), dict) else {}
        self.dry_run = bool(general.get("dry_run", True) or execution_cfg.get("dry_run", False))
        self.poll_seconds = max(0.1, float(general.get("poll_seconds", 1)))
        self.heartbeat_seconds = max(1, int(general.get("heartbeat_seconds", 10)))
        self.bars_per_request = max(50, int(general.get("bars_per_request", 300)))

        storage_cfg = self.config.get("storage", {})
        self.store = JsonStore(
            state_path=storage_cfg.get("state_path", "./state.json"),
            events_path=storage_cfg.get("events_path", "./events.jsonl"),
        )
        self.state = self.store.load_state()
        if not isinstance(self.state, dict):
            self.state = {}
        self.bar_gate = ClosedBarGate(
            snapshot=dict(self.state.get("bar_gate", {})) if isinstance(self.state.get("bar_gate"), dict) else {}
        )

        # notifier
        self.notifier = TelegramNotifier(self.config.get("telegram", {}))

        # broker
        if self.mode == BotMode.BACKTEST:
            self.broker: BrokerGateway = BacktestGateway(
                universe=list(self.config.get("universe", []) or []),
                general_cfg=dict(self.config.get("general", {}) or {}),
                backtest_cfg=dict(self.config.get("backtest", {}) or {}),
                execution_cfg=dict(self.config.get("execution", {}) or {}),
            )
        else:
            self.broker = MT5LiveGateway(self.config, notifier=self.notifier)

        # risk + execution
        risk_cfg = self._effective_risk_guard_config()
        self.risk_engine = RiskEngine(
            risk_cfg,
            snapshot=dict(self.state.get("risk_guard", {})) if isinstance(self.state.get("risk_guard"), dict) else {},
        )
        self.trailing_guard = DynamicTrailingProfitGuard(
            config=self.config.get("trailing_profit_guard", {}),
            snapshot=dict(self.state.get("trailing_profit_guard", {}))
            if isinstance(self.state.get("trailing_profit_guard"), dict)
            else {},
        )
        profit_lock_cfg = self.config.get("profit_lock", {}) if isinstance(self.config.get("profit_lock", {}), dict) else {}
        self.v4_profit_lock_enabled = bool(profit_lock_cfg.get("enabled", False))
        self.v4_profit_lock = ProfitLockTrailingManager(
            min_seconds_between_sltp_updates=float(profit_lock_cfg.get("min_seconds_between_sltp_updates", 10.0) or 10.0)
        )
        self.daily_bleed_guard = DailyBleedGuard(
            config=self.config.get("daily_bleed_guard", {}),
            snapshot=dict(self.state.get("daily_bleed_guard", {}))
            if isinstance(self.state.get("daily_bleed_guard"), dict)
            else {},
        )
        self.trade_opportunity_scanner = TradeOpportunityScanner(**self._trade_opportunity_scanner_config())
        self.fee_aware_entry_filter = FeeAwareEntryFilter(self._fee_aware_entry_filter_config())
        self.no_trade_guard = NoTradeBiasGuard(
            config=self.config.get("no_trade_bias_guard", {}),
            snapshot=dict(self.state.get("no_trade_bias_guard", {}))
            if isinstance(self.state.get("no_trade_bias_guard"), dict)
            else {},
        )
        self._active_opportunity_candidates: list[Dict[str, Any]] = []
        self.manual_position_guard = ManualPositionGuard(
            config=self.config.get("manual_position_guard", {}),
            snapshot=dict(self.state.get("manual_position_guard", {}))
            if isinstance(self.state.get("manual_position_guard"), dict)
            else {},
        )
        self.execution_churn_guard = ExecutionChurnGuard(
            config=self.config.get("execution_churn_guard", {}),
            snapshot=dict(self.state.get("execution_churn_guard", {}))
            if isinstance(self.state.get("execution_churn_guard"), dict)
            else {},
        )
        self.entry_quality_guard = EntryQualityGuard(
            config=self.config.get("entry_quality_guard", {}),
            snapshot=dict(self.state.get("entry_quality_guard", {}))
            if isinstance(self.state.get("entry_quality_guard"), dict)
            else {},
        )
        self.cost_edge_guard = CostEdgeGuard(
            config=self.config.get("cost_edge_guard", {}),
            snapshot=dict(self.state.get("cost_edge_guard", {}))
            if isinstance(self.state.get("cost_edge_guard"), dict)
            else {},
        )
        self.exit_quality_guard = ExitQualityGuard(self.config.get("exit_quality_guard", {}))
        self.exit_retry_guard = ExitRetryGuard(
            snapshot=dict(self.state.get("exit_retry_guard", {}))
            if isinstance(self.state.get("exit_retry_guard"), dict)
            else {},
        )
        self.exit_engine = ExitEngine()
        self.mtf_confirm = MtfDirectionConfirm(self.config.get("mtf_confirm", {}))
        self.trade_journal = TradeJournal(self.config.get("trade_journal", {}))
        self.order_manager = OrderManager(
            broker=self.broker,
            store=self.store,
            notifier=self.notifier,
            execution_cfg=self._order_manager_execution_config(),
            risk_engine=self.risk_engine,
            dry_run=self.dry_run,
        )
        self.order_manager.on_position_closed = self._on_position_closed
        self.order_manager.daily_bleed_guard = self.daily_bleed_guard
        self._sync_order_manager_execution_config()

        # strategies
        self.strategies = build_strategies(
            config=self.config,
            state_snapshot=self.state.get("strategy_state", {}) if isinstance(self.state, dict) else {},
        )
        self.auto_tuning = ParameterAutoTuningLoop(
            config=self.config,
            snapshot=dict(self.state.get("auto_tuning", {})) if isinstance(self.state.get("auto_tuning"), dict) else {},
        )
        if self.auto_tuning.overrides:
            self._apply_runtime_overrides(self.auto_tuning.overrides, context="startup:snapshot_restore")
        self.state["auto_tuning"] = self.auto_tuning.snapshot()
        self.state["trailing_profit_guard"] = self.trailing_guard.snapshot()
        self.state["daily_bleed_guard"] = self.daily_bleed_guard.snapshot()
        if hasattr(self, "no_trade_guard"):
            self.state["no_trade_bias_guard"] = self.no_trade_guard.snapshot()
        self.state["manual_position_guard"] = self.manual_position_guard.snapshot()
        self.state["execution_churn_guard"] = self.execution_churn_guard.snapshot()
        self.state["entry_quality_guard"] = self.entry_quality_guard.snapshot()
        self.state["cost_edge_guard"] = self.cost_edge_guard.snapshot()
        self.state["exit_retry_guard"] = self.exit_retry_guard.snapshot()

        self.dashboard_enabled = False
        self.control_channel: Optional[RuntimeControlChannel] = None
        self._sync_dashboard_control_channel()

        self.lifecycle = LifecycleController(self._on_emergency_shutdown)
        self._last_heartbeat_ts = 0.0
        self._watchdog_heartbeat_path = Path(__file__).resolve().parents[1] / "runtime" / "heartbeat.json"
        self._last_risk_guard_reason = ""
        self._broker_fatal_handled = False
        self._auto_tune_skip_signature: Optional[str] = None
        self._pending_protection_updates: Dict[str, Dict[str, Any]] = {}
        self._protection_blocked_symbols: set[str] = set()
        self._daily_reference_cache: Dict[str, Dict[str, Any]] = {}
        self._daily_reference_refresh_interval_seconds = max(
            60.0,
            float(general.get("daily_reference_refresh_seconds", 3600.0)),
        )
        self._last_daily_reference_refresh_ts = 0.0
        self._last_tick_fetch_by_symbol: Dict[str, datetime] = {}
        self._next_market_data_missing_log_ts_by_symbol: Dict[str, float] = {}

        self._pending_protection_updates = self._restore_pending_protection_updates(
            self.state.get("pending_protection_updates")
        )
        self.state["pending_protection_updates"] = dict(self._pending_protection_updates)
        self._refresh_daily_reference_levels()

        # Track positions across cycles so we can reconcile broker-side SL/TP closes that bypass
        # bot-initiated close_position() (and therefore never emit position_exit/trade_ledger).
        self._last_positions_by_ticket: Dict[str, Position] = {}
        self._recent_closed_tickets: Dict[str, float] = {}

    def _strategy_for_symbol(self, symbol: str) -> str:
        sym = str(symbol or "").strip().upper()
        for item in self.config.get("universe", []) or []:
            if str(item.get("symbol", "")).strip().upper() == sym:
                return str(item.get("strategy", "")).strip() or ""
        return ""

    def _detect_and_record_broker_closed_positions(self, broker_positions: list[Any]) -> None:
        if not hasattr(self, "_last_positions_by_ticket") or not isinstance(getattr(self, "_last_positions_by_ticket", None), dict):
            self._last_positions_by_ticket = {}
        if not hasattr(self, "_recent_closed_tickets") or not isinstance(getattr(self, "_recent_closed_tickets", None), dict):
            self._recent_closed_tickets = {}

        current: Dict[str, Position] = {}
        for p in broker_positions:
            try:
                ticket = str(int(getattr(p, "ticket")))
            except Exception:
                continue
            if not ticket:
                continue
            current[ticket] = p

        now_ts = time.time()
        for t, ts in list(self._recent_closed_tickets.items()):
            if (now_ts - float(ts or 0.0)) > 300.0:
                self._recent_closed_tickets.pop(t, None)

        disappeared = [t for t in self._last_positions_by_ticket.keys() if t not in current]
        if not disappeared:
            self._last_positions_by_ticket = current
            return

        for ticket in disappeared:
            if ticket in self._recent_closed_tickets:
                continue
            pos = self._last_positions_by_ticket.get(ticket)
            if pos is None:
                continue

            close_info = None
            try:
                close_info = self.broker.get_position_close_info(int(ticket))
            except Exception:
                close_info = None

            symbol = str(getattr(pos, "symbol", "") or "").strip().upper()
            strategy_name = self._strategy_for_symbol(symbol)

            if close_info is None:
                self.store.append_event(
                    {
                        "event": "broker_position_closed_untracked",
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "ticket": int(ticket),
                        "reason": "position_disappeared_no_history",
                    }
                )
                continue

            class _SyntheticResult:
                def __init__(self, info: Dict[str, Any]) -> None:
                    self.ok = True
                    self.status = "CLOSED_BROKER"
                    self.message = "broker_auto_close"
                    self.ticket = int(info.get("ticket") or 0)
                    self.retcode = 0
                    self.filled_price = info.get("exit_price")
                    self.pnl = info.get("pnl")
                    self.raw = {"close_info": dict(info)}

            result = _SyntheticResult(close_info)
            hold_seconds = None
            try:
                if getattr(pos, "time_open_utc", None) is not None:
                    # Fix: Broker time might be ahead of UTC, causing negative hold_seconds.
                    # Clamp to 0.0 to prevent logic errors in guards.
                    raw_seconds = (datetime.now(timezone.utc) - pos.time_open_utc.astimezone(timezone.utc)).total_seconds()
                    hold_seconds = max(0.0, raw_seconds)
            except Exception:
                hold_seconds = None

            reason = (
                f"BROKER_AUTO_CLOSE:{close_info.get('close_reason')}"
                if close_info.get("close_reason")
                else "BROKER_AUTO_CLOSE"
            )

            self.store.append_event(
                {
                    "event": "position_exit",
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "reason": reason,
                    "result": {
                        "ok": True,
                        "status": "CLOSED_BROKER",
                        "ticket": int(ticket),
                        "filled_price": close_info.get("exit_price"),
                        "pnl": close_info.get("pnl"),
                        "raw": {"close_info": dict(close_info)},
                    },
                    "exit_attempt_no": 0,
                }
            )
            trade_ledger = {
                "event": "trade_ledger",
                "ticket": int(ticket),
                "symbol": symbol,
                "strategy": strategy_name,
                "side": getattr(pos, "side", None).value if getattr(pos, "side", None) is not None else None,
                "entry_price": getattr(pos, "price_open", None),
                "exit_price": close_info.get("exit_price"),
                "volume": getattr(pos, "volume", None),
                "realized_pnl": close_info.get("pnl"),
                "pnl_status": "known" if close_info.get("pnl") is not None else "unknown",
                "reason": reason,
                "exit_attempt_no": 0,
                "exit_ok": True,
                "retcode": 0,
                "exit_fill_status": "FILLED" if close_info.get("exit_price") is not None else "UNFILLED",
                "exit_fail_reason": None,
            }
            self.store.append_event(trade_ledger)
            normalized_payload = dict(trade_ledger)
            normalized_payload["event"] = "trade_ledger_normalized"
            self.store.append_event(normalized_payload)

            self._on_position_closed(symbol=symbol, position=pos, result=result, reason=reason, hold_seconds=hold_seconds)
            self._recent_closed_tickets[ticket] = now_ts

        self._last_positions_by_ticket = current

    @staticmethod
    def _pending_ticket_key(ticket: Any) -> str:
        return str(int(ticket))

    def _restore_pending_protection_updates(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        restored: Dict[str, Dict[str, Any]] = {}
        for ticket, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            try:
                key = self._pending_ticket_key(ticket)
            except Exception:
                continue
            symbol = str(payload.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            entry = {
                "ticket": int(key),
                "symbol": symbol,
                "sl": payload.get("sl"),
                "tp": payload.get("tp"),
                "reason": str(payload.get("reason", "profit_lock_pending") or "profit_lock_pending"),
                "attempts": int(payload.get("attempts", 0) or 0),
                "last_attempt_ts": float(payload.get("last_attempt_ts", 0.0) or 0.0),
                "last_error": str(payload.get("last_error", "") or ""),
            }
            restored[key] = entry
        return restored

    def _queue_protection_retry(
        self,
        *,
        position: Position,
        sl: Optional[float],
        tp: Optional[float],
        reason: str,
        error_message: str,
    ) -> None:
        key = self._pending_ticket_key(position.ticket)
        existing = dict(self._pending_protection_updates.get(key, {}))
        attempts = int(existing.get("attempts", 0) or 0) + 1
        payload = {
            "ticket": int(position.ticket),
            "symbol": str(position.symbol).upper(),
            "sl": sl,
            "tp": tp,
            "reason": str(reason or "profit_lock_pending"),
            "attempts": attempts,
            "last_attempt_ts": float(time.time()),
            "last_error": str(error_message or ""),
        }
        self._pending_protection_updates[key] = payload
        self._protection_blocked_symbols.add(str(position.symbol).upper())
        report = self.execution_churn_guard.record_protection_failure(symbol=position.symbol, now_ts=time.time())
        self.store.append_event({"event": "execution_churn_guard_trigger", **report})
        self.store.append_event(
            {
                "event": "protection_retry_queued",
                "ticket": int(position.ticket),
                "symbol": str(position.symbol).upper(),
                "reason": payload["reason"],
                "attempts": attempts,
                "last_error": payload["last_error"],
                "sl": sl,
                "tp": tp,
            }
        )

    def _run_pending_protection_cycle(self, broker_positions: list[Position], excluded_symbols: set[str]) -> set[str]:
        blocked_symbols: set[str] = set()
        if not self._pending_protection_updates:
            return blocked_symbols
        by_ticket = {self._pending_ticket_key(pos.ticket): pos for pos in broker_positions}
        now_ts = time.time()
        retry_interval = float(self.execution_churn_guard.protection_retry_interval_seconds)
        max_attempts = int(self.execution_churn_guard.protection_retry_max_attempts)
        to_delete: list[str] = []
        for key, payload in list(self._pending_protection_updates.items()):
            position = by_ticket.get(key)
            symbol = str(payload.get("symbol", "")).strip().upper()
            if position is None:
                to_delete.append(key)
                continue
            if symbol in excluded_symbols:
                blocked_symbols.add(symbol)
                continue
            blocked_symbols.add(str(position.symbol).upper())
            if self.execution_churn_guard.is_protection_locked(symbol=symbol, now_ts=now_ts):
                continue
            last_attempt_ts = float(payload.get("last_attempt_ts", 0.0) or 0.0)
            if (now_ts - last_attempt_ts) < retry_interval:
                continue
            result = self.broker.modify_position_sl_tp(
                position=position,
                sl=payload.get("sl"),
                tp=payload.get("tp"),
                reason=str(payload.get("reason", "profit_lock_pending")),
            )
            self.store.append_event(
                {
                    "event": "pending_protection_retry",
                    "ticket": int(position.ticket),
                    "symbol": str(position.symbol).upper(),
                    "result": result.__dict__,
                }
            )
            if result.ok:
                to_delete.append(key)
                continue
            payload["attempts"] = int(payload.get("attempts", 0) or 0) + 1
            payload["last_attempt_ts"] = now_ts
            failure_message = str(getattr(result, "message", "") or "")
            payload["last_error"] = failure_message
            lowered_message = failure_message.strip().lower()
            if lowered_message == "no changes":
                to_delete.append(key)
                self.store.append_event(
                    {
                        "event": "protection_retry_abandoned",
                        "ticket": int(position.ticket),
                        "symbol": str(position.symbol).upper(),
                        "reason": "NO_CHANGES",
                        "attempts": int(payload["attempts"]),
                        "last_error": failure_message,
                    }
                )
                continue
            if int(payload["attempts"]) >= max_attempts:
                to_delete.append(key)
                self.store.append_event(
                    {
                        "event": "protection_retry_abandoned",
                        "ticket": int(position.ticket),
                        "symbol": str(position.symbol).upper(),
                        "reason": "MAX_ATTEMPTS",
                        "attempts": int(payload["attempts"]),
                        "last_error": failure_message,
                    }
                )
                continue
            self._pending_protection_updates[key] = payload
            report = self.execution_churn_guard.record_protection_failure(symbol=position.symbol, now_ts=now_ts)
            self.store.append_event({"event": "execution_churn_guard_trigger", **report})
        for key in to_delete:
            payload = self._pending_protection_updates.pop(key, None)
            if isinstance(payload, dict):
                symbol = str(payload.get("symbol", "")).strip().upper()
                if symbol:
                    blocked_symbols.discard(symbol)
        return blocked_symbols

    def _order_manager_execution_config(self) -> Dict[str, Any]:
        execution_cfg = dict(self.config.get("execution", {}) or {})
        style_cfg = self.config.get("execution_style", {}) if isinstance(self.config.get("execution_style", {}), dict) else {}
        risk_cfg = self.config.get("risk_per_trade", {}) if isinstance(self.config.get("risk_per_trade", {}), dict) else {}
        entry_cfg = self.config.get("entry_quality", {}) if isinstance(self.config.get("entry_quality", {}), dict) else {}
        exit_cfg = self.config.get("initial_exit", {}) if isinstance(self.config.get("initial_exit", {}), dict) else {}
        fee_cfg = dict(execution_cfg.get("fee_aware_fixed_risk", {}) or {})
        enabled_by_style = str(style_cfg.get("name", "")).strip() == "fee_aware_fixed_risk_profit_lock"
        if enabled_by_style or risk_cfg:
            fee_cfg["enabled"] = bool(fee_cfg.get("enabled", enabled_by_style or risk_cfg.get("enabled", True)))
            fee_cfg.setdefault("target_net_loss_usd", risk_cfg.get("target_net_loss_usd", 1.0))
            fee_cfg.setdefault("hard_max_net_loss_usd", risk_cfg.get("hard_max_net_loss_usd", 1.25))
            fee_cfg.setdefault("commission_per_lot", risk_cfg.get("commission_per_lot", 0.0))
            fee_cfg.setdefault("spread_points", risk_cfg.get("spread_points", 0.0))
            fee_cfg.setdefault("expected_slippage_points", risk_cfg.get("expected_slippage_points", 0.0))
            fee_cfg.setdefault("min_reward_to_net_risk_ratio", entry_cfg.get("min_reward_to_net_risk_ratio", 3.0))
            tp_cfg = exit_cfg.get("take_profit", {}) if isinstance(exit_cfg.get("take_profit", {}), dict) else {}
            fee_cfg.setdefault("min_tp_net_profit_usd", tp_cfg.get("min_profit_usd", 3.0))
            fee_cfg.setdefault("preferred_tp_net_profit_usd", tp_cfg.get("preferred_profit_usd", 5.0))
            execution_cfg["fee_aware_fixed_risk"] = fee_cfg
        return execution_cfg

    def _sync_order_manager_execution_config(self) -> None:
        execution_cfg = self._order_manager_execution_config()
        self.order_manager.execution_cfg = execution_cfg
        self.order_manager.comment_prefix = str(
            execution_cfg.get("comment_prefix", self.order_manager.comment_prefix or "quant_bot")
        )
        self.order_manager.allow_opposite_position = bool(
            execution_cfg.get("allow_opposite_position", self.order_manager.allow_opposite_position)
        )
        self.order_manager.default_volume = max(
            0.001,
            self._to_float(
                execution_cfg.get("default_volume", self.order_manager.default_volume),
                self.order_manager.default_volume,
            ),
        )
        try:
            max_positions = int(
                execution_cfg.get("max_positions_per_symbol", self.order_manager.max_positions_per_symbol)
            )
        except (TypeError, ValueError):
            max_positions = self.order_manager.max_positions_per_symbol
        self.order_manager.max_positions_per_symbol = max(1, max_positions)

    def _snapshot_all_strategies(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        persisted = self.state.get("strategy_state", {})
        if isinstance(persisted, dict):
            snapshot.update(persisted)
        for name, strategy in self.strategies.items():
            state = self._strategy_snapshot(strategy)
            if isinstance(state, dict):
                snapshot[name] = state
        return snapshot

    def _apply_runtime_overrides(self, requested_overrides: Dict[str, Any], context: str) -> Dict[str, Any]:
        requested = dict(requested_overrides or {})
        if not requested:
            return requested
        try:
            if "trailing_start_rr" in requested:
                requested["trailing_start_rr"] = max(0.8, float(requested["trailing_start_rr"]))
            if "trailing_atr_mult" in requested:
                requested["trailing_atr_mult"] = max(0.8, float(requested["trailing_atr_mult"]))
            if "regime_flip_exit_threshold" in requested:
                requested["regime_flip_exit_threshold"] = max(0.14, float(requested["regime_flip_exit_threshold"]))
        except (TypeError, ValueError):
            pass

        applied: Dict[str, Any] = {}
        for strategy in self.strategies.values():
            apply_runtime_overrides = getattr(strategy, "apply_runtime_overrides", None)
            if not callable(apply_runtime_overrides):
                continue
            try:
                strategy_applied = apply_runtime_overrides(requested)
            except Exception:
                LOGGER.exception(
                    "Failed applying strategy runtime overrides. context=%s strategy=%s",
                    context,
                    getattr(strategy, "name", strategy.__class__.__name__),
                )
                continue
            if isinstance(strategy_applied, dict):
                applied.update(strategy_applied)
        return applied

    def _refresh_strategy_runtime_stack(self, source: str) -> None:
        strategy_snapshot = self._snapshot_all_strategies()
        self.strategies = build_strategies(config=self.config, state_snapshot=strategy_snapshot)

        auto_snapshot = self.auto_tuning.snapshot()
        self.auto_tuning = ParameterAutoTuningLoop(config=self.config, snapshot=auto_snapshot)
        if self.auto_tuning.overrides:
            self._apply_runtime_overrides(self.auto_tuning.overrides, context=f"{source}:reload")

        self.state["strategy_state"] = strategy_snapshot
        self.state["auto_tuning"] = self.auto_tuning.snapshot()
        self._auto_tune_skip_signature = None

    @staticmethod
    def _strategy_state_value(strategy: Any, symbol: str) -> str:
        getter = getattr(strategy, "get_symbol_state", None)
        if not callable(getter):
            return "UNKNOWN"
        try:
            state_obj = getter(symbol)
            state = getattr(state_obj, "state", None)
            return str(getattr(state, "value", state) or "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def _force_strategy_cooldown(
        self,
        *,
        strategy: Any,
        symbol: str,
        strategy_name: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        getter = getattr(strategy, "get_symbol_state", None)
        if not callable(getter):
            return
        try:
            st = getter(symbol)
            min_cooldown = max(1, int(getattr(strategy, "min_cooldown_bars", 1) or 1))
            st.state = StrategyState.COOLDOWN
            st.pending_order = False
            st.cooldown_bars_remaining = max(min_cooldown, int(getattr(st, "cooldown_bars_remaining", 0) or 0))
            st.last_reason = str(reason or "ENTRY_FAILURE_FALLBACK")
            st.updated_at_utc = datetime.now(timezone.utc)
            self.store.append_event(
                {
                    "event": "strategy_state_forced_cooldown",
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "reason": st.last_reason,
                    "cooldown_bars_remaining": int(st.cooldown_bars_remaining),
                    "details": dict(details or {}),
                }
            )
        except Exception:
            LOGGER.exception(
                "Failed to force cooldown after order failure. symbol=%s strategy=%s",
                symbol,
                strategy_name,
            )

    def _skip_entry_and_cooldown(
        self,
        *,
        strategy: Any,
        symbol: str,
        strategy_name: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._force_strategy_cooldown(
            strategy=strategy,
            symbol=symbol,
            strategy_name=strategy_name,
            reason=f"ENTRY_BLOCKED:{reason}",
            details=details,
        )

    def _force_all_active_strategies_cooldown(self, reason: str) -> None:
        seen: set[tuple[str, str]] = set()
        for instrument in self.config.get("universe", []) or []:
            symbol = str(instrument.get("symbol", "")).strip().upper()
            strategy_name = str(instrument.get("strategy", "")).strip()
            if not symbol or not strategy_name:
                continue
            key = (symbol, strategy_name)
            if key in seen:
                continue
            seen.add(key)

            strategy = self.strategies.get(strategy_name)
            if strategy is None or not getattr(strategy, "enabled", True):
                continue

            self._force_strategy_cooldown(
                strategy=strategy,
                symbol=symbol,
                strategy_name=strategy_name,
                reason=f"RISK_GUARD:{reason}",
                details={"source": "risk_guard"},
            )

    @staticmethod
    def _strategy_snapshot(strategy: Any) -> Dict[str, Any]:
        getter = getattr(strategy, "get_all_symbol_states", None)
        if not callable(getter):
            return {}
        try:
            snapshot = getter()
            return snapshot if isinstance(snapshot, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _safe_mtime_ns(path: Optional[Path]) -> Optional[int]:
        if path is None:
            return None
        try:
            return int(path.stat().st_mtime_ns)
        except Exception:
            return None

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out

    @staticmethod
    def _finite_optional_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    def _trade_opportunity_scanner_config(self) -> Dict[str, Any]:
        scanner_cfg = self.config.get("opportunity_scanner", {})
        if not isinstance(scanner_cfg, dict):
            scanner_cfg = {}
        entry_quality = self.config.get("entry_quality", {}) if isinstance(self.config.get("entry_quality", {}), dict) else {}
        risk_cfg = self.config.get("risk_per_trade", {}) if isinstance(self.config.get("risk_per_trade", {}), dict) else {}
        return {
            "lookback_bars": int(scanner_cfg.get("lookback_bars", 20) or 20),
            "atr_period": int(scanner_cfg.get("atr_period", 14) or 14),
            "min_signal_score": float(scanner_cfg.get("min_signal_score", entry_quality.get("min_signal_score", 70.0)) or 70.0),
            "sweep_buffer_atr": float(scanner_cfg.get("sweep_buffer_atr", 0.05) or 0.05),
            "stop_buffer_atr": float(scanner_cfg.get("stop_buffer_atr", 0.05) or 0.05),
            "min_atr": float(scanner_cfg.get("min_atr", 0.0) or 0.0),
            "late_entry_atr_mult": float(scanner_cfg.get("late_entry_atr_mult", 0.75) or 0.75),
            "late_entry_min_rr": float(scanner_cfg.get("late_entry_min_rr", 1.0) or 1.0),
            "round_turn_cost": float(scanner_cfg.get("round_turn_cost", risk_cfg.get("round_turn_cost", 0.0)) or 0.0),
        }

    def _opportunity_scanner_enabled(self) -> bool:
        scanner_cfg = self.config.get("opportunity_scanner", {})
        if isinstance(scanner_cfg, dict) and "enabled" in scanner_cfg:
            return bool(scanner_cfg.get("enabled"))
        style = self.config.get("execution_style", {}) if isinstance(self.config.get("execution_style", {}), dict) else {}
        return str(style.get("name", "")).strip() == "fee_aware_fixed_risk_profit_lock"

    def _opportunity_scanner_drives_entries(self) -> bool:
        scanner_cfg = self.config.get("opportunity_scanner", {})
        if not isinstance(scanner_cfg, dict):
            return True
        return bool(scanner_cfg.get("drive_entries", True))

    def _fee_aware_entry_filter_config(self) -> Dict[str, Any]:
        entry_quality = self.config.get("entry_quality", {}) if isinstance(self.config.get("entry_quality", {}), dict) else {}
        risk_cfg = self.config.get("risk_per_trade", {}) if isinstance(self.config.get("risk_per_trade", {}), dict) else {}
        position_limits = self.config.get("position_limits", {}) if isinstance(self.config.get("position_limits", {}), dict) else {}
        execution_cfg = self.config.get("execution", {}) if isinstance(self.config.get("execution", {}), dict) else {}
        filter_cfg = self.config.get("fee_aware_entry_filter", {}) if isinstance(self.config.get("fee_aware_entry_filter", {}), dict) else {}
        merged = dict(filter_cfg)
        merged.setdefault("enabled", True)
        merged.setdefault("min_signal_score", entry_quality.get("min_signal_score", 70.0))
        merged.setdefault("min_reward_to_net_risk_ratio", entry_quality.get("min_reward_to_net_risk_ratio", 3.0))
        merged.setdefault("hard_max_net_loss_usd", risk_cfg.get("hard_max_net_loss_usd", 1.25))
        merged.setdefault("max_spread_points", execution_cfg.get("max_spread", risk_cfg.get("max_spread_points", 60.0)))
        merged.setdefault("max_open_positions", position_limits.get("max_open_positions_total", 1))
        return merged

    def _refresh_v4_opportunity_runtime_stack(self) -> None:
        self.trade_opportunity_scanner = TradeOpportunityScanner(**self._trade_opportunity_scanner_config())
        self.fee_aware_entry_filter = FeeAwareEntryFilter(self._fee_aware_entry_filter_config())
        self.no_trade_guard = NoTradeBiasGuard(
            config=self.config.get("no_trade_bias_guard", {}),
            snapshot=self.no_trade_guard.snapshot() if hasattr(self, "no_trade_guard") else {},
        )

    @staticmethod
    def _opportunity_value(opportunity: Any, name: str, default: Any = None) -> Any:
        if isinstance(opportunity, dict):
            return opportunity.get(name, default)
        return getattr(opportunity, name, default)

    def _opportunity_to_candidate_dict(
        self,
        opportunity: Any,
        *,
        filter_decision: Any = None,
        eligible: Optional[bool] = None,
    ) -> Dict[str, Any]:
        components = self._opportunity_value(opportunity, "components", {})
        if not isinstance(components, dict):
            components = {}
        direction = str(self._opportunity_value(opportunity, "direction", "")).lower()
        reasons = []
        if filter_decision is not None:
            raw_reasons = getattr(filter_decision, "reasons", None)
            if raw_reasons is None and isinstance(filter_decision, dict):
                raw_reasons = filter_decision.get("reasons")
            if isinstance(raw_reasons, (list, tuple, set)):
                reasons = [str(item) for item in raw_reasons if str(item)]
            elif raw_reasons:
                reasons = [str(raw_reasons)]
        allow = bool(getattr(filter_decision, "allow", eligible if eligible is not None else not reasons)) if filter_decision is not None else bool(eligible if eligible is not None else not reasons)
        return {
            "symbol": str(self._opportunity_value(opportunity, "symbol", "")).upper(),
            "timeframe": str(self._opportunity_value(opportunity, "timeframe", "")),
            "direction": direction,
            "entry_price": self._opportunity_value(opportunity, "entry_price"),
            "invalidation_price": self._opportunity_value(opportunity, "invalidation_price"),
            "target_reference_price": self._opportunity_value(opportunity, "target_reference_price"),
            "score": float(self._opportunity_value(opportunity, "signal_score", 0.0) or 0.0),
            "signal_score": float(self._opportunity_value(opportunity, "signal_score", 0.0) or 0.0),
            "fee_adjusted_rr": float(components.get("fee_adjusted_rr_value", components.get("fee_adjusted_rr", 0.0)) or 0.0),
            "spread": components.get("spread"),
            "current_spread": components.get("spread"),
            "setup": str(self._opportunity_value(opportunity, "reason", "")),
            "reason": str(self._opportunity_value(opportunity, "reason", "")),
            "late_entry": bool(self._opportunity_value(opportunity, "late_entry", False)),
            "eligible": allow,
            "allow": allow,
            "block_reasons": reasons,
            "components": dict(components),
        }

    def _entry_filter_context(self, *, symbol: str, direction: str, position_count: int) -> Dict[str, Any]:
        position_limits = self.config.get("position_limits", {}) if isinstance(self.config.get("position_limits", {}), dict) else {}
        execution_cfg = self.config.get("execution", {}) if isinstance(self.config.get("execution", {}), dict) else {}
        live_gate_open = bool(self.dry_run or self.mode == BotMode.BACKTEST or execution_cfg.get("live_trading_enabled", False))
        return {
            "symbol": str(symbol or "").upper(),
            "direction": direction,
            "setup_key": "v4_opportunity_scanner",
            "daily_bleed_guard": self.daily_bleed_guard,
            "now_ts": time.time(),
            "open_positions_count": int(position_count),
            "max_open_positions": int(position_limits.get("max_open_positions_total", 1) or 1),
            "live_gate_open": live_gate_open,
            "paper_only_mode": False,
        }

    def _scan_and_filter_opportunities(
        self,
        *,
        symbol: str,
        timeframe: str,
        bars: Any,
        position_count: int,
    ) -> list[tuple[Any, Any, Dict[str, Any]]]:
        if not self._opportunity_scanner_enabled():
            return []
        spread_points = None
        spread_getter = getattr(self.broker, "get_live_spread", None)
        if callable(spread_getter):
            try:
                spread_points = spread_getter(symbol)
            except Exception:
                spread_points = None
        if spread_points is None:
            risk_cfg = self.config.get("risk_per_trade", {}) if isinstance(self.config.get("risk_per_trade", {}), dict) else {}
            spread_points = self._finite_optional_float(risk_cfg.get("spread_points")) or 0.0
        opportunities = self.trade_opportunity_scanner.scan(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            spread=spread_points,
        )
        out: list[tuple[Any, Any, Dict[str, Any]]] = []
        for opportunity in opportunities:
            self.no_trade_guard.record_raw_signal(opportunity)
            self.no_trade_guard.record_scored_signal(opportunity)
            candidate = self._opportunity_to_candidate_dict(opportunity)
            ctx = self._entry_filter_context(symbol=symbol, direction=str(candidate.get("direction", "")), position_count=position_count)
            filter_decision = self.fee_aware_entry_filter.evaluate(candidate, ctx)
            self.no_trade_guard.record_filter_decision(candidate, filter_decision)
            candidate = self._opportunity_to_candidate_dict(opportunity, filter_decision=filter_decision)
            self._active_opportunity_candidates.append(candidate)
            out.append((opportunity, filter_decision, candidate))
        return out

    def _decision_from_opportunity(self, opportunity: Any, candidate: Dict[str, Any]) -> Optional[StrategyDecision]:
        direction = str(candidate.get("direction", "")).lower()
        action = DecisionAction.BUY if direction == "long" else DecisionAction.SELL if direction == "short" else None
        if action is None:
            return None
        return StrategyDecision(
            action=action,
            reason=str(candidate.get("setup") or "V4_OPPORTUNITY_SCANNER_ENTRY"),
            strategy="v4_opportunity_scanner",
            confidence=float(candidate.get("score", 0.0) or 0.0) / 100.0,
            sl=self._finite_optional_float(candidate.get("invalidation_price")),
            tp=self._finite_optional_float(candidate.get("target_reference_price")),
            metadata={
                "v4_opportunity_scanner": True,
                "entry_price": self._finite_optional_float(candidate.get("entry_price")),
                "target_reference_price": self._finite_optional_float(candidate.get("target_reference_price")),
                "signal_score": float(candidate.get("score", 0.0) or 0.0),
                "fee_adjusted_rr": float(candidate.get("fee_adjusted_rr", 0.0) or 0.0),
                "opportunity": dict(candidate),
            },
        )

    def _write_v4_opportunity_reports(self) -> None:
        if not self._opportunity_scanner_enabled() and "reports" not in self.config:
            return
        if not hasattr(self, "no_trade_guard"):
            self.no_trade_guard = NoTradeBiasGuard(config=self.config.get("no_trade_bias_guard", {}))
        reports_cfg = self.config.get("reports", {}) if isinstance(self.config.get("reports", {}), dict) else {}
        if not bool(reports_cfg.get("enabled", True)):
            return
        output_dir = Path(str(reports_cfg.get("output_dir", "reports")))
        try:
            write_active_opportunity_reports(
                {
                    "current_symbols": [str(item.get("symbol", "")).strip().upper() for item in self.config.get("universe", []) or []],
                    "current_timeframe": str((self.config.get("universe", [{}]) or [{}])[0].get("timeframe", "")),
                    "opportunities": list(self._active_opportunity_candidates),
                },
                json_path=output_dir / "active_opportunities.json",
                markdown_path=output_dir / "active_opportunities.md",
            )
            no_trade_snapshot = self.no_trade_guard.snapshot()
            (output_dir / "no_trade_diagnostics.json").write_text(
                build_no_trade_report_json(no_trade_snapshot) + "\n",
                encoding="utf-8",
            )
            (output_dir / "no_trade_diagnostics.md").write_text(
                build_no_trade_report_markdown(no_trade_snapshot),
                encoding="utf-8",
            )
            self.store.append_event(
                {
                    "event": "v4_opportunity_reports_written",
                    "output_dir": str(output_dir),
                    "candidate_count": len(self._active_opportunity_candidates),
                    "no_trade_status": no_trade_snapshot.get("status"),
                }
            )
        except Exception as exc:
            LOGGER.exception("Failed to write v4 opportunity reports")
            self.store.append_event({"event": "v4_opportunity_report_error", "error": str(exc)})

    @classmethod
    def _normalize_runtime_universe(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(config or {})
        execution_cfg = dict(cfg.get("execution", {}) or {})
        default_volume = cls._to_float(execution_cfg.get("default_volume", 0.01), 0.01)
        if default_volume <= 0:
            default_volume = 0.01

        entries_by_symbol: Dict[str, Dict[str, Any]] = {}
        ordered_symbols = []

        for raw_item in cfg.get("universe", []) or []:
            if not isinstance(raw_item, dict):
                continue
            symbol = cls._normalize_symbol(raw_item.get("symbol"))
            if not symbol:
                continue
            item = dict(raw_item)
            item["symbol"] = symbol
            strategy = str(item.get("strategy", "")).strip() or "trend_regime_sm"
            timeframe = str(item.get("timeframe", "")).strip() or "TIMEFRAME_M1"
            volume = cls._to_float(item.get("volume", default_volume), default_volume)
            if volume <= 0:
                volume = default_volume
            item["strategy"] = strategy
            item["timeframe"] = timeframe
            item["volume"] = volume
            entries_by_symbol[symbol] = item
            if symbol not in ordered_symbols:
                ordered_symbols.append(symbol)

        raw_required = cfg.get("required_active_symbols", REQUIRED_ACTIVE_SYMBOLS)
        required_symbols = []
        for value in raw_required or []:
            symbol = cls._normalize_symbol(value)
            if symbol and symbol not in required_symbols:
                required_symbols.append(symbol)
        if not required_symbols:
            required_symbols = list(REQUIRED_ACTIVE_SYMBOLS)

        raw_nasdaq = cfg.get("nasdaq_universe", DEFAULT_NASDAQ_UNIVERSE)
        nasdaq_symbols = []
        for value in raw_nasdaq or []:
            symbol = cls._normalize_symbol(value)
            if symbol and symbol not in nasdaq_symbols:
                nasdaq_symbols.append(symbol)
        if not nasdaq_symbols:
            nasdaq_symbols = list(DEFAULT_NASDAQ_UNIVERSE)

        for symbol in [*required_symbols, *nasdaq_symbols]:
            if symbol in entries_by_symbol:
                continue
            entries_by_symbol[symbol] = {
                "symbol": symbol,
                "strategy": "trend_regime_sm",
                "timeframe": "TIMEFRAME_M1",
                "volume": default_volume,
            }
            ordered_symbols.append(symbol)

        cfg["required_active_symbols"] = required_symbols
        cfg["nasdaq_universe"] = nasdaq_symbols
        cfg["universe"] = [entries_by_symbol[symbol] for symbol in ordered_symbols if symbol in entries_by_symbol]
        return cfg

    def _sync_dashboard_control_channel(self) -> None:
        dashboard_cfg = self.config.get("dashboard", {}) or {}
        enabled = bool(dashboard_cfg.get("enabled", False))
        self.dashboard_enabled = enabled
        # Safe stop/resume must work even when dashboard UI is disabled.
        control_path = str(dashboard_cfg.get("control_path", "./runtime_control.json"))
        existing = self.control_channel
        if isinstance(existing, RuntimeControlChannel):
            try:
                if str(existing.path) == control_path:
                    return
            except Exception:
                pass
        self.control_channel = RuntimeControlChannel(path=control_path)

    def _effective_risk_guard_config(self) -> Dict[str, Any]:
        risk_cfg = dict(self.config.get("risk_guard", {}) or {})
        if self.mode == BotMode.BACKTEST:
            risk_cfg["dynamic_lot_enabled"] = False
        return risk_cfg

    def _assert_live_readiness_before_startup(self) -> None:
        if self.mode != BotMode.LIVE:
            return
        validation_cfg = self.config.get("validation", {}) or {}
        report_path = str(validation_cfg.get("report_path", ""))
        require_oos = bool(validation_cfg.get("require_oos_pass", False))
        if not require_oos:
            return
        ok, reason = check_live_readiness(report_path)
        if not ok:
            raise ValueError(f"Live readiness failed: {reason}")

    def _apply_runtime_config(self, next_config: Dict[str, Any], source: str) -> None:
        self.config = self._normalize_runtime_universe(next_config or {})

        general = self.config.get("general", {})
        execution_cfg = self.config.get("execution", {}) if isinstance(self.config.get("execution", {}), dict) else {}
        self.dry_run = bool(general.get("dry_run", self.dry_run) or execution_cfg.get("dry_run", False))
        self.poll_seconds = max(0.1, float(general.get("poll_seconds", self.poll_seconds)))
        self.heartbeat_seconds = max(1, int(general.get("heartbeat_seconds", self.heartbeat_seconds)))
        self.bars_per_request = max(50, int(general.get("bars_per_request", self.bars_per_request)))
        self._sync_dashboard_control_channel()
        self._daily_reference_refresh_interval_seconds = max(
            60.0,
            float(general.get("daily_reference_refresh_seconds", self._daily_reference_refresh_interval_seconds)),
        )

        update_order_gate = getattr(self.broker, "update_order_gate", None)
        if callable(update_order_gate):
            update_order_gate(self.config)

        self.risk_engine.update_config(self._effective_risk_guard_config())
        self.trailing_guard.update_config(self.config.get("trailing_profit_guard", {}))
        self.manual_position_guard.update_config(self.config.get("manual_position_guard", {}))
        self.execution_churn_guard.update_config(self.config.get("execution_churn_guard", {}))
        self.entry_quality_guard.update_config(self.config.get("entry_quality_guard", {}))
        self.cost_edge_guard.update_config(self.config.get("cost_edge_guard", {}))
        self._refresh_v4_opportunity_runtime_stack()
        self.exit_quality_guard.update_config(self.config.get("exit_quality_guard", {}))
        self.mtf_confirm.update_config(self.config.get("mtf_confirm", {}))
        self.trade_journal = TradeJournal(self.config.get("trade_journal", {}))
        self.order_manager.dry_run = self.dry_run
        self._sync_order_manager_execution_config()
        self._refresh_strategy_runtime_stack(source=source)
        self._refresh_daily_reference_levels()
        self.state["entry_quality_guard"] = self.entry_quality_guard.snapshot()
        self.state["cost_edge_guard"] = self.cost_edge_guard.snapshot()
        self.state["exit_retry_guard"] = self.exit_retry_guard.snapshot()

        self.store.append_event(
            {
                "event": "runtime_config_applied",
                "source": source,
                "poll_seconds": self.poll_seconds,
                "heartbeat_seconds": self.heartbeat_seconds,
                "bars_per_request": self.bars_per_request,
                "universe_size": len(self.config.get("universe", []) or []),
                "monitored_symbols": [str(item.get("symbol", "")).strip() for item in self.config.get("universe", []) or []],
                "required_active_symbols": list(self.config.get("required_active_symbols", []) or []),
                "nasdaq_universe": list(self.config.get("nasdaq_universe", []) or []),
                "trailing_profit_guard": {
                    "enabled": bool(self.trailing_guard.enabled),
                    "retain_ratio": float(self.trailing_guard.retain_ratio),
                    "min_activation_profit_usd": float(self.trailing_guard.min_activation_profit_usd),
                    "break_even_enabled": bool(self.trailing_guard.break_even_enabled),
                    "break_even_activation_profit_usd": float(self.trailing_guard.break_even_activation_profit_usd),
                    "break_even_lock_profit_usd": float(self.trailing_guard.break_even_lock_profit_usd),
                },
                "manual_position_guard": {
                    "enabled": bool(self.manual_position_guard.enabled),
                    "symbols": sorted(self.manual_position_guard.symbols),
                    "retain_ratio": float(self.manual_position_guard.retain_ratio),
                    "min_activation_profit_usd": float(self.manual_position_guard.min_activation_profit_usd),
                },
                "execution_churn_guard": {
                    "enabled": bool(self.execution_churn_guard.enabled),
                    "reentry_cooldown_seconds": float(self.execution_churn_guard.reentry_cooldown_seconds),
                    "max_entries_per_symbol_per_hour": int(self.execution_churn_guard.max_entries_per_symbol_per_hour),
                    "min_hold_bars_floor": int(self.execution_churn_guard.min_hold_bars_floor),
                },
                "entry_quality_guard": {
                    "enabled": bool(self.entry_quality_guard.enabled),
                    "min_score": float(self.entry_quality_guard.min_score),
                    "min_score_risk_off": float(self.entry_quality_guard.min_score_risk_off),
                    "min_score_risk_on": float(self.entry_quality_guard.min_score_risk_on),
                },
                "cost_edge_guard": {
                    "enabled": bool(self.cost_edge_guard.enabled),
                    "min_edge_to_cost_ratio_default": float(self.cost_edge_guard.min_edge_to_cost_ratio_default),
                },
                "exit_quality_guard": {
                    "enabled": bool(self.exit_quality_guard.enabled),
                    "tiny_profit_block_usd": float(self.exit_quality_guard.tiny_profit_block_usd),
                    "min_hold_seconds_for_soft_exit": float(self.exit_quality_guard.min_hold_seconds_for_soft_exit),
                },
                "trade_journal": {
                    "enabled": bool(self.trade_journal.enabled),
                    "output_dir": str(self.trade_journal.output_dir),
                },
            }
        )

    def _estimate_close_pnl(
        self,
        position: Position,
        filled_price: Optional[float],
        reported_pnl: Optional[float] = None,
    ) -> Optional[float]:
        try:
            if reported_pnl is not None:
                parsed = float(reported_pnl)
                if math.isfinite(parsed):
                    return parsed
        except (TypeError, ValueError):
            pass
        if filled_price is None:
            return None

        try:
            constraints = self.broker.get_symbol_constraints(position.symbol)
            contract_size = float(constraints.contract_size) if constraints is not None else 1.0
            direction = 1.0 if position.side == Side.BUY else -1.0
            gross = (float(filled_price) - float(position.price_open)) * direction * float(position.volume) * contract_size
            metadata = position.metadata if isinstance(position.metadata, dict) else {}
            swap = self._to_float(metadata.get("swap"), 0.0)
            commission = self._to_float(metadata.get("commission"), 0.0)
            pnl = gross + swap + commission
            if math.isfinite(pnl):
                return pnl
        except Exception:
            return None
        return None

    def _reload_config_if_changed(self) -> None:
        if self.config_path is None:
            return
        mtime_ns = self._safe_mtime_ns(self.config_path)
        if mtime_ns is None or mtime_ns == self._config_mtime_ns:
            return

        try:
            reloaded = load_config(self.config_path)
        except Exception as exc:
            LOGGER.exception("Runtime config reload failed. path=%s", self.config_path)
            self.store.append_event(
                {
                    "event": "runtime_config_reload_failed",
                    "path": str(self.config_path),
                    "error": str(exc),
                }
            )
            return

        reloaded.setdefault("general", {})["mode"] = self.mode.value
        self._config_mtime_ns = mtime_ns
        self._apply_runtime_config(reloaded, source="config_file")

    def _broker_fatal_reason(self) -> Optional[str]:
        getter = getattr(self.broker, "fatal_error", None)
        if callable(getter):
            try:
                reason = getter()
            except Exception:
                return None
            return str(reason) if reason else None
        return None

    def _halt_for_broker_fatal(self) -> bool:
        reason = self._broker_fatal_reason()
        if not reason:
            return False
        if not self._broker_fatal_handled:
            self._broker_fatal_handled = True
            LOGGER.error("Broker fatal condition detected: %s", reason)
            self.store.append_event({"event": "broker_fatal", "reason": reason})
            self.notifier.send_error(f"Broker fatal condition: {reason}")
            self.lifecycle.request_stop(f"broker_fatal:{reason}")
        return True

    def _on_emergency_shutdown(self, reason: str) -> None:
        LOGGER.warning("Emergency shutdown triggered: %s", reason)
        try:
            self.broker.disconnect()
        finally:
            try:
                self.store.save_state(self.state)
            except Exception:
                pass

    def _run_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat_ts < self.heartbeat_seconds:
            return

        ok = self.broker.heartbeat()
        self._last_heartbeat_ts = now
        if not ok:
            if self._halt_for_broker_fatal():
                return
            self.store.append_event({"event": "heartbeat_failed"})

    def _write_watchdog_heartbeat(self, now_ts: Optional[float] = None) -> None:
        ts = float(now_ts if now_ts is not None else time.time())
        hb_path = getattr(self, "_watchdog_heartbeat_path", None)
        if hb_path is None:
            hb_path = Path(__file__).resolve().parents[1] / "runtime" / "heartbeat.json"
            self._watchdog_heartbeat_path = hb_path
        payload = json.dumps({"ts": ts, "state": "RUN"}, ensure_ascii=False)

        # Increased attempts and backoff for Windows stability
        for attempt in range(12):
            tmp_path = hb_path.with_name(f"{hb_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                hb_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(payload, encoding="utf-8")
                try:
                    tmp_path.replace(hb_path)
                except (PermissionError, OSError):
                    if attempt < 11:
                        # Linear backoff to desync from watchdog reads
                        time.sleep(0.1 + (attempt * 0.05))
                        continue
                    LOGGER.warning("PermissionError writing heartbeat after 12 attempts: %s", hb_path)
                    return  # Suppress crash
                break
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                if attempt < 11:
                    time.sleep(0.25)
                else:
                    LOGGER.warning("Exception writing heartbeat after 12 attempts.")

    def _active_strategy_symbols(self) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        for instrument in self.config.get("universe", []) or []:
            symbol = str(instrument.get("symbol", "")).strip().upper()
            strategy_name = str(instrument.get("strategy", "")).strip()
            if not symbol or symbol in seen:
                continue
            strategy = self.strategies.get(strategy_name)
            if not strategy or not getattr(strategy, "enabled", True):
                continue
            symbols.append(symbol)
            seen.add(symbol)
        return symbols

    def _refresh_daily_reference_levels(self) -> None:
        if not bool(getattr(self.broker, "connected", True)):
            return

        active_symbols = self._active_strategy_symbols()
        active_set = set(active_symbols)
        for cached_symbol in list(self._daily_reference_cache.keys()):
            if cached_symbol not in active_set:
                self._daily_reference_cache.pop(cached_symbol, None)

        if not active_symbols:
            self._last_daily_reference_refresh_ts = time.time()
            return

        refreshed_at = datetime.now(timezone.utc).isoformat()
        refreshed_count = 0
        available_count = 0
        for symbol in active_symbols:
            bars = self.broker.fetch_bars(symbol=symbol, timeframe="TIMEFRAME_D1", bars=5)
            if self._halt_for_broker_fatal():
                return

            pdh: Optional[float] = None
            pdl: Optional[float] = None
            bar_count = 0
            if bars is not None and not bars.empty:
                bar_count = int(len(bars))
                raw_pdh, raw_pdl = find_daily_levels(bars)
                pdh = self._finite_optional_float(raw_pdh)
                pdl = self._finite_optional_float(raw_pdl)
                if pdh is not None and pdl is not None:
                    available_count += 1

            self._daily_reference_cache[symbol] = {
                "symbol": symbol,
                "pdh": pdh,
                "pdl": pdl,
                "timeframe": "TIMEFRAME_D1",
                "bars": bar_count,
                "updated_at_utc": refreshed_at,
            }
            refreshed_count += 1

        self._last_daily_reference_refresh_ts = time.time()
        self.store.append_event(
            {
                "event": "daily_reference_levels_refreshed",
                "symbols": active_symbols,
                "refreshed_count": int(refreshed_count),
                "available_count": int(available_count),
                "timeframe": "TIMEFRAME_D1",
            }
        )

    def _m5_reverse_confirmed_for_exit(self, symbol: str, position: Position) -> bool:
        if not self.mtf_confirm.is_symbol_enabled(symbol):
            return False
        bars = self.broker.fetch_bars(
            symbol=symbol,
            timeframe=self.mtf_confirm.confirm_timeframe,
            bars=self.bars_per_request,
        )
        if bars is None or bars.empty:
            return False
        side_action = DecisionAction.SELL if position.side == Side.BUY else DecisionAction.BUY
        return bool(self.mtf_confirm.allow_entry(symbol=symbol, action=side_action, bars=bars))

    def _evaluate_profit_lock_for_position(self, position: Position) -> Optional[Any]:
        constraints = self.broker.get_symbol_constraints(position.symbol)
        contract_size = float(constraints.contract_size) if constraints is not None else 1.0
        metadata = position.metadata if isinstance(position.metadata, dict) else {}
        if self.v4_profit_lock_enabled:
            current_price = self._to_float(metadata.get("price_current"), 0.0)
            if current_price > 0:
                net_pnl = self._to_float(metadata.get("floating_pnl"), 0.0) + self._to_float(metadata.get("swap"), 0.0) + self._to_float(metadata.get("commission"), 0.0)
                decision = self.v4_profit_lock.evaluate(
                    position=position,
                    current_price=current_price,
                    now=time.time(),
                    symbol_spec=constraints,
                    contract_size=contract_size,
                    net_unrealized_pnl=net_pnl,
                    record_update=False,
                )
                if decision.should_modify:
                    modify = self.broker.modify_position_sl_tp(
                        position=position,
                        sl=decision.sl_price,
                        tp=decision.tp_price if decision.tp_price is not None else position.tp,
                        reason="v4_profit_lock",
                    )
                    self.store.append_event(
                        {
                            "event": "v4_profit_lock_sltp_sync",
                            "symbol": str(position.symbol),
                            "ticket": int(position.ticket),
                            "sl": decision.sl_price,
                            "tp": decision.tp_price,
                            "lock_net_profit": decision.lock_net_profit,
                            "target_net_profit": decision.target_net_profit,
                            "trigger_net_profit": decision.trigger_net_profit,
                            "net_unrealized_pnl": decision.net_unrealized_pnl,
                            "result": modify.__dict__,
                        }
                    )
                    if modify.ok:
                        self.v4_profit_lock.mark_updated(position.ticket, time.time())
                    else:
                        self._queue_protection_retry(
                            position=position,
                            sl=decision.sl_price,
                            tp=decision.tp_price if decision.tp_price is not None else position.tp,
                            reason="v4_profit_lock",
                            error_message=str(getattr(modify, "message", "") or ""),
                        )
        sl_signal = self.trailing_guard.evaluate_profit_lock_sl(position=position, contract_size=contract_size)
        if sl_signal is not None and self.trailing_guard.break_even_sync_sl:
            modify = self.broker.modify_position_sl_tp(
                position=position,
                sl=sl_signal.desired_sl,
                tp=position.tp,
                reason=sl_signal.reason,
            )
            self.store.append_event(
                {
                    "event": "profit_lock_sl_sync",
                    "symbol": sl_signal.symbol,
                    "ticket": sl_signal.ticket,
                    "stage": sl_signal.stage,
                    "existing_sl": sl_signal.existing_sl,
                    "desired_sl": sl_signal.desired_sl,
                    "lock_pnl_usd": sl_signal.lock_pnl_usd,
                    "current_pnl_usd": sl_signal.current_pnl_usd,
                    "peak_pnl_usd": sl_signal.peak_pnl_usd,
                    "result": modify.__dict__,
                }
            )
            if not modify.ok:
                message = str(getattr(modify, "message", "") or "")
                lowered = message.strip().lower()
                if lowered == "no changes" or getattr(modify, "retcode", None) == 10025:
                    self.store.append_event(
                        {
                            "event": "profit_lock_noop",
                            "symbol": sl_signal.symbol,
                            "ticket": sl_signal.ticket,
                            "stage": sl_signal.stage,
                            "existing_sl": sl_signal.existing_sl,
                            "desired_sl": sl_signal.desired_sl,
                            "result": modify.__dict__,
                        }
                    )
                else:
                    self._queue_protection_retry(
                        position=position,
                        sl=sl_signal.desired_sl,
                        tp=position.tp,
                        reason=sl_signal.reason,
                        error_message=message,
                    )
        return self.trailing_guard.evaluate_position(position)

    def _save_runtime_state(self) -> None:
        self.state["strategy_state"] = {name: self._strategy_snapshot(s) for name, s in self.strategies.items()}
        self.state["bar_gate"] = self.bar_gate.snapshot()
        self.state["risk_guard"] = self.risk_engine.snapshot()
        self.state["trailing_profit_guard"] = self.trailing_guard.snapshot()
        self.state["daily_bleed_guard"] = self.daily_bleed_guard.snapshot()
        if hasattr(self, "no_trade_guard"):
            self.state["no_trade_bias_guard"] = self.no_trade_guard.snapshot()
        self.state["manual_position_guard"] = self.manual_position_guard.snapshot()
        self.state["execution_churn_guard"] = self.execution_churn_guard.snapshot()
        self.state["entry_quality_guard"] = self.entry_quality_guard.snapshot()
        self.state["cost_edge_guard"] = self.cost_edge_guard.snapshot()
        self.state["exit_retry_guard"] = self.exit_retry_guard.snapshot()
        self.state["pending_protection_updates"] = dict(self._pending_protection_updates)
        self.store.save_state(self.state)

    def _on_position_closed(
        self,
        symbol: str,
        position: Position,
        result: Any,
        reason: str,
        hold_seconds: Optional[float],
    ) -> None:
        try:
            pnl = float(getattr(result, "pnl", None))
        except (TypeError, ValueError):
            pnl = None
        metadata = position.metadata if isinstance(position.metadata, dict) else {}
        swap = self._to_float(metadata.get("swap"), 0.0)
        commission = self._to_float(metadata.get("commission"), 0.0)
        fee = 0.0
        raw = getattr(result, "raw", None)
        if isinstance(raw, dict):
            fee = self._to_float(raw.get("fee"), 0.0)
        realized_cost_usd = abs(float(swap + commission + fee))
        self.cost_edge_guard.record_cost(symbol=str(symbol or "").upper(), realized_cost_usd=realized_cost_usd)

        report = self.execution_churn_guard.record_close(
            symbol=symbol,
            now_ts=time.time(),
            realized_pnl=pnl,
            hold_seconds=hold_seconds,
        )
        if report is not None:
            self.store.append_event({"event": "execution_churn_guard_trigger", **report})
        if pnl is not None:
            daily_report = self.daily_bleed_guard.record_trade_close(
                symbol=symbol,
                realized_pnl=float(pnl),
                now_ts=time.time(),
                direction=getattr(getattr(position, "side", None), "value", None),
                setup_key=str(reason or ""),
            )
            self.store.append_event({"event": "daily_bleed_guard_update", **daily_report})
            self.state["daily_bleed_guard"] = self.daily_bleed_guard.snapshot()
        journal = self.trade_journal.record_trade(
            symbol=str(symbol or "").upper(),
            reason=str(reason or ""),
            pnl=pnl,
            hold_seconds=hold_seconds,
            entry_price=getattr(position, "price_open", None),
            exit_price=getattr(result, "filled_price", None),
        )
        if journal is not None:
            self.store.append_event(
                {
                    "event": "trade_journal_entry",
                    "symbol": str(symbol or "").upper(),
                    "classification": journal.get("classification"),
                    "is_anomaly": bool(journal.get("is_anomaly", False)),
                    "path": journal.get("path"),
                }
            )
        profile_report = self.entry_quality_guard.record_closed_trade(
            ticket=getattr(position, "ticket", None),
            symbol=str(symbol or "").upper(),
            pnl=pnl,
            hold_seconds=hold_seconds,
        )
        if bool(profile_report.get("updated", False)):
            self.store.append_event(
                {
                    "event": "winner_profile_updated",
                    "symbol": str(symbol or "").upper(),
                    "closed_count": int(profile_report.get("closed_count", 0)),
                    "winner_profile": dict(profile_report.get("winner_profile", {})),
                }
            )
        self.state["entry_quality_guard"] = self.entry_quality_guard.snapshot()
        self.state["cost_edge_guard"] = self.cost_edge_guard.snapshot()
        # Prevent duplicate reconciliation closes for a short window.
        try:
            ticket = getattr(position, "ticket", None)
            if ticket is not None:
                self._recent_closed_tickets[str(int(ticket))] = time.time()
        except Exception:
            pass

    def _reconcile_strategy_states(self, broker_positions: list[Any]) -> None:
        """
        Syncs strategy states with actual broker positions to prevent 'EXIT_READY' ghost locks.
        """
        active_symbols = {str(p.symbol).strip().upper() for p in broker_positions if hasattr(p, "symbol")}
        seen_pairs: set[tuple[str, str]] = set()
        for instrument in self.config.get("universe", []) or []:
            symbol = str(instrument.get("symbol", "")).strip().upper()
            strategy_name = str(instrument.get("strategy", "")).strip()
            strategy = self.strategies.get(strategy_name)
            if not symbol or strategy is None:
                continue
            pair = (strategy_name, symbol)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            try:
                st = strategy.get_symbol_state(symbol)
                # If bot thinks we are in a position related state but broker has nothing
                if st.state in {StrategyState.IN_POSITION, StrategyState.EXIT_READY}:
                    if symbol not in active_symbols:
                        LOGGER.warning(
                            "Ghost position detected for %s (State: %s). Applying reconciliation cooldown.",
                            symbol,
                            st.state.name,
                        )
                        prev_name = st.state.name
                        min_cooldown = max(1, int(getattr(strategy, "min_cooldown_bars", 1) or 1))
                        existing_cooldown = int(getattr(st, "cooldown_bars_remaining", 0) or 0)
                        cooldown_bars = max(1, min_cooldown, existing_cooldown)
                        # If we thought we were IN_POSITION but broker has nothing, reset to IDLE.
                        # Cooldown is still applied for EXIT_READY to avoid rapid re-entry loops.
                        if st.state == StrategyState.IN_POSITION:
                            st.state = StrategyState.IDLE
                            st.cooldown_bars_remaining = 0
                            applied_state = "IDLE"
                        else:
                            st.state = StrategyState.COOLDOWN
                            st.cooldown_bars_remaining = cooldown_bars
                            applied_state = "RECONCILIATION_COOLDOWN"
                        st.pending_order = False
                        st.last_reason = "RECONCILIATION_COOLDOWN"
                        st.updated_at_utc = datetime.now(timezone.utc)
                        if isinstance(st.metadata, dict):
                            st.metadata["bars_in_trade"] = 0
                        self.store.append_event({
                            "event": "ghost_reconciliation",
                            "symbol": symbol,
                            "previous_state": prev_name,
                            "reason": "broker_position_missing",
                            "applied_state": applied_state,
                            "cooldown_bars": int(cooldown_bars),
                        })
            except Exception:
                continue

    def _reset_strategy_state_for_protected_symbol(self, symbol: str) -> None:
        symbol_upper = str(symbol or "").strip().upper()
        if not symbol_upper:
            return
        reset_states = {"IN_POSITION", "EXIT_READY", "ENTRY_PENDING", "ENTRY_READY", "SETUP"}
        seen_pairs: set[tuple[str, str]] = set()
        for instrument in self.config.get("universe", []) or []:
            instrument_symbol = str(instrument.get("symbol", "")).strip().upper()
            strategy_name = str(instrument.get("strategy", "")).strip()
            if instrument_symbol != symbol_upper:
                continue
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                continue
            pair = (strategy_name, instrument_symbol)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            getter = getattr(strategy, "get_symbol_state", None)
            if not callable(getter):
                continue
            try:
                st = getter(symbol_upper)
            except Exception:
                continue

            state_value = str(getattr(getattr(st, "state", None), "value", ""))
            if state_value not in reset_states:
                continue
            st.state = StrategyState.IDLE
            st.pending_order = False
            st.cooldown_bars_remaining = 0
            st.last_reason = "MANUAL_POSITION_GUARD_PROTECTED"
            if isinstance(st.metadata, dict):
                st.metadata["bars_in_trade"] = 0
                st.metadata["last_manage_bar_time"] = ""

    @staticmethod
    def _compute_atr_from_bars(bars: Any, period: int = 14) -> Optional[float]:
        if bars is None or len(bars) < 2:
            return None
        try:
            frame = bars.tail(max(3, int(period) + 1))
            prev_close = frame["close"].shift(1)
            tr = (frame["high"] - frame["low"]).abs()
            tr2 = (frame["high"] - prev_close).abs()
            tr3 = (frame["low"] - prev_close).abs()
            true_range = tr.combine(tr2, max).combine(tr3, max)
            atr = float(true_range.tail(period).mean())
            if atr > 0:
                return atr
        except Exception:
            return None
        return None

    def _build_manual_decision(self, action: DecisionAction, bars: Any) -> Optional[StrategyDecision]:
        if bars is None or bars.empty:
            return None
        try:
            close_price = float(bars["close"].iloc[-1])
        except Exception:
            return None
        if close_price <= 0:
            return None

        trend_cfg = dict((self.config.get("strategies", {}) or {}).get("trend_regime_sm", {}) or {})
        sl_mult = float(trend_cfg.get("trend_sl_atr_mult", trend_cfg.get("sl_atr_mult", 1.2)) or 1.2)
        atr = self._compute_atr_from_bars(bars=bars, period=int(trend_cfg.get("atr_period", 14) or 14))
        stop_distance = max((atr or 0.0) * sl_mult, close_price * 0.001)

        if action == DecisionAction.BUY:
            sl = close_price - stop_distance
            tp = None
        else:
            sl = close_price + stop_distance
            tp = None

        return StrategyDecision(
            action=action,
            reason="MANUAL_ENTRY",
            strategy="manual_entry",
            confidence=1.0,
            sl=float(sl),
            tp=None,
            signal_bar_time=datetime.now(timezone.utc),
            min_hold_bars=2,
            metadata={
                "manual_entry": True,
                "signal_close": close_price,
                "atr_estimate": atr,
            },
        )

    def _clear_manual_entry_request(self) -> None:
        if self.dashboard_enabled and self.control_channel is not None:
            self.control_channel.clear_manual_entry()

    def _clear_flatten_request(self) -> None:
        if self.dashboard_enabled and self.control_channel is not None:
            self.control_channel.clear_flatten()

    def _consume_manual_entry_request(self, control_state: Dict[str, Any]) -> bool:
        payload = control_state.get("manual_entry")
        if not isinstance(payload, dict):
            return False

        symbol = str(payload.get("symbol", "BTCUSD") or "BTCUSD").strip().upper()
        action_raw = str(payload.get("action", "") or "").strip().upper()
        action = DecisionAction.BUY if action_raw in {"BUY", "LONG"} else DecisionAction.SELL if action_raw in {"SELL", "SHORT"} else None
        if action is None:
            self.store.append_event({"event": "manual_entry_rejected", "symbol": symbol, "reason": "INVALID_ACTION", "action": action_raw})
            self._clear_manual_entry_request()
            return True

        instrument = None
        strategy_name = ""
        for item in self.config.get("universe", []) or []:
            if str(item.get("symbol", "")).strip().upper() == symbol:
                instrument = item
                strategy_name = str(item.get("strategy", "")).strip()
                break
        if instrument is None:
            self.store.append_event({"event": "manual_entry_rejected", "symbol": symbol, "reason": "SYMBOL_NOT_IN_UNIVERSE"})
            self._clear_manual_entry_request()
            return True

        timeframe = str(instrument.get("timeframe", "TIMEFRAME_M1")).strip()
        bars = self.broker.fetch_bars(symbol=symbol, timeframe=timeframe, bars=self.bars_per_request)
        if bars is None or bars.empty:
            self.store.append_event({"event": "manual_entry_rejected", "symbol": symbol, "reason": "MARKET_DATA_MISSING"})
            self._clear_manual_entry_request()
            return True

        positions = self.broker.get_positions(symbol=symbol)
        current_position = positions[0] if positions else None
        decision = self._build_manual_decision(action=action, bars=bars)
        if decision is None:
            self.store.append_event({"event": "manual_entry_rejected", "symbol": symbol, "reason": "DECISION_BUILD_FAILED"})
            self._clear_manual_entry_request()
            return True

        result = self.order_manager.process_decision(
            instrument=instrument,
            decision=decision,
            current_position=current_position,
        )
        strategy = self.strategies.get(strategy_name)
        if strategy is not None:
            apply_result = getattr(strategy, "apply_order_result", None)
            if callable(apply_result):
                try:
                    apply_result(symbol, decision, result)
                except Exception:
                    LOGGER.exception("Failed to sync strategy state after manual entry. symbol=%s strategy=%s", symbol, strategy_name)
        self.store.append_event(
            {
                "event": "manual_entry_dispatched",
                "symbol": symbol,
                "action": action.value,
                "decision": {"sl": decision.sl, "tp": decision.tp},
                "result": result.__dict__ if result is not None else None,
            }
        )
        self._clear_manual_entry_request()
        return True

    def _run_auto_tuning_cycle(self, bars_by_symbol: Dict[str, Any]) -> None:
        if not self.auto_tuning.enabled:
            self._auto_tune_skip_signature = None
            self.state["auto_tuning"] = self.auto_tuning.snapshot()
            return

        self.auto_tuning.ingest_bars(bars_by_symbol)
        result = self.auto_tuning.step(now_ts=time.time())

        if bool(result.get("updated", False)):
            self._auto_tune_skip_signature = None
            requested = dict(result.get("overrides", {})) if isinstance(result.get("overrides"), dict) else {}
            applied = self._apply_runtime_overrides(requested, context="auto_tune_update")
            self.store.append_event(
                {
                    "event": "auto_tune_update",
                    "overrides": applied,
                    "requested_overrides": requested,
                    "metrics": dict(result.get("metrics", {})) if isinstance(result.get("metrics"), dict) else {},
                    "symbol_metrics": dict(result.get("symbol_metrics", {}))
                    if isinstance(result.get("symbol_metrics"), dict)
                    else {},
                }
            )
        else:
            reason = str(result.get("reason", "unknown"))
            details = {k: v for k, v in result.items() if k not in {"updated", "reason"}}
            signature_details = {k: v for k, v in details.items() if k != "timestamp"}
            signature = json.dumps({"reason": reason, "details": signature_details}, sort_keys=True, default=str)
            previous_signature = self._auto_tune_skip_signature
            self._auto_tune_skip_signature = signature
            if reason != "interval_not_elapsed" and signature != previous_signature:
                self.store.append_event(
                    {
                        "event": "auto_tune_skip",
                        "reason": reason,
                        "details": details,
                    }
                )

        self.state["auto_tuning"] = self.auto_tuning.snapshot()

    def run_cycle(self) -> None:
        self._write_watchdog_heartbeat()
        if self._halt_for_broker_fatal():
            return

        self._reload_config_if_changed()
        if self._halt_for_broker_fatal():
            return

        # dashboard control
        ctrl: Dict[str, Any] = {}
        if self.dashboard_enabled and self.control_channel is not None:
            ctrl = self.control_channel.load()
            if ctrl.get("manual_halt"):
                write_desired_state(
                    DESIRED_STATE_STOP,
                    source="runtime",
                    reason="manual_halt_detected",
                    metadata={"trigger": "dashboard_control"},
                )
                self.lifecycle.request_stop("manual_halt")
                return
            if ctrl.get("resume_requested"):
                self.risk_engine.resume()
                ctrl["resume_requested"] = False
                self.control_channel.save(ctrl)
                self.store.append_event(
                    {"event": "runtime_resumed", "risk_guard_halted": bool(self.risk_engine.status().halted)}
                )

        self._run_heartbeat()
        if self._halt_for_broker_fatal():
            return
        if (
            not self._daily_reference_cache
            or (time.time() - self._last_daily_reference_refresh_ts) >= self._daily_reference_refresh_interval_seconds
        ):
            self._refresh_daily_reference_levels()
            if self._halt_for_broker_fatal():
                return

        broker_positions = self.broker.get_positions(symbol=None)
        self._detect_and_record_broker_closed_positions(broker_positions)
        self._reconcile_strategy_states(broker_positions)

        manual_guard_report = self.manual_position_guard.run_cycle(
            positions=broker_positions,
            broker=self.broker,
            now_ts=time.time(),
        )
        protected_symbols = {
            str(item).strip().upper() for item in manual_guard_report.get("protected_symbols", set()) if str(item).strip()
        }
        for item in manual_guard_report.get("events", []):
            event = getattr(item, "event", "")
            payload = dict(getattr(item, "payload", {}) or {})
            if event:
                self.store.append_event({"event": event, **payload})
        for symbol in protected_symbols:
            self._reset_strategy_state_for_protected_symbol(symbol)

        self._protection_blocked_symbols = self._run_pending_protection_cycle(
            broker_positions=broker_positions,
            excluded_symbols=protected_symbols,
        )

        if self.dashboard_enabled and ctrl.get("paused"):
            self._save_runtime_state()
            return
        if self.dashboard_enabled and ctrl.get("flatten_requested"):
            flatten_results = self.broker.close_all_positions(reason="flatten_requested")
            closed_count = sum(1 for item in flatten_results if getattr(item, "ok", False))
            failed_count = max(0, len(flatten_results) - closed_count)
            self.store.append_event(
                {
                    "event": "manual_flatten",
                    "requested": True,
                    "total_positions": len(flatten_results),
                    "closed": closed_count,
                    "failed": failed_count,
                    "results": [item.__dict__ for item in flatten_results],
                }
            )
            if failed_count > 0:
                self.notifier.send_error(
                    f"manual_flatten completed with failures: closed={closed_count} failed={failed_count}"
                )
            elif len(flatten_results) > 0:
                self.notifier.send_trade(f"manual_flatten completed: closed={closed_count}")
            self._clear_flatten_request()
            refreshed_positions = self.broker.get_positions(symbol=None)
            if self._halt_for_broker_fatal():
                return
            self.trailing_guard.drop_closed_positions(refreshed_positions)
            self.state["trailing_profit_guard"] = self.trailing_guard.snapshot()
            self._save_runtime_state()
            return

        if self.dashboard_enabled and self._consume_manual_entry_request(ctrl):
            self._save_runtime_state()
            return

        account = self.broker.account_info()
        if self._halt_for_broker_fatal():
            return
        if not account:
            self.store.append_event({"event": "account_unavailable", "action": "HOLD_SKIP_CYCLE"})
            self._save_runtime_state()
            return
        self.state["account"] = dict(account)

        # risk guard
        risk_reason = self.risk_engine.evaluate_limits(account)
        if risk_reason:
            reason_text = str(risk_reason)
            if reason_text != self._last_risk_guard_reason:
                self.store.append_event({"event": "risk_guard_halt", "reason": reason_text})
                self.notifier.send_error(f"Risk guard halt: {reason_text}")
                self._force_all_active_strategies_cooldown(reason_text)
                self._last_risk_guard_reason = reason_text
            self._save_runtime_state()
            return
        self._last_risk_guard_reason = ""

        risk_status = self.risk_engine.status()
        current_equity = self._finite_optional_float(account.get("equity"))
        if current_equity is None:
            current_equity = self._finite_optional_float(account.get("balance"))
        equity_peak = self._finite_optional_float(risk_status.equity_peak)
        loss_streak = max(0, int(risk_status.consecutive_losses))
        daily_start_equity = self._finite_optional_float(risk_status.daily_start_equity)
        daily_pnl = None
        if current_equity is not None and daily_start_equity is not None:
            daily_pnl = current_equity - daily_start_equity

        self._active_opportunity_candidates = []
        bars_by_symbol: Dict[str, Any] = {}
        for instrument in self.config.get("universe", []) or []:
            symbol = str(instrument.get("symbol", "")).strip()
            symbol_upper = symbol.upper()
            strategy_name = str(instrument.get("strategy", "")).strip()
            timeframe = str(instrument.get("timeframe", "TIMEFRAME_M1")).strip()

            strategy = self.strategies.get(strategy_name)
            if not strategy or not getattr(strategy, "enabled", True):
                continue
            if (
                self.manual_position_guard.block_strategy_for_protected_symbols
                and self.manual_position_guard.enabled
                and symbol_upper in protected_symbols
            ):
                self.store.append_event(
                    {
                        "event": "strategy_blocked_for_protected_symbol",
                        "symbol": symbol,
                        "strategy": strategy_name,
                    }
                )
                continue

            bars = self.broker.fetch_bars(symbol=symbol, timeframe=timeframe, bars=self.bars_per_request)
            if self._halt_for_broker_fatal():
                return
            if bars is None or bars.empty:
                now_utc = datetime.now(timezone.utc)
                now_ts = time.time()
                nasdaq_symbols = [s.upper() for s in (self.config.get("nasdaq_universe", []) or [])]
                is_nasdaq_symbol = symbol_upper in nasdaq_symbols

                # NASDAQ-like symbols are optional/unreliable by session on this setup.
                # Suppress market_data_missing event spam entirely for these symbols.
                if is_nasdaq_symbol:
                    continue

                # Throttle noisy market_data_missing logs for other symbols.
                # - Other symbols: once per 60 sec
                throttle_seconds = 60.0
                next_allowed_ts = float(self._next_market_data_missing_log_ts_by_symbol.get(symbol_upper, 0.0) or 0.0)
                if now_ts >= next_allowed_ts:
                    self.store.append_event({"event": "market_data_missing", "symbol": symbol, "timeframe": timeframe})
                    self._next_market_data_missing_log_ts_by_symbol[symbol_upper] = now_ts + throttle_seconds
                continue
            bars_by_symbol[symbol_upper] = bars

            is_tick_strategy = bool(getattr(strategy, "tick_driven", False))
            should_evaluate, closed_time = self.bar_gate.should_evaluate(symbol=symbol, bars=bars)
            mark_closed_bar = getattr(strategy, "mark_closed_bar", None)
            if should_evaluate and callable(mark_closed_bar):
                mark_closed_bar(symbol, closed_time)
            if not should_evaluate and not is_tick_strategy:
                continue

            positions = self.broker.get_positions(symbol=symbol)
            if self._halt_for_broker_fatal():
                return
            position: Optional[Position] = positions[0] if positions else None

            trailing_signal = None
            if position is not None and symbol_upper not in protected_symbols:
                trailing_signal = self._evaluate_profit_lock_for_position(position)

            opportunity_results = self._scan_and_filter_opportunities(
                symbol=symbol,
                timeframe=timeframe,
                bars=bars,
                position_count=len(broker_positions),
            )

            daily_reference = dict(self._daily_reference_cache.get(symbol_upper, {}))
            context = StrategyEvaluationContext(
                mtf_info={
                    "daily_reference": {
                        "pdh": self._finite_optional_float(daily_reference.get("pdh")),
                        "pdl": self._finite_optional_float(daily_reference.get("pdl")),
                        "updated_at_utc": daily_reference.get("updated_at_utc"),
                        "timeframe": str(daily_reference.get("timeframe", "TIMEFRAME_D1")),
                    }
                },
                equity=current_equity,
                equity_peak=equity_peak,
                loss_streak=loss_streak,
                daily_pnl=daily_pnl,
            )
            auto_tuning = getattr(self, "auto_tuning", None)
            symbol_metrics = getattr(auto_tuning, "symbol_metrics", {})
            if not isinstance(symbol_metrics, dict):
                symbol_metrics = {}
            metrics = symbol_metrics.get(symbol_upper, {})
            if not isinstance(metrics, dict):
                metrics = {}
            context.metadata["metrics"] = dict(metrics)

            if is_tick_strategy:
                now_utc = datetime.now(timezone.utc)
                warmup_seconds = float(self.config.get("general", {}).get("tick_warmup_seconds", 120.0))
                overlap_seconds = float(self.config.get("general", {}).get("tick_fetch_overlap_seconds", 1.0))
                max_ticks = int(self.config.get("general", {}).get("tick_max_per_cycle", 2000))
                last_dt = self._last_tick_fetch_by_symbol.get(symbol_upper)
                if last_dt is None:
                    from_dt = now_utc - timedelta(seconds=max(1.0, warmup_seconds))
                else:
                    from_dt = last_dt - timedelta(seconds=max(0.0, overlap_seconds))
                to_dt = now_utc
                ticks: list[Any] = []
                try:
                    ticks = list(self.broker.fetch_ticks(symbol=symbol, from_dt=from_dt, to_dt=to_dt, max_ticks=max_ticks))
                except Exception as exc:
                    ticks = []
                    self.store.append_event(
                        {
                            "event": "tick_fetch_error",
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "error": str(exc),
                        }
                    )
                if ticks:
                    last_tick_dt = getattr(ticks[-1], "time_utc", None)
                    if isinstance(last_tick_dt, datetime):
                        last_tick_dt = last_tick_dt.astimezone(timezone.utc) if last_tick_dt.tzinfo else last_tick_dt.replace(tzinfo=timezone.utc)
                        prev = self._last_tick_fetch_by_symbol.get(symbol_upper)
                        if prev is None or last_tick_dt > prev:
                            self._last_tick_fetch_by_symbol[symbol_upper] = last_tick_dt
                ingest = getattr(strategy, "ingest_ticks", None)
                if callable(ingest):
                    try:
                        ingest(symbol, ticks)
                    except Exception as exc:
                        self.store.append_event(
                            {
                                "event": "tick_ingest_error",
                                "symbol": symbol,
                                "strategy": strategy_name,
                                "error": str(exc),
                            }
                        )

            decision = strategy.evaluate(symbol=symbol, bars=bars, position=position, context=context)
            if not isinstance(decision.metadata, dict):
                decision.metadata = {}
            if (
                self._opportunity_scanner_drives_entries()
                and position is None
                and decision.action == DecisionAction.HOLD
            ):
                eligible = [item for item in opportunity_results if bool(getattr(item[1], "allow", False))]
                if eligible:
                    best_opportunity, _filter_decision, best_candidate = max(
                        eligible, key=lambda item: float(item[2].get("score", 0.0) or 0.0)
                    )
                    scanner_decision = self._decision_from_opportunity(best_opportunity, best_candidate)
                    if scanner_decision is not None:
                        decision = scanner_decision
                        self.store.append_event(
                            {
                                "event": "v4_opportunity_scanner_entry_selected",
                                "symbol": symbol,
                                "strategy": strategy_name,
                                "action": decision.action.value,
                                "score": best_candidate.get("score"),
                                "fee_adjusted_rr": best_candidate.get("fee_adjusted_rr"),
                                "reason": decision.reason,
                            }
                        )
            raw_decision = decision
            decision = self.exit_engine.choose(
                position=position,
                strategy_decision=decision,
                trailing_signal=trailing_signal,
            )
            if (
                raw_decision.action != DecisionAction.EXIT
                and decision.action == DecisionAction.EXIT
                and str(decision.strategy) == "profit_lock_guard"
            ):
                self.store.append_event(
                    {
                        "event": "exit_engine_override",
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "chosen_strategy": decision.strategy,
                        "reason": decision.reason,
                        "metadata": decision.metadata,
                    }
                )
            if decision.action in {DecisionAction.BUY, DecisionAction.SELL}:
                decision.min_hold_bars = self.execution_churn_guard.enforce_min_hold(
                    decision.min_hold_bars, symbol=symbol
                )

            reason_text = str(decision.reason or "")
            should_log_decision = not (
                decision.action == DecisionAction.HOLD
                and reason_text in {"NO_SWEEP_SETUP", "IDLE"}
            )
            if is_tick_strategy and decision.action == DecisionAction.HOLD:
                if decision.sl is None and decision.tp is None:
                    should_log_decision = should_log_decision and reason_text in {
                        "LSR_TICK_SWEEP_DETECTED",
                        "LSR_TICK_RECLAIM_WINDOW_EXTENDED",
                    }
            if should_log_decision:
                self.store.append_event(
                    {
                        "event": "decision",
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "action": decision.action.value,
                        "reason": decision.reason,
                        "state": self._strategy_state_value(strategy, symbol),
                        "metadata": decision.metadata,
                    }
                )

            if decision.action == DecisionAction.EXIT and position is not None:
                reverse_confirm = self._m5_reverse_confirmed_for_exit(symbol=symbol, position=position)
                exit_quality = self.exit_quality_guard.should_block_exit(
                    position=position,
                    reason=decision.reason,
                    m5_reverse_confirmed=reverse_confirm,
                )
                if not bool(exit_quality.get("allow", True)):
                    self.store.append_event(
                        {
                            "event": "order_skip",
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "reason": "SOFT_EXIT_BLOCKED",
                            "details": exit_quality,
                        }
                    )
                    continue
                retry_reason = str(decision.reason or "")
                if retry_reason.startswith("TREND_REGIME_EXIT:"):
                    retry_reason = "TREND_REGIME_EXIT"

                allow_exit, cooldown_left, attempt_count = self.exit_retry_guard.should_allow(
                    ticket=position.ticket,
                    reason=retry_reason,
                    now_ts=time.time(),
                )
                if not allow_exit:
                    self.store.append_event(
                        {
                            "event": "position_exit_retry_backoff",
                            "symbol": symbol,
                            "ticket": int(position.ticket),
                            "reason": decision.reason,
                            "retry_reason": retry_reason,
                            "cooldown_remaining_seconds": float(cooldown_left),
                            "attempt": int(attempt_count),
                        }
                    )
                    continue
                decision.metadata["exit_attempt_no"] = int(attempt_count) + 1

            if decision.action in {DecisionAction.BUY, DecisionAction.SELL}:
                spread_getter = getattr(self.broker, "get_live_spread", None)
                if callable(spread_getter):
                    current_spread = spread_getter(symbol)
                    max_spread = float(self.config.get("execution", {}).get("max_spread", 60.0))
                    if current_spread is not None and current_spread > max_spread:
                        self.store.append_event(
                            {
                                "event": "order_skip",
                                "symbol": symbol,
                                "strategy": strategy_name,
                                "reason": f"SPREAD_TOO_HIGH:{current_spread}>{max_spread}",
                            }
                        )
                        self._skip_entry_and_cooldown(
                            strategy=strategy,
                            symbol=symbol,
                            strategy_name=strategy_name,
                            reason=f"SPREAD_TOO_HIGH:{current_spread}",
                        )
                        continue

                if symbol_upper in self._protection_blocked_symbols:
                    self.store.append_event(
                        {
                            "event": "order_skip",
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "reason": "PROTECTION_PENDING",
                        }
                    )
                    self._skip_entry_and_cooldown(
                        strategy=strategy,
                        symbol=symbol,
                        strategy_name=strategy_name,
                        reason="PROTECTION_PENDING",
                    )
                    continue
                m5_allow = True
                if self.mtf_confirm.is_symbol_enabled(symbol):
                    confirm_bars = self.broker.fetch_bars(
                        symbol=symbol,
                        timeframe=self.mtf_confirm.confirm_timeframe,
                        bars=self.bars_per_request,
                    )
                    m5_allow = self.mtf_confirm.allow_entry(
                        symbol=symbol,
                        action=decision.action,
                        bars=confirm_bars if confirm_bars is not None else bars,
                    )
                    if not m5_allow:
                        self.store.append_event(
                            {
                                "event": "order_skip",
                                "symbol": symbol,
                                "strategy": strategy_name,
                                "reason": "M5_CONFIRM_BLOCK",
                            }
                        )
                        self._skip_entry_and_cooldown(
                            strategy=strategy,
                            symbol=symbol,
                            strategy_name=strategy_name,
                            reason="M5_CONFIRM_BLOCK",
                        )
                        continue

                quality_report = self.entry_quality_guard.evaluate_entry(
                    symbol=symbol,
                    decision_action=decision.action,
                    decision_metadata=decision.metadata,
                    m5_aligned=bool(m5_allow),
                )
                self.store.append_event(
                    {
                        "event": "entry_quality_score",
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "score": float(quality_report.get("score", 0.0)),
                        "threshold": float(quality_report.get("threshold", 0.0)),
                        "allow": bool(quality_report.get("allow", False)),
                        "risk_mode": quality_report.get("risk_mode"),
                        "features": quality_report.get("features"),
                    }
                )
                if not bool(quality_report.get("allow", True)):
                    self.store.append_event(
                        {
                            "event": "order_skip",
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "reason": "ENTRY_QUALITY_BLOCK",
                            "details": quality_report,
                        }
                    )
                    self._skip_entry_and_cooldown(
                        strategy=strategy,
                        symbol=symbol,
                        strategy_name=strategy_name,
                        reason="ENTRY_QUALITY_BLOCK",
                        details=quality_report,
                    )
                    continue

                constraints = self.broker.get_symbol_constraints(symbol)
                requested_volume = float(decision.volume) if decision.volume is not None else float(
                    instrument.get("volume", self.order_manager.default_volume)
                )
                edge_report = self.cost_edge_guard.evaluate_entry(
                    symbol=symbol,
                    decision_metadata=decision.metadata,
                    requested_volume=requested_volume,
                    constraints=constraints,
                )
                if not bool(edge_report.get("allow", True)):
                    self.store.append_event(
                        {
                            "event": "order_skip",
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "reason": "EDGE_TOO_LOW",
                            "details": edge_report,
                        }
                    )
                    self._skip_entry_and_cooldown(
                        strategy=strategy,
                        symbol=symbol,
                        strategy_name=strategy_name,
                        reason="EDGE_TOO_LOW",
                        details=edge_report,
                    )
                    continue

                is_flip = bool(
                    position is not None
                    and ((position.side == Side.BUY and decision.action == DecisionAction.SELL)
                         or (position.side == Side.SELL and decision.action == DecisionAction.BUY))
                )
                churn_reason = self.execution_churn_guard.should_block_entry(
                    symbol=symbol,
                    now_ts=time.time(),
                    is_flip=is_flip,
                )
                if churn_reason is not None:
                    self.store.append_event(
                        {
                            "event": "order_skip",
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "reason": churn_reason,
                        }
                    )
                    self._skip_entry_and_cooldown(
                        strategy=strategy,
                        symbol=symbol,
                        strategy_name=strategy_name,
                        reason=str(churn_reason),
                    )
                    continue

            # execute trade
            result: Optional[OrderResult]
            try:
                result = self.order_manager.process_decision(
                    instrument=instrument,
                    decision=decision,
                    current_position=position,
                )
            except Exception as exc:
                LOGGER.exception(
                    "Order execution raised exception. symbol=%s strategy=%s action=%s",
                    symbol,
                    strategy_name,
                    decision.action.value,
                )
                result = OrderResult(
                    ok=False,
                    status="ERROR",
                    message=str(exc),
                )
                self.store.append_event(
                    {
                        "event": "order_execution_exception",
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "action": decision.action.value,
                        "reason": decision.reason,
                        "error": str(exc),
                    }
                )
                self.notifier.send_error(
                    f"{symbol} {decision.action.value} EXECUTION EXCEPTION {strategy_name} | {exc}"
                )
            if decision.action in {DecisionAction.BUY, DecisionAction.SELL} and result is not None and result.ok:
                if isinstance(decision.metadata, dict) and bool(decision.metadata.get("v4_opportunity_scanner", False)):
                    self.no_trade_guard.record_executed_trade(decision.metadata.get("opportunity"), now_ts=time.time())
                self.execution_churn_guard.record_entry(symbol=symbol, now_ts=time.time())
                decision.metadata["m5_align"] = bool(m5_allow)
                if getattr(result, "ticket", None) is not None:
                    self.entry_quality_guard.record_entry_context(
                        ticket=result.ticket,
                        symbol=symbol,
                        metadata=decision.metadata,
                    )
            if decision.action == DecisionAction.EXIT and position is not None and result is not None:
                retry_reason = str(decision.reason or "")
                if retry_reason.startswith("TREND_REGIME_EXIT:"):
                    retry_reason = "TREND_REGIME_EXIT"

                retry_report = self.exit_retry_guard.on_attempt(
                    ticket=position.ticket,
                    reason=retry_reason,
                    now_ts=time.time(),
                    success=bool(result.ok),
                )
                if not result.ok:
                    retry_report["decision_reason"] = decision.reason
                    self.store.append_event({"event": "position_exit_retry_backoff", **retry_report})
            apply_result = getattr(strategy, "apply_order_result", None)
            apply_result_ok = False
            if callable(apply_result):
                try:
                    apply_result(symbol, decision, result)
                    apply_result_ok = True
                except Exception:
                    LOGGER.exception("Failed to sync strategy state after order result. symbol=%s strategy=%s", symbol, strategy_name)
            if (
                decision.action in {DecisionAction.BUY, DecisionAction.SELL}
                and (result is None or not bool(getattr(result, "ok", False)))
                and not apply_result_ok
            ):
                fallback_reason = "ENTRY_FAILED_SAFE_COOLDOWN"
                if result is not None and str(result.status or "").strip():
                    fallback_reason = f"ENTRY_FAILED_SAFE_COOLDOWN:{result.status}"
                self._force_strategy_cooldown(
                    strategy=strategy,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    reason=fallback_reason,
                    details={
                        "decision_reason": decision.reason,
                        "result_message": str(getattr(result, "message", "") or ""),
                    },
                )

        self._write_v4_opportunity_reports()
        self._run_auto_tuning_cycle(bars_by_symbol)
        self.state["entry_quality_guard"] = self.entry_quality_guard.snapshot()
        self.state["cost_edge_guard"] = self.cost_edge_guard.snapshot()
        if hasattr(self, "no_trade_guard"):
            self.state["no_trade_bias_guard"] = self.no_trade_guard.snapshot()
        self.state["exit_retry_guard"] = self.exit_retry_guard.snapshot()

        # snapshot strategy + runtime state
        self._save_runtime_state()

        if self.mode == BotMode.BACKTEST:
            stepped = self.broker.step()
            if not stepped:
                self.store.append_event({"event": "backtest_complete"})
                self.lifecycle.request_stop("backtest_data_exhausted")

    def run(self, once: bool = False) -> int:
        if self.mode == BotMode.LIVE:
            validation_cfg = self.config.get("validation", {}) or {}
            report_path = str(validation_cfg.get("report_path", ""))
            require_oos = bool(validation_cfg.get("require_oos_pass", False))
            if require_oos and report_path:
                ok, reason = check_live_readiness(report_path)
                if not ok:
                    self.notifier.send_error(f"Live readiness failed: {reason}")
                    return 2

        # Signal watchdog we are alive before potentially slow connection
        self._write_watchdog_heartbeat()

        if not self.broker.connect():
            fatal_reason = self._broker_fatal_reason()
            if fatal_reason:
                self.notifier.send_error(f"Startup blocked by broker session guard: {fatal_reason}")
                return 3
            self.notifier.send_error("Startup failed: broker connection failed")
            return 2

        LOGGER.info(
            "Bot started. mode=%s dry_run=%s symbols=%d",
            self.mode.value,
            self.dry_run,
            len(self.config.get("universe", []) or []),
        )

        try:
            while not self.lifecycle.stop_requested:
                started = time.time()
                self.run_cycle()
                if once:
                    break
                elapsed = time.time() - started
                latency_threshold = self.poll_seconds + 0.5
                if elapsed > latency_threshold:
                    self.store.append_event(
                        {
                            "event": "cycle_latency_warning",
                            "elapsed_seconds": float(elapsed),
                            "threshold_seconds": float(latency_threshold),
                        }
                    )
                time.sleep(max(0.05, self.poll_seconds - elapsed))
        except KeyboardInterrupt:
            self.lifecycle.request_stop("keyboard_interrupt")
        except Exception:
            LOGGER.exception("Fatal runtime exception")
            self.notifier.send_error("Fatal runtime exception occurred. Check logs/events.")
            return 1
        finally:
            reason = self.lifecycle.stop_reason or "normal_exit"
            self.lifecycle.execute_shutdown(reason)

        if str(self.lifecycle.stop_reason or "").startswith("broker_fatal:"):
            return 3
        return 0
