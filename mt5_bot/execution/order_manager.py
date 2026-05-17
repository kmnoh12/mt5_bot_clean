from __future__ import annotations

import logging
import math
import time
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from brokers.base import BrokerGateway
from core.models import DecisionAction, OrderIntent, OrderResult, Position, Side, StrategyDecision, SymbolConstraints
from core.risk_model import FeeAwareRiskInput, FeeAwareRiskModel
from execution.exit_planner import InitialExitPlanner
from execution.position_sizer import NetRiskPositionSizer, NetRiskPositionSizeInput, SymbolVolumeSpec
from execution.risk_manager import RiskEngine
from storage.json_store import JsonStore


LOGGER = logging.getLogger(__name__)


class OrderManager:
    def __init__(
        self,
        broker: BrokerGateway,
        store: JsonStore,
        notifier: Any,
        execution_cfg: Dict[str, Any],
        risk_engine: RiskEngine,
        dry_run: bool,
    ) -> None:
        self.broker = broker
        self.store = store
        self.notifier = notifier
        self.execution_cfg = execution_cfg or {}
        self.risk_engine = risk_engine
        self.dry_run = bool(dry_run)

        self.comment_prefix = str(self.execution_cfg.get("comment_prefix", "quant_bot"))
        self.allow_opposite_position = bool(self.execution_cfg.get("allow_opposite_position", False))
        self.default_volume = max(0.001, float(self.execution_cfg.get("default_volume", 0.01)))
        self.max_positions_per_symbol = max(1, int(self.execution_cfg.get("max_positions_per_symbol", 1)))
        self.fee_aware_risk_model = FeeAwareRiskModel()
        self.net_risk_position_sizer = NetRiskPositionSizer(self.fee_aware_risk_model)
        self.initial_exit_planner = InitialExitPlanner()
        self.daily_bleed_guard = None
        self.on_position_closed = None
        self._last_trade_event_ticket: Optional[int] = None
        self._last_trade_event_type: Optional[str] = None

    @staticmethod
    def _trade_event_path() -> Path:
        return Path(__file__).resolve().parents[1] / "runtime" / "trade_event.json"

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        for attempt in range(1, 8 + 1):
            try:
                with tmp_path.open("w", encoding="utf-8") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(str(tmp_path), str(path))
                return
            except Exception as exc:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                if attempt >= 8 and not isinstance(exc, (PermissionError, OSError)):
                    raise
                time.sleep(0.05 * attempt)

    def _should_emit_trade_event(self, *, event_type: str, ticket: Optional[int]) -> bool:
        ticket_int = int(ticket) if ticket is not None else None
        if ticket_int is None:
            return False
        if self._last_trade_event_ticket == ticket_int and self._last_trade_event_type == event_type:
            return False
        self._last_trade_event_ticket = ticket_int
        self._last_trade_event_type = event_type
        return True

    @staticmethod
    def _is_success_trade_result(result: Optional[OrderResult]) -> bool:
        if result is None or not result.ok:
            return False
        try:
            return int(result.retcode) == 10009
        except Exception:
            return False

    @staticmethod
    def _safe_value(value: Any) -> Any:
        return None if value is None else value

    def _emit_trade_event(
        self,
        *,
        event_type: str,
        symbol: str,
        side: str,
        volume: float,
        price: Optional[float],
        sl: Optional[float],
        tp: Optional[float],
        strategy_name: str,
        expected_pnl_usd: Optional[float],
        ticket: Optional[int],
    ) -> None:
        if not self._should_emit_trade_event(event_type=event_type, ticket=ticket):
            return
        payload = {
            "event_type": event_type,
            "symbol": str(symbol or ""),
            "side": str(side or "").upper(),
            "volume": self._safe_value(volume),
            "price": self._safe_value(price),
            "sl": self._safe_value(sl),
            "tp": self._safe_value(tp),
            "expected_pnl_usd": self._safe_value(expected_pnl_usd),
            "strategy_name": str(strategy_name or ""),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "mt5_order_manager",
        }
        try:
            self._atomic_write_json(self._trade_event_path(), payload)
        except Exception:
            LOGGER.exception("Failed to emit trade event: %s", payload.get("event_type"))

    def _notify_trade(self, message: str) -> None:
        if self.notifier is not None:
            self.notifier.send_trade(message)

    def _notify_error(self, message: str) -> None:
        if self.notifier is not None:
            self.notifier.send_error(message)

    def _same_side(self, position: Position, decision: StrategyDecision) -> bool:
        if decision.action == DecisionAction.BUY and position.side == Side.BUY:
            return True
        if decision.action == DecisionAction.SELL and position.side == Side.SELL:
            return True
        return False

    def _requested_volume(
        self,
        decision: StrategyDecision,
        instrument: Dict[str, Any],
        constraints: Optional[SymbolConstraints] = None,
    ) -> float:
        if decision.volume is not None:
            return max(0.001, float(decision.volume))

        scale = float(decision.metadata.get("volume_scale", 1.0) or 1.0)
        raw = float(instrument.get("volume", self.default_volume)) * max(0.1, scale)

        if constraints is not None:
            try:
                min_volume = float(getattr(constraints, "min_volume", 0.0) or 0.0)
            except (TypeError, ValueError):
                min_volume = 0.0
            if min_volume > 0:
                raw = max(raw, min_volume)

        return max(0.001, raw)

    def _dry_result(self, status: str, decision: StrategyDecision) -> OrderResult:
        return OrderResult(ok=True, status=status, message=f"dry_run:{decision.reason}")

    @staticmethod
    def _safe_upper_text(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _quantize_by_point(value: float, point: float) -> float:
        value_f = float(value)
        point_f = float(point)
        if point_f <= 0:
            return value_f
        return round(value_f / point_f) * point_f

    def _entry_metadata_with_spread_snapshot(
        self,
        *,
        symbol: str,
        decision: StrategyDecision,
        plan_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = {**decision.metadata, **plan_meta}

        max_spread = self._finite_float(self.execution_cfg.get("max_spread"))
        if max_spread is not None:
            metadata.setdefault("max_spread_points", max_spread)

        if any(key in metadata for key in ("spread_points", "current_spread", "current_spread_points", "spread")):
            return metadata

        spread_getter = getattr(self.broker, "get_live_spread", None)
        if not callable(spread_getter):
            return metadata

        try:
            spread = self._finite_float(spread_getter(symbol))
        except Exception:
            return metadata
        if spread is None:
            return metadata

        metadata["spread_points"] = spread
        metadata["current_spread"] = spread
        metadata["spread_snapshot_source"] = "broker.get_live_spread"
        return metadata

    def _adjust_stops_for_constraints(
        self,
        *,
        intent: OrderIntent,
        constraints: SymbolConstraints,
        symbol: str,
        min_distance_points_override: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        sl = self._finite_float(intent.sl)
        if sl is None:
            return None

        point = float(constraints.point or 0.0)
        if point <= 0:
            return None

        anchor = self._latest_symbol_price(symbol)
        if anchor is None:
            anchor = self._finite_float(intent.metadata.get("signal_close"))
        if anchor is None:
            anchor = self._finite_float(intent.metadata.get("entry_price"))
        if anchor is None:
            return None

        stop_points = float(constraints.trade_stops_level or 0.0)
        freeze_points = float(constraints.trade_freeze_level or 0.0)
        cfg_floor_points = self._finite_float(self.execution_cfg.get("invalid_stops_floor_points"))
        if cfg_floor_points is None:
            cfg_floor_points = 30.0
        min_distance_points = max(0.0, stop_points, freeze_points, float(cfg_floor_points))
        if min_distance_points_override is not None:
            try:
                min_distance_points = max(min_distance_points, float(min_distance_points_override))
            except (TypeError, ValueError):
                pass
        if min_distance_points <= 0:
            return None

        min_distance = min_distance_points * point
        if min_distance <= 0:
            return None

        tp = self._finite_float(intent.tp)

        original_sl = sl
        original_tp = tp
        new_sl = sl
        new_tp = tp

        if intent.side == Side.BUY:
            sl_min = anchor - min_distance
            tp_min = anchor + min_distance
            if sl >= anchor:
                new_sl = sl_min
            elif (anchor - sl) < min_distance:
                new_sl = sl_min

            if new_tp is not None:
                if new_tp <= anchor:
                    new_tp = tp_min
                elif (new_tp - anchor) < min_distance:
                    new_tp = tp_min
        else:
            sl_max = anchor + min_distance
            tp_max = anchor - min_distance
            if sl <= anchor:
                new_sl = sl_max
            elif (sl - anchor) < min_distance:
                new_sl = sl_max

            if new_tp is not None:
                if new_tp >= anchor:
                    new_tp = tp_max
                elif (anchor - new_tp) < min_distance:
                    new_tp = tp_max

        new_sl = self._quantize_by_point(new_sl, point)
        if new_tp is not None:
            new_tp = self._quantize_by_point(new_tp, point)

        if not math.isfinite(new_sl):
            return None
        if intent.side == Side.BUY and new_sl >= anchor:
            return None
        if intent.side == Side.SELL and new_sl <= anchor:
            return None

        if new_tp is not None and not math.isfinite(new_tp):
            return None
        if intent.side == Side.BUY and new_tp is not None and new_tp <= anchor:
            return None
        if intent.side == Side.SELL and new_tp is not None and new_tp >= anchor:
            return None

        changed = False
        if abs(new_sl - original_sl) > 0:
            changed = True
        if original_tp is None and new_tp is not None:
            changed = True
        if original_tp is not None and new_tp is not None and abs(new_tp - original_tp) > 0:
            changed = True

        if not changed:
            return None

        intent.sl = float(new_sl)
        if original_tp is None and new_tp is not None:
            intent.tp = float(new_tp)
        elif original_tp is not None and new_tp is not None:
            intent.tp = float(new_tp)

        intent.metadata["stop_adjustment_applied"] = True
        meta = {
            "anchor": float(anchor),
            "point": float(point),
            "min_distance_points": float(min_distance_points),
            "min_distance": float(min_distance),
            "trade_stops_level": float(stop_points),
            "trade_freeze_level": float(freeze_points),
            "original": {
                "sl": original_sl,
                "tp": original_tp,
            },
            "adjusted": {
                "sl": float(new_sl),
                "tp": None if new_tp is None else float(new_tp),
            },
        }
        return meta

    def _expected_loss_cap_for_symbol(self, symbol: str) -> Optional[float]:
        caps = []
        max_loss_map = self.execution_cfg.get("max_expected_loss_usd_by_symbol", {}) or {}
        if isinstance(max_loss_map, dict):
            for key in (symbol, str(symbol or "").upper()):
                raw_cap = max_loss_map.get(key)
                cap = self._finite_float(raw_cap)
                if cap is not None and cap > 0:
                    caps.append(float(cap))
                    break
        fee_cfg = self._fee_aware_fixed_risk_config()
        if fee_cfg is not None:
            cap = self._finite_float(fee_cfg.get("hard_max_net_loss_usd"))
            if cap is not None and cap > 0:
                caps.append(float(cap))
        if not caps:
            return None
        return float(min(caps))

    def _estimate_intent_loss_usd(self, intent: OrderIntent, constraints: SymbolConstraints) -> Optional[float]:
        metadata = dict(intent.metadata or {})
        entry = self._finite_float(metadata.get("signal_close"))
        if entry is None:
            entry = self._finite_float(metadata.get("entry_price"))
        if entry is None:
            entry = self._finite_float(metadata.get("anchor"))
        sl = self._finite_float(intent.sl)
        if entry is None or sl is None:
            return None

        risk_model = metadata.get("risk_model")
        risk_model = dict(risk_model) if isinstance(risk_model, dict) else {}
        tick_size = self._finite_float(risk_model.get("tick_size"))
        if tick_size is None:
            tick_size = self._constraint_tick_size(constraints)
        tick_value = self._finite_float(risk_model.get("tick_value"))
        if tick_value is None:
            tick_value = self._constraint_tick_value(constraints, self._fee_aware_fixed_risk_config() or {})
        spread = self._finite_float(risk_model.get("spread"))
        if spread is None:
            spread_points = self._finite_float(metadata.get("spread_points"))
            spread = (spread_points or 0.0) * float(tick_size)
        commission = self._finite_float(risk_model.get("commission_per_lot"))
        if commission is None:
            commission = self._finite_float(metadata.get("commission_per_lot")) or 0.0
        slippage_points = self._finite_float(risk_model.get("expected_slippage_points"))
        if slippage_points is None:
            slippage_points = self._finite_float(metadata.get("expected_slippage_points")) or 0.0

        try:
            estimate = self.fee_aware_risk_model.estimate(
                FeeAwareRiskInput(
                    symbol=str(intent.symbol),
                    entry_price=float(entry),
                    stop_price=float(sl),
                    direction="long" if intent.side == Side.BUY else "short",
                    lot=float(intent.volume),
                    spread=max(0.0, float(spread or 0.0)),
                    commission_per_lot=max(0.0, float(commission or 0.0)),
                    expected_slippage_points=max(0.0, float(slippage_points or 0.0)),
                    tick_size=max(1e-12, float(tick_size)),
                    tick_value=max(1e-12, float(tick_value)),
                    contract_size=float(getattr(constraints, "contract_size", 1.0) or 1.0),
                    take_profit_price=self._finite_float(intent.tp),
                    hard_max_net_loss_usd=self._expected_loss_cap_for_symbol(str(intent.symbol)),
                )
            )
            return float(estimate.estimated_net_loss_usd)
        except Exception:
            pass

        previous_loss = self._finite_float(metadata.get("estimated_net_loss"))
        if previous_loss is None:
            expected_pnl = self._finite_float(metadata.get("expected_pnl_usd"))
            previous_loss = abs(float(expected_pnl)) if expected_pnl is not None else None
        previous_risk = self._finite_float(metadata.get("risk_per_unit"))
        if previous_loss is None or previous_risk is None or previous_risk <= 0:
            return None
        adjusted_risk = abs(float(entry) - float(sl))
        return float(previous_loss) * (float(adjusted_risk) / float(previous_risk))

    def _block_stop_adjustment_if_over_cap(
        self,
        *,
        intent: OrderIntent,
        constraints: SymbolConstraints,
        symbol: str,
        strategy: str,
        phase: str,
        adjustment: Dict[str, Any],
    ) -> Optional[OrderResult]:
        cap = self._expected_loss_cap_for_symbol(symbol)
        if cap is None:
            return None
        adjusted_loss = self._estimate_intent_loss_usd(intent, constraints)
        if adjusted_loss is None or adjusted_loss <= cap + 1e-12:
            return None
        details = {
            "expected_loss_usd": float(adjusted_loss),
            "max_expected_loss_usd": float(cap),
            "adjustment": dict(adjustment or {}),
        }
        self.store.append_event(
            {
                "event": "order_skip",
                "symbol": symbol,
                "strategy": strategy,
                "reason": "EXPECTED_LOSS_CAP_AFTER_STOP_ADJUSTMENT",
                "details": details,
            }
        )
        self.store.append_event(
            {
                "event": "invalid_stops_adjustment_blocked_by_cap",
                "symbol": symbol,
                "strategy": strategy,
                "phase": phase,
                "details": details,
            }
        )
        return OrderResult(
            ok=False,
            status="EXPECTED_LOSS_CAP_AFTER_STOP_ADJUSTMENT",
            message=f"adjusted stop loss {adjusted_loss:.2f} exceeds cap {cap:.2f}",
            raw=details,
        )

    @staticmethod
    def _volume_precision(step: float) -> int:
        text = f"{step:.12f}".rstrip("0")
        return len(text.split(".")[1]) if "." in text else 0

    @classmethod
    def _quantize_volume_floor(cls, raw_volume: float, min_volume: float, step: float) -> Optional[float]:
        if raw_volume < (min_volume - 1e-12):
            return None
        units = math.floor(((raw_volume - min_volume) / step) + 1e-12)
        quantized = min_volume + (units * step)
        return round(max(min_volume, quantized), cls._volume_precision(step))

    def _resolve_partial_exit_volume(
        self,
        *,
        symbol: str,
        position_volume: float,
        requested_volume: float,
    ) -> Optional[float]:
        constraints = self.broker.get_symbol_constraints(symbol) or SymbolConstraints()
        step = float(constraints.volume_step) if float(constraints.volume_step or 0.0) > 0 else 0.01
        min_volume = max(step, float(constraints.min_volume or step))
        tolerance = max(1e-9, step * 0.25)

        total_volume = max(0.0, float(position_volume))
        if total_volume <= (min_volume + tolerance):
            return None

        clamped_request = min(total_volume, max(0.0, float(requested_volume)))
        quantized = self._quantize_volume_floor(clamped_request, min_volume=min_volume, step=step)
        if quantized is None:
            return None
        if quantized >= (total_volume - tolerance):
            return None
        return float(quantized)

    @classmethod
    def _profit_currency_for_symbol(cls, symbol: str, constraints: SymbolConstraints) -> str:
        for candidate in (getattr(constraints, "profit_currency", ""), getattr(constraints, "quote_currency", "")):
            text = cls._safe_upper_text(candidate)
            if text:
                return text

        compact = "".join(ch for ch in cls._safe_upper_text(symbol) if ch.isalpha())
        if len(compact) >= 6:
            tail = compact[-3:]
            if tail in {"USD", "EUR", "JPY", "GBP", "KRW", "CHF", "CAD", "AUD", "NZD"}:
                return tail
        if compact.endswith(("GOLD", "SILVER", "XAU", "XAG", "BTC", "ETH")):
            return "USD"
        return ""

    def _latest_symbol_price(self, symbol: str) -> Optional[float]:
        getter = getattr(self.broker, "get_latest_price", None)
        if callable(getter):
            try:
                value = float(getter(symbol))
            except (TypeError, ValueError):
                value = 0.0
            if value > 0 and math.isfinite(value):
                return value

        bars = self.broker.fetch_bars(symbol=symbol, timeframe="TIMEFRAME_M1", bars=2)
        if bars is None or bars.empty or "close" not in bars:
            return None
        try:
            value = float(bars["close"].iloc[-1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if value > 0 and math.isfinite(value):
            return value
        return None

    def _resolve_fx_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        base = self._safe_upper_text(from_currency)
        quote = self._safe_upper_text(to_currency)
        if not base or not quote:
            return None
        if base == quote:
            return 1.0

        direct_symbol = f"{base}{quote}"
        direct = self._latest_symbol_price(direct_symbol)
        if direct is not None and direct > 0:
            return direct

        inverse_symbol = f"{quote}{base}"
        inverse = self._latest_symbol_price(inverse_symbol)
        if inverse is not None and inverse > 0:
            return 1.0 / inverse
        return None

    @staticmethod
    def _to_dict(result: Optional[OrderResult]) -> Dict[str, Any]:
        return result.__dict__ if result is not None else {}

    def _estimate_close_pnl(self, position: Position, result: OrderResult) -> Optional[float]:
        if result.pnl is not None:
            return float(result.pnl)
        if result.filled_price is None:
            return None

        constraints = self.broker.get_symbol_constraints(position.symbol)
        contract_size = constraints.contract_size if constraints is not None else 1.0
        direction = 1.0 if position.side == Side.BUY else -1.0
        gross = (float(result.filled_price) - float(position.price_open)) * direction * float(position.volume) * float(contract_size)
        metadata = position.metadata if isinstance(position.metadata, dict) else {}
        swap = float(metadata.get("swap", 0.0) or 0.0)
        commission = float(metadata.get("commission", 0.0) or 0.0)
        pnl = gross + swap + commission
        if math.isfinite(pnl):
            return pnl
        return None

    @staticmethod
    def _result_meta(result: Optional[OrderResult]) -> Dict[str, Any]:
        raw = getattr(result, "raw", None)
        if not isinstance(raw, dict):
            return {}
        meta = raw.get("_meta")
        if isinstance(meta, dict):
            return dict(meta)
        return {}

    def _emit_broker_request_events(
        self,
        *,
        symbol: str,
        strategy: str,
        phase: str,
        result: Optional[OrderResult],
    ) -> None:
        meta = self._result_meta(result)
        if not meta:
            return
        if bool(meta.get("comment_sanitized_changed", False)):
            self.store.append_event(
                {
                    "event": "broker_request_sanitized_comment",
                    "symbol": symbol,
                    "strategy": strategy,
                    "phase": phase,
                    "comment_original": meta.get("comment_original"),
                    "comment_sanitized": meta.get("comment_sanitized"),
                }
            )
        if bool(meta.get("retried_without_comment", False)):
            self.store.append_event(
                {
                    "event": "broker_request_retry_without_comment",
                    "symbol": symbol,
                    "strategy": strategy,
                    "phase": phase,
                    "retry_success": bool(meta.get("retry_success", False)),
                    "comment_original": meta.get("comment_original"),
                }
            )

    def _maybe_modify_protection(self, symbol: str, strategy: str, decision: StrategyDecision, current_position: Optional[Position]) -> Optional[OrderResult]:
        if current_position is None:
            return None
        if decision.sl is None and decision.tp is None:
            return None

        existing_sl = float(current_position.sl) if current_position.sl is not None else None
        existing_tp = float(current_position.tp) if current_position.tp is not None else None

        new_sl = float(decision.sl) if decision.sl is not None else existing_sl
        new_tp = float(decision.tp) if decision.tp is not None else existing_tp

        constraints = self.broker.get_symbol_constraints(symbol)
        point = float(getattr(constraints, "point", 0.0) or 0.0)
        if point > 0:
            if new_sl is not None:
                new_sl = self._quantize_by_point(new_sl, point)
            if new_tp is not None:
                new_tp = self._quantize_by_point(new_tp, point)

        if new_sl == existing_sl and new_tp == existing_tp:
            return None

        if self.dry_run:
            result = self._dry_result("DRY_MODIFY", decision)
        else:
            result = self.broker.modify_position_sl_tp(
                position=current_position,
                sl=new_sl,
                tp=new_tp,
                reason=f"{strategy}:{decision.reason}",
            )
        if self._is_success_trade_result(result):
            changed_sl = new_sl != existing_sl
            changed_tp = new_tp != existing_tp
            if changed_sl:
                self._emit_trade_event(
                    event_type="SL_UPDATE",
                    symbol=str(symbol),
                    side=current_position.side.value,
                    volume=float(current_position.volume),
                    price=float(current_position.price_open),
                    sl=new_sl,
                    tp=new_tp,
                    strategy_name=strategy,
                    expected_pnl_usd=None,
                    ticket=current_position.ticket,
                )
            if changed_tp:
                self._emit_trade_event(
                    event_type="TP_UPDATE",
                    symbol=str(symbol),
                    side=current_position.side.value,
                    volume=float(current_position.volume),
                    price=float(current_position.price_open),
                    sl=new_sl,
                    tp=new_tp,
                    strategy_name=strategy,
                    expected_pnl_usd=None,
                    ticket=current_position.ticket,
                )

        self.store.append_event(
            {
                "event": "position_modify",
                "symbol": symbol,
                "strategy": strategy,
                "reason": decision.reason,
                "sl": new_sl,
                "tp": new_tp,
                "result": result.__dict__,
            }
        )
        return result

    def _guard_open_permission(
        self,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        setup_key: Optional[str] = None,
    ) -> Optional[str]:
        account = self.broker.account_info()
        allowed, reason = self.risk_engine.can_trade(account)
        if not allowed:
            return reason
        guard = getattr(self, "daily_bleed_guard", None)
        if guard is not None:
            block = guard.should_block_entry(
                symbol=str(symbol or ""),
                now_ts=time.time(),
                direction=direction,
                setup_key=setup_key,
            )
            if block:
                return block
        return None

    def _fee_aware_fixed_risk_config(self) -> Optional[Dict[str, Any]]:
        cfg = self.execution_cfg.get("fee_aware_fixed_risk")
        if not isinstance(cfg, dict):
            return None
        if not bool(cfg.get("enabled", False)):
            return None
        return cfg

    @staticmethod
    def _constraint_tick_size(constraints: SymbolConstraints) -> float:
        point = float(getattr(constraints, "point", 0.0) or 0.0)
        return point if point > 0 else 0.0001

    @classmethod
    def _constraint_tick_value(cls, constraints: SymbolConstraints, cfg: Dict[str, Any]) -> float:
        configured = cls._finite_float(cfg.get("tick_value"))
        if configured is not None and configured > 0:
            return configured
        tick_size = cls._constraint_tick_size(constraints)
        contract_size = float(getattr(constraints, "contract_size", 1.0) or 1.0)
        return max(1e-12, tick_size * contract_size)

    def _fee_aware_cost_values(self, decision: StrategyDecision, constraints: SymbolConstraints, cfg: Dict[str, Any]) -> Dict[str, float]:
        metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
        spread = self._finite_float(metadata.get("spread"))
        if spread is None:
            spread_points = self._finite_float(metadata.get("spread_points"))
            if spread_points is None:
                spread_points = self._finite_float(cfg.get("spread_points"))
            spread = (spread_points or 0.0) * self._constraint_tick_size(constraints)
        commission = self._finite_float(metadata.get("commission_per_lot"))
        if commission is None:
            commission = self._finite_float(cfg.get("commission_per_lot")) or 0.0
        slippage_points = self._finite_float(metadata.get("expected_slippage_points"))
        if slippage_points is None:
            slippage_points = self._finite_float(cfg.get("expected_slippage_points")) or 0.0
        return {
            "spread": max(0.0, float(spread or 0.0)),
            "commission_per_lot": max(0.0, float(commission or 0.0)),
            "expected_slippage_points": max(0.0, float(slippage_points or 0.0)),
        }

    def _fee_aware_size_and_plan_entry(
        self,
        *,
        symbol: str,
        side: Side,
        decision: StrategyDecision,
        constraints: SymbolConstraints,
        entry_price: Optional[float],
        cfg: Dict[str, Any],
    ) -> tuple[Optional[float], Optional[str], Dict[str, Any], Optional[float], Optional[float]]:
        entry = self._finite_float(entry_price)
        sl = self._finite_float(decision.sl)
        if entry is None or sl is None:
            return None, "FEE_AWARE_INVALID_ENTRY_OR_SL", {}, decision.sl, decision.tp
        direction = "long" if side == Side.BUY else "short"
        target_loss = self._finite_float(cfg.get("target_net_loss_usd")) or 1.0
        hard_max = self._finite_float(cfg.get("hard_max_net_loss_usd")) or 1.25
        costs = self._fee_aware_cost_values(decision, constraints, cfg)
        tick_size = self._constraint_tick_size(constraints)
        tick_value = self._constraint_tick_value(constraints, cfg)
        contract_size = float(getattr(constraints, "contract_size", 1.0) or 1.0)
        spec = SymbolVolumeSpec(
            volume_min=float(getattr(constraints, "min_volume", 0.01) or 0.01),
            volume_step=float(getattr(constraints, "volume_step", 0.01) or 0.01),
            volume_max=float(getattr(constraints, "max_volume", 100.0) or 100.0),
            tick_size=tick_size,
            tick_value=tick_value,
            contract_size=contract_size,
        )
        try:
            size_result = self.net_risk_position_sizer.size(
                NetRiskPositionSizeInput(
                    symbol=symbol,
                    target_net_loss_usd=target_loss,
                    hard_max_net_loss_usd=hard_max,
                    entry_price=entry,
                    stop_price=sl,
                    direction=direction,
                    symbol_spec=spec,
                    spread=costs["spread"],
                    commission_per_lot=costs["commission_per_lot"],
                    expected_slippage_points=costs["expected_slippage_points"],
                )
            )
        except Exception as exc:
            return None, f"FEE_AWARE_SIZE_ERROR:{exc}", {}, decision.sl, decision.tp
        if not size_result.passed or size_result.recommended_lot is None:
            meta = {
                "fee_aware": True,
                "failure_reason": size_result.failure_reason,
                "estimated_net_loss": size_result.estimated_net_loss,
            }
            return None, str(size_result.failure_reason or "FEE_AWARE_SIZE_FAILED"), meta, decision.sl, decision.tp
        lot = float(size_result.recommended_lot)
        risk_result = self.fee_aware_risk_model.estimate(
            FeeAwareRiskInput(
                symbol=symbol,
                entry_price=entry,
                stop_price=sl,
                direction=direction,
                lot=lot,
                spread=costs["spread"],
                commission_per_lot=costs["commission_per_lot"],
                expected_slippage_points=costs["expected_slippage_points"],
                tick_size=tick_size,
                tick_value=tick_value,
                contract_size=contract_size,
                take_profit_price=self._finite_float(decision.tp),
                target_net_loss_usd=target_loss,
                hard_max_net_loss_usd=hard_max,
            )
        )
        min_rr = self._finite_float(cfg.get("min_reward_to_net_risk_ratio")) or 3.0
        min_tp_profit = self._finite_float(cfg.get("min_tp_net_profit_usd")) or 3.0
        preferred_tp_profit = self._finite_float(cfg.get("preferred_tp_net_profit_usd")) or 5.0
        target_reference = self._finite_float(decision.metadata.get("target_reference_price")) if isinstance(decision.metadata, dict) else None
        if target_reference is None:
            target_reference = self._finite_float(decision.tp)
        plan = self.initial_exit_planner.plan(
            direction=side.value,
            entry_price=entry,
            invalidation_price=sl,
            target_reference_price=target_reference,
            position_size=lot,
            contract_size=contract_size,
            estimated_round_trip_cost=risk_result.estimated_cost_usd,
            min_reward_to_net_risk_ratio=min_rr,
            hard_max_loss=hard_max,
            min_tp_net_profit=min_tp_profit,
            preferred_tp_net_profit=preferred_tp_profit,
            symbol_spec=constraints,
        )
        if not plan.passed:
            return None, f"INITIAL_EXIT_PLAN_FAILED:{plan.reason}", {"fee_aware": True, "exit_plan": plan.__dict__}, decision.sl, decision.tp
        meta = {
            "fee_aware": True,
            "volume_source": "fee_aware_fixed_risk",
            "expected_pnl_usd": -float(plan.expected_net_loss_at_sl),
            "estimated_net_loss": float(plan.expected_net_loss_at_sl),
            "estimated_net_profit_at_tp": float(plan.expected_net_profit_at_tp),
            "fee_adjusted_rr": float(plan.fee_adjusted_rr),
            "target_net_loss_usd": float(target_loss),
            "hard_max_net_loss_usd": float(hard_max),
            "risk_model": risk_result.__dict__,
        }
        return lot, None, meta, plan.sl_price, plan.tp_price

    def process_decision(
        self,
        instrument: Dict[str, Any],
        decision: StrategyDecision,
        current_position: Optional[Position],
    ) -> Optional[OrderResult]:
        symbol = str(instrument.get("symbol", "")).strip()
        strategy = str(decision.strategy)

        if decision.action == DecisionAction.HOLD:
            return self._maybe_modify_protection(symbol=symbol, strategy=strategy, decision=decision, current_position=current_position)

        if decision.action == DecisionAction.EXIT:
            if current_position is None:
                return None

            if not isinstance(decision.metadata, dict):
                decision.metadata = {}

            position_volume_before = float(current_position.volume)
            requested_exit_volume = self._finite_float(decision.volume)
            is_partial_exit = False
            exit_position = current_position

            if requested_exit_volume is not None and requested_exit_volume > 0:
                partial_volume = self._resolve_partial_exit_volume(
                    symbol=symbol,
                    position_volume=position_volume_before,
                    requested_volume=requested_exit_volume,
                )
                if partial_volume is not None:
                    is_partial_exit = True
                    decision.metadata["is_partial"] = True
                    decision.metadata["position_volume_before"] = position_volume_before
                    decision.metadata["partial_volume_requested"] = partial_volume
                    exit_position = Position(
                        ticket=int(current_position.ticket),
                        symbol=str(current_position.symbol),
                        side=current_position.side,
                        volume=float(partial_volume),
                        price_open=float(current_position.price_open),
                        sl=current_position.sl,
                        tp=current_position.tp,
                        comment=str(current_position.comment),
                        magic=current_position.magic,
                        time_open_utc=current_position.time_open_utc,
                        metadata=dict(current_position.metadata) if isinstance(current_position.metadata, dict) else {},
                    )
            if self.dry_run:
                result = self._dry_result("DRY_EXIT", decision)
            else:
                result = self.broker.close_position(exit_position, reason=decision.reason)

            result.pnl = self._estimate_close_pnl(exit_position, result)
            self._emit_broker_request_events(symbol=symbol, strategy=strategy, phase="exit", result=result)

            try:
                exit_attempt_no = max(1, int(decision.metadata.get("exit_attempt_no", 1)))
            except Exception:
                exit_attempt_no = 1

            if is_partial_exit:
                self.store.append_event(
                    {
                        "event": "position_partial_exit",
                        "symbol": symbol,
                        "strategy": strategy,
                        "reason": decision.reason,
                        "position_volume_before": position_volume_before,
                        "closed_volume": float(exit_position.volume),
                        "remaining_volume_estimate": max(0.0, position_volume_before - float(exit_position.volume)),
                        "result": result.__dict__,
                        "exit_attempt_no": exit_attempt_no,
                    }
                )
                if self._is_success_trade_result(result):
                    self._emit_trade_event(
                        event_type="EXIT",
                        symbol=symbol,
                        side=current_position.side.value,
                        volume=float(exit_position.volume),
                        price=result.filled_price,
                        sl=float(exit_position.sl) if exit_position.sl is not None else None,
                        tp=float(exit_position.tp) if exit_position.tp is not None else None,
                        strategy_name=strategy,
                        expected_pnl_usd=None,
                        ticket=exit_position.ticket,
                    )
                if result.ok:
                    self._notify_trade(
                        f"{symbol} EXIT_PARTIAL {strategy} vol={exit_position.volume} | "
                        f"{decision.reason} | {result.status} pnl={result.pnl}"
                    )
                else:
                    self._notify_error(
                        f"{symbol} EXIT_PARTIAL FAILED {strategy} | {result.status} | {result.message}"
                    )
                return result

            self.risk_engine.on_trade_close(result.pnl, symbol=symbol)
            if result.ok and callable(self.on_position_closed):
                hold_seconds = None
                if current_position.time_open_utc is not None:
                    hold_seconds = (
                        datetime.now(timezone.utc) - current_position.time_open_utc.astimezone(timezone.utc)
                    ).total_seconds()
                self.on_position_closed(
                    symbol=symbol,
                    position=current_position,
                    result=result,
                    reason=decision.reason,
                    hold_seconds=hold_seconds,
                )

            pnl_status = "known" if result.pnl is not None else "unknown"
            trade_ledger = {
                "event": "trade_ledger",
                "ticket": int(current_position.ticket),
                "symbol": symbol,
                "strategy": strategy,
                "side": current_position.side.value,
                "entry_price": current_position.price_open,
                "exit_price": result.filled_price,
                "volume": current_position.volume,
                "realized_pnl": result.pnl,
                "pnl_status": pnl_status,
                "reason": decision.reason,
                "exit_attempt_no": exit_attempt_no,
                "exit_ok": bool(result.ok),
                "retcode": result.retcode,
                "exit_fill_status": "FILLED" if result.filled_price is not None else "UNFILLED",
                "exit_fail_reason": None if bool(result.ok) else str(result.message or ""),
            }

            self.store.append_event(
                {
                    "event": "position_exit",
                    "symbol": symbol,
                    "strategy": strategy,
                    "reason": decision.reason,
                    "result": result.__dict__,
                    "exit_attempt_no": exit_attempt_no,
                }
            )
            self.store.append_event(trade_ledger)
            normalized_payload = dict(trade_ledger)
            normalized_payload["event"] = "trade_ledger_normalized"
            self.store.append_event(normalized_payload)

            if self._is_success_trade_result(result):
                self._emit_trade_event(
                    event_type="EXIT",
                    symbol=symbol,
                    side=current_position.side.value,
                    volume=float(exit_position.volume),
                    price=result.filled_price,
                    sl=float(exit_position.sl) if exit_position.sl is not None else None,
                    tp=float(exit_position.tp) if exit_position.tp is not None else None,
                    strategy_name=strategy,
                    expected_pnl_usd=None,
                    ticket=exit_position.ticket,
                )
                self._notify_trade(f"{symbol} EXIT {strategy} | {decision.reason} | {result.status} pnl={result.pnl}")
            else:
                self._notify_error(f"{symbol} EXIT FAILED {strategy} | {result.status} | {result.message}")
            return result

        if decision.action not in {DecisionAction.BUY, DecisionAction.SELL}:
            return None

        side = Side.BUY if decision.action == DecisionAction.BUY else Side.SELL
        setup_key = None
        if isinstance(decision.metadata, dict):
            setup_key = decision.metadata.get("setup_key") or decision.metadata.get("setup") or decision.reason
        block_reason = self._guard_open_permission(
            symbol=symbol,
            direction=side.value,
            setup_key=str(setup_key or "") or None,
        )
        if block_reason is not None:
            self.store.append_event(
                {
                    "event": "order_skip",
                    "symbol": symbol,
                    "strategy": strategy,
                    "reason": f"RISK_GUARD_BLOCKED:{block_reason}",
                }
            )
            return None

        current_positions = self.broker.get_positions(symbol=symbol)
        if len(current_positions) >= self.max_positions_per_symbol:
            self.store.append_event(
                {
                    "event": "order_skip",
                    "symbol": symbol,
                    "strategy": strategy,
                    "reason": "MAX_POSITIONS_REACHED",
                }
            )
            return None

        if current_position is not None:
            if self._same_side(current_position, decision):
                return None
            if not self.allow_opposite_position:
                if self.dry_run:
                    close_result = self._dry_result("DRY_AUTO_CLOSE_OPPOSITE", decision)
                else:
                    close_result = self.broker.close_position(current_position, reason="flip_before_new_entry")
                close_result.pnl = self._estimate_close_pnl(current_position, close_result)
                self.risk_engine.on_trade_close(close_result.pnl, symbol=symbol)
                self._emit_broker_request_events(
                    symbol=symbol,
                    strategy=strategy,
                    phase="flip_close",
                    result=close_result,
                )
                if self._is_success_trade_result(close_result):
                    self._emit_trade_event(
                        event_type="EXIT",
                        symbol=symbol,
                        side=current_position.side.value,
                        volume=float(current_position.volume),
                        price=close_result.filled_price,
                        sl=float(current_position.sl) if current_position.sl is not None else None,
                        tp=float(current_position.tp) if current_position.tp is not None else None,
                        strategy_name=strategy,
                        expected_pnl_usd=None,
                        ticket=current_position.ticket,
                    )
                if close_result.ok and callable(self.on_position_closed):
                    hold_seconds = None
                    if current_position.time_open_utc is not None:
                        hold_seconds = (
                            datetime.now(timezone.utc) - current_position.time_open_utc.astimezone(timezone.utc)
                        ).total_seconds()
                    self.on_position_closed(
                        symbol=symbol,
                        position=current_position,
                        result=close_result,
                        reason="flip_before_new_entry",
                        hold_seconds=hold_seconds,
                    )
                trade_ledger = {
                    "event": "trade_ledger",
                    "ticket": int(current_position.ticket),
                    "symbol": symbol,
                    "strategy": strategy,
                    "side": current_position.side.value,
                    "entry_price": current_position.price_open,
                    "exit_price": close_result.filled_price,
                    "volume": current_position.volume,
                    "realized_pnl": close_result.pnl,
                    "pnl_status": "known" if close_result.pnl is not None else "unknown",
                    "reason": "flip_before_new_entry",
                    "exit_attempt_no": 1,
                    "exit_ok": bool(close_result.ok),
                    "retcode": close_result.retcode,
                    "exit_fill_status": "FILLED" if close_result.filled_price is not None else "UNFILLED",
                    "exit_fail_reason": None if bool(close_result.ok) else str(close_result.message or ""),
                }
                self.store.append_event(trade_ledger)
                normalized_payload = dict(trade_ledger)
                normalized_payload["event"] = "trade_ledger_normalized"
                self.store.append_event(normalized_payload)
                if not close_result.ok:
                    self._notify_error(
                        f"{symbol} flip close failed | {close_result.status} | {close_result.message}"
                    )
                    return close_result

        if decision.sl is None:
            self.store.append_event(
                {
                    "event": "order_skip",
                    "symbol": symbol,
                    "strategy": strategy,
                    "reason": "MISSING_SL",
                }
            )
            return None

        constraints = self.broker.get_symbol_constraints(symbol)
        if constraints is None:
            constraints = SymbolConstraints()

        account = self.broker.account_info()
        equity = account.get("equity") if isinstance(account, dict) else None
        account_currency = self._safe_upper_text(account.get("currency")) if isinstance(account, dict) else ""
        profit_currency = self._profit_currency_for_symbol(symbol=symbol, constraints=constraints)
        quote_to_account_rate = None
        require_fx_rate = False
        if account_currency and profit_currency and account_currency != profit_currency:
            require_fx_rate = True
            quote_to_account_rate = self._resolve_fx_rate(from_currency=profit_currency, to_currency=account_currency)

        signal_price = decision.metadata.get("signal_close")
        if signal_price is None:
            signal_price = decision.metadata.get("entry_price")
        if signal_price is None and decision.sl is not None and decision.tp is not None:
            signal_price = (float(decision.sl) + float(decision.tp)) / 2.0
        if signal_price is None and decision.sl is not None:
            signal_price = float(decision.sl)

        requested_volume = self._requested_volume(decision, instrument, constraints)
        volume_scale = float(decision.metadata.get("volume_scale", 1.0) or 1.0)
        win_probability = self._finite_float(decision.metadata.get("win_probability"))
        if win_probability is None:
            win_probability = self._finite_float(decision.metadata.get("edge_win_rate"))
        payoff_ratio = self._finite_float(decision.metadata.get("payoff_ratio"))
        if payoff_ratio is None:
            payoff_ratio = self._finite_float(decision.metadata.get("expected_rr"))
        fee_aware_cfg = self._fee_aware_fixed_risk_config()
        if fee_aware_cfg is not None:
            volume, volume_err, plan_meta, planned_sl, planned_tp = self._fee_aware_size_and_plan_entry(
                symbol=symbol,
                side=side,
                decision=decision,
                constraints=constraints,
                entry_price=signal_price,
                cfg=fee_aware_cfg,
            )
            if planned_sl is not None:
                decision.sl = planned_sl
            if planned_tp is not None:
                decision.tp = planned_tp
        else:
            volume, volume_err, plan_meta = self.risk_engine.plan_entry_volume(
                constraints=constraints,
                equity=equity,
                entry_price=signal_price,
                sl_price=decision.sl,
                requested_volume=requested_volume,
                side=side.value,
                volume_scale=volume_scale,
                quote_to_account_rate=quote_to_account_rate,
                require_fx_rate=require_fx_rate,
                win_probability=win_probability,
                payoff_ratio=payoff_ratio,
                symbol=symbol,
            )
        if fee_aware_cfg is None and volume is None and str(volume_err or "") == "MIN_VOLUME_EXCEEDS_RISK_LIMIT":
            can_retry_with_auto_volume = False
            if isinstance(plan_meta, dict):
                risk_amount = self._finite_float(plan_meta.get("risk_amount"))
                min_required_risk_amount = self._finite_float(plan_meta.get("min_required_risk_amount"))
                can_retry_with_auto_volume = (
                    risk_amount is not None
                    and min_required_risk_amount is not None
                    and risk_amount >= min_required_risk_amount
                )
            else:
                can_retry_with_auto_volume = True

            if can_retry_with_auto_volume:
                retry_volume, retry_volume_err, retry_plan_meta = self.risk_engine.plan_entry_volume(
                    constraints=constraints,
                    equity=equity,
                    entry_price=signal_price,
                    sl_price=decision.sl,
                    requested_volume=None,
                    side=side.value,
                    volume_scale=volume_scale,
                    quote_to_account_rate=quote_to_account_rate,
                    require_fx_rate=require_fx_rate,
                    win_probability=win_probability,
                    payoff_ratio=payoff_ratio,
                    symbol=symbol,
                )
                if retry_volume is not None:
                    volume = retry_volume
                    volume_err = retry_volume_err
                    plan_meta = dict(retry_plan_meta or {})
                    plan_meta["risk_plan_retried"] = True
                else:
                    plan_meta = dict(plan_meta or {})
                    plan_meta["risk_plan_retried"] = True
                    if isinstance(retry_plan_meta, dict) and retry_plan_meta.get("risk_amount") is not None:
                        plan_meta["risk_plan_retry_reason"] = retry_volume_err
                    volume = retry_volume
        if volume is None:
            self.store.append_event(
                {
                    "event": "order_skip",
                    "symbol": symbol,
                    "strategy": strategy,
                    "reason": f"RISK_PLAN_FAILED:{volume_err}",
                    "details": plan_meta,
                }
            )
            self._notify_error(
                f"{symbol} {decision.action.value} RISK_PLAN_FAILED {strategy} | {volume_err}"
            )
            return OrderResult(
                ok=False,
                status="RISK_PLAN_FAILED",
                message=str(volume_err or "RISK_PLAN_FAILED"),
                raw={"plan_meta": dict(plan_meta or {})},
            )

        expected_pnl_usd = None
        if isinstance(plan_meta, dict):
            try:
                expected_pnl_usd = float(plan_meta.get("expected_pnl_usd"))
            except (TypeError, ValueError):
                expected_pnl_usd = None

        max_loss_map = self.execution_cfg.get("max_expected_loss_usd_by_symbol", {}) or {}
        max_expected_loss_usd = None
        if isinstance(max_loss_map, dict):
            try:
                raw_cap = max_loss_map.get(symbol)
                if raw_cap is None:
                    raw_cap = max_loss_map.get(str(symbol).upper())
                if raw_cap is not None:
                    cap_val = float(raw_cap)
                    if math.isfinite(cap_val) and cap_val > 0:
                        max_expected_loss_usd = cap_val
            except (TypeError, ValueError):
                max_expected_loss_usd = None

        if expected_pnl_usd is not None and max_expected_loss_usd is not None:
            expected_loss_abs = abs(float(expected_pnl_usd))
            if expected_loss_abs > float(max_expected_loss_usd):
                self.store.append_event(
                    {
                        "event": "order_skip",
                        "symbol": symbol,
                        "strategy": strategy,
                        "reason": "EXPECTED_LOSS_CAP",
                        "details": {
                            "expected_loss_usd": float(expected_loss_abs),
                            "max_expected_loss_usd": float(max_expected_loss_usd),
                        },
                    }
                )
                return OrderResult(
                    ok=False,
                    status="EXPECTED_LOSS_CAP",
                    message=f"expected loss {expected_loss_abs:.2f} exceeds cap {max_expected_loss_usd:.2f}",
                    raw={"expected_loss_usd": float(expected_loss_abs), "max_expected_loss_usd": float(max_expected_loss_usd)},
                )

        magic = int(self.execution_cfg.get("magic", 0) or 0)
        sl_val = float(decision.sl) if decision.sl is not None else None
        tp_val = float(decision.tp) if decision.tp is not None else None
        point = float(getattr(constraints, "point", 0.0) or 0.0)
        if point > 0:
            if sl_val is not None:
                sl_val = self._quantize_by_point(sl_val, point)
            if tp_val is not None:
                tp_val = self._quantize_by_point(tp_val, point)

        intent = OrderIntent(
            symbol=symbol,
            side=side,
            volume=volume,
            reason=decision.reason,
            strategy=strategy,
            comment=f"{self.comment_prefix}:{strategy}"[:31],
            magic=magic,
            sl=sl_val,
            tp=tp_val,
            external_signal_id=decision.metadata.get("external_signal_id"),
            metadata=self._entry_metadata_with_spread_snapshot(
                symbol=symbol,
                decision=decision,
                plan_meta=plan_meta,
            ),
        )

        if self.dry_run:
            result = self._dry_result("DRY_ENTRY", decision)
        else:
            precheck = self.broker.precheck_order(intent)
            precheck_retry_meta: Optional[Dict[str, Any]] = None
            if not precheck.ok and int(precheck.retcode or 0) == 10014:
                intent.volume = self.risk_engine.repair_volume_for_10014(constraints)
                precheck = self.broker.precheck_order(intent)

            if not precheck.ok and int(precheck.retcode or 0) == 10016:
                precheck_retry_meta = self._adjust_stops_for_constraints(
                    intent=intent,
                    constraints=constraints,
                    symbol=symbol,
                )
                if precheck_retry_meta is not None:
                    cap_block = self._block_stop_adjustment_if_over_cap(
                        intent=intent,
                        constraints=constraints,
                        symbol=symbol,
                        strategy=strategy,
                        phase="precheck",
                        adjustment=precheck_retry_meta,
                    )
                    if cap_block is not None:
                        return cap_block
                    precheck = self.broker.precheck_order(intent)
                    self.store.append_event(
                        {
                            "event": "order_stops_auto_adjusted",
                            "symbol": symbol,
                            "strategy": strategy,
                            "phase": "precheck",
                            "adjustment": precheck_retry_meta,
                            "result": precheck.__dict__,
                            "recheck": True,
                        }
                    )
                    if not precheck.ok and int(precheck.retcode or 0) == 10016:
                        second_floor = float(precheck_retry_meta.get("min_distance_points", 0.0) or 0.0) * 2.0
                        precheck_retry_meta_2 = self._adjust_stops_for_constraints(
                            intent=intent,
                            constraints=constraints,
                            symbol=symbol,
                            min_distance_points_override=second_floor,
                        )
                        if precheck_retry_meta_2 is not None:
                            cap_block = self._block_stop_adjustment_if_over_cap(
                                intent=intent,
                                constraints=constraints,
                                symbol=symbol,
                                strategy=strategy,
                                phase="precheck_retry2",
                                adjustment=precheck_retry_meta_2,
                            )
                            if cap_block is not None:
                                return cap_block
                            precheck = self.broker.precheck_order(intent)
                            self.store.append_event(
                                {
                                    "event": "order_stops_auto_adjusted",
                                    "symbol": symbol,
                                    "strategy": strategy,
                                    "phase": "precheck_retry2",
                                    "adjustment": precheck_retry_meta_2,
                                    "result": precheck.__dict__,
                                    "recheck": True,
                                }
                            )

            if not precheck.ok:
                self.risk_engine.on_order_result(precheck)
                self.store.append_event(
                    {
                        "event": "order_submit",
                        "symbol": symbol,
                        "strategy": strategy,
                        "action": decision.action.value,
                        "reason": decision.reason,
                        "intent": intent.__dict__,
                        "result": precheck.__dict__,
                        "phase": "precheck",
                    }
                )
                if precheck_retry_meta is not None:
                    self.store.append_event(
                        {
                            "event": "order_stops_auto_adjusted",
                            "symbol": symbol,
                            "strategy": strategy,
                            "phase": "precheck_failed",
                            "adjustment": precheck_retry_meta,
                            "result": precheck.__dict__,
                        }
                    )
                self._emit_broker_request_events(symbol=symbol, strategy=strategy, phase="precheck", result=precheck)
                self._notify_error(
                    f"{symbol} {decision.action.value} PRECHECK FAILED {strategy} | {precheck.status} | {precheck.message}"
                )
                return precheck

            result = self.broker.send_order(intent)
            if not result.ok and int(result.retcode or 0) == 10014:
                intent.volume = self.risk_engine.repair_volume_for_10014(constraints)
                precheck_retry = self.broker.precheck_order(intent)
                if precheck_retry.ok:
                    result = self.broker.send_order(intent)
            elif not result.ok and int(result.retcode or 0) == 10016:
                send_retry_meta = self._adjust_stops_for_constraints(
                    intent=intent,
                    constraints=constraints,
                    symbol=symbol,
                )
                if send_retry_meta is not None:
                    cap_block = self._block_stop_adjustment_if_over_cap(
                        intent=intent,
                        constraints=constraints,
                        symbol=symbol,
                        strategy=strategy,
                        phase="send",
                        adjustment=send_retry_meta,
                    )
                    if cap_block is not None:
                        return cap_block
                    self.store.append_event(
                        {
                            "event": "order_stops_auto_adjusted",
                            "symbol": symbol,
                            "strategy": strategy,
                            "phase": "send",
                            "adjustment": send_retry_meta,
                            "result": result.__dict__,
                        }
                    )
                    result = self.broker.send_order(intent)

        self.risk_engine.on_order_result(result)

        self.store.append_event(
            {
                "event": "order_submit",
                "symbol": symbol,
                "strategy": strategy,
                "action": decision.action.value,
                "reason": decision.reason,
                "intent": intent.__dict__,
                "result": result.__dict__,
            }
        )
        self._emit_broker_request_events(symbol=symbol, strategy=strategy, phase="entry", result=result)

        if result.ok and result.ticket:
            # Ghost Fill Detection: Verify position existence immediately to prevent state desync.
            verified = False
            for _ in range(3):
                try:
                    current_positions = self.broker.get_positions(symbol=symbol)
                    if any(int(p.ticket) == int(result.ticket) for p in current_positions):
                        verified = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)

            if not verified:
                self.store.append_event(
                    {
                        "event": "ghost_fill_detected",
                        "symbol": symbol,
                        "ticket": result.ticket,
                        "strategy": strategy,
                        "reason": "broker_confirmed_but_position_missing",
                    }
                )
                self._notify_error(
                    f"{symbol} GHOST FILL DETECTED {strategy} | Ticket {result.ticket} missing from positions"
                )
                result.ok = False
                result.status = "GHOST_FILL"
                result.message = "Broker confirmed order but position missing"

        if self._is_success_trade_result(result):
            self._emit_trade_event(
                event_type="ENTRY",
                symbol=symbol,
                side=side.value,
                volume=float(intent.volume),
                price=result.filled_price if result.filled_price is not None else self._finite_float(signal_price),
                sl=float(intent.sl),
                tp=float(intent.tp) if intent.tp is not None else None,
                strategy_name=strategy,
                expected_pnl_usd=expected_pnl_usd,
                ticket=result.ticket,
            )
            self._notify_trade(
                f"{symbol} {decision.action.value} {strategy} vol={intent.volume} | {decision.reason} | {result.status}"
            )
        elif result.ok:
            self._notify_trade(
                f"{symbol} {decision.action.value} {strategy} vol={intent.volume} | {decision.reason} | {result.status}"
            )
        else:
            self._notify_error(
                f"{symbol} {decision.action.value} FAILED {strategy} | {result.status} | {result.message}"
            )
        return result
