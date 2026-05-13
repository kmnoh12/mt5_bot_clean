from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.models import BotMode, DecisionAction, OrderResult, StrategyDecision, StrategySymbolState, SymbolConstraints
from core.runtime import TradingRuntime


class _FakeStore:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, payload):
        self.events.append(dict(payload))

    def save_state(self, state):
        self.saved_state = dict(state)


class _FakeBroker:
    def __init__(self, bars) -> None:
        self._bars = bars

    def fetch_bars(self, symbol: str, timeframe: str, bars: int):  # noqa: ARG002
        return self._bars.copy()

    def get_positions(self, symbol=None):  # noqa: ARG002
        return []

    def account_info(self):
        return {"equity": 10000.0, "balance": 10000.0}

    def get_live_spread(self, symbol: str):  # noqa: ARG002
        return 0.0

    def get_symbol_constraints(self, symbol: str):  # noqa: ARG002
        return SymbolConstraints(min_volume=0.01, max_volume=1.0, volume_step=0.01, point=0.01, contract_size=1.0)


class _HoldStrategy:
    enabled = True
    min_cooldown_bars = 1

    def __init__(self) -> None:
        self._states = {}
        self.results = []

    def get_symbol_state(self, symbol: str) -> StrategySymbolState:
        key = str(symbol).upper()
        self._states.setdefault(key, StrategySymbolState())
        return self._states[key]

    def evaluate(self, symbol, bars, position, context=None):  # noqa: ARG002
        return StrategyDecision(
            action=DecisionAction.HOLD,
            strategy="hold_strategy",
            reason="IDLE",
            metadata={},
        )

    def apply_order_result(self, symbol, decision, result):  # noqa: ARG002
        self.results.append((decision, result))


class _FakeOrderManager:
    default_volume = 0.01

    def __init__(self) -> None:
        self.calls = []

    def process_decision(self, instrument, decision, current_position):  # noqa: ARG002
        self.calls.append(decision)
        return OrderResult(ok=True, status="DRY_RUN", message="ok", ticket=123)


class _FakeRiskEngine:
    @staticmethod
    def evaluate_limits(account):  # noqa: ARG002
        return None

    @staticmethod
    def status():
        return SimpleNamespace(
            halted=False,
            reason="",
            session_start_equity=10000.0,
            daily_start_equity=10000.0,
            daily_date_utc="2026-01-01",
            consecutive_losses=0,
            equity_peak=10000.0,
        )

    @staticmethod
    def snapshot():
        return {}


class _NoopManualGuard:
    enabled = False
    block_strategy_for_protected_symbols = False

    @staticmethod
    def run_cycle(positions, broker, now_ts):  # noqa: ARG002
        return {"protected_symbols": set(), "events": []}

    @staticmethod
    def snapshot():
        return {}


class _NoopMtfConfirm:
    confirm_timeframe = "TIMEFRAME_M5"

    @staticmethod
    def is_symbol_enabled(symbol):  # noqa: ARG002
        return False


class _NoopExitEngine:
    @staticmethod
    def choose(position, strategy_decision, trailing_signal):  # noqa: ARG002
        return strategy_decision


class _NoopChurnGuard:
    @staticmethod
    def enforce_min_hold(value, symbol=None):  # noqa: ARG002
        return value

    @staticmethod
    def should_block_entry(symbol, now_ts, is_flip):  # noqa: ARG002
        return None

    @staticmethod
    def record_entry(symbol, now_ts):  # noqa: ARG002
        return None

    @staticmethod
    def snapshot():
        return {}


class _NoopEntryQualityGuard:
    @staticmethod
    def evaluate_entry(*args, **kwargs):  # noqa: ARG002
        return {"allow": True, "score": 1.0, "threshold": 0.0}

    @staticmethod
    def record_entry_context(*args, **kwargs):  # noqa: ARG002
        return None

    @staticmethod
    def snapshot():
        return {}


class _NoopCostEdgeGuard:
    @staticmethod
    def evaluate_entry(*args, **kwargs):  # noqa: ARG002
        return {"allow": True}

    @staticmethod
    def snapshot():
        return {}


class _NoopExitQualityGuard:
    @staticmethod
    def should_block_exit(position, reason, m5_reverse_confirmed):  # noqa: ARG002
        return {"allow": True}


class _NoopExitRetryGuard:
    @staticmethod
    def should_allow(ticket, reason, now_ts):  # noqa: ARG002
        return True, 0.0, 0

    @staticmethod
    def on_attempt(ticket, reason, now_ts, success):  # noqa: ARG002
        return {}

    @staticmethod
    def snapshot():
        return {}


class _NoopTrailingGuard:
    @staticmethod
    def snapshot():
        return {}


class _FakeLifecycle:
    stop_requested = False
    stop_reason = None

    def request_stop(self, reason):
        self.stop_reason = reason
        self.stop_requested = True


def _long_frame() -> pd.DataFrame:
    rows = []
    for idx, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="min")):
        close = 100.0 + ((idx % 5) * 0.05)
        rows.append({"time": timestamp, "open": close - 0.02, "high": 101.0, "low": 99.0, "close": close})
    rows.append({"time": pd.Timestamp("2026-01-01T00:20:00Z"), "open": 98.9, "high": 101.2, "low": 98.55, "close": 99.35})
    return pd.DataFrame(rows)


def _runtime(report_dir: Path) -> TradingRuntime:
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.config = {
        "general": {"dry_run": True},
        "execution": {"live_trading_enabled": False, "max_spread": 1200},
        "execution_style": {"name": "fee_aware_fixed_risk_profit_lock"},
        "entry_quality": {"min_signal_score": 70, "min_reward_to_net_risk_ratio": 1.5},
        "risk_per_trade": {"hard_max_net_loss_usd": 1.25, "spread_points": 0.0},
        "position_limits": {"max_open_positions_total": 1},
        "opportunity_scanner": {"enabled": True, "drive_entries": True, "lookback_bars": 20, "atr_period": 5},
        "fee_aware_entry_filter": {"enabled": True},
        "no_trade_bias_guard": {"warning_no_trade_hours": 24.0, "failure_no_trade_hours": 48.0},
        "reports": {"enabled": True, "output_dir": str(report_dir)},
        "universe": [{"symbol": "XAUUSD", "strategy": "hold_strategy", "timeframe": "M5", "volume": 0.01}],
    }
    runtime.mode = BotMode.LIVE
    runtime.dry_run = True
    runtime.broker = _FakeBroker(_long_frame())
    runtime.store = _FakeStore()
    runtime.state = {}
    runtime.strategies = {"hold_strategy": _HoldStrategy()}
    runtime.order_manager = _FakeOrderManager()
    runtime.risk_engine = _FakeRiskEngine()
    runtime.manual_position_guard = _NoopManualGuard()
    runtime.exit_engine = _NoopExitEngine()
    runtime.execution_churn_guard = _NoopChurnGuard()
    runtime.entry_quality_guard = _NoopEntryQualityGuard()
    runtime.cost_edge_guard = _NoopCostEdgeGuard()
    runtime.exit_quality_guard = _NoopExitQualityGuard()
    runtime.exit_retry_guard = _NoopExitRetryGuard()
    runtime.trailing_guard = _NoopTrailingGuard()
    runtime.mtf_confirm = _NoopMtfConfirm()
    runtime.dashboard_enabled = False
    runtime.control_channel = None
    runtime.lifecycle = _FakeLifecycle()
    runtime.bar_gate = SimpleNamespace(should_evaluate=lambda symbol, bars: (True, None), snapshot=lambda: {})
    runtime.bars_per_request = 50
    runtime._last_risk_guard_reason = ""
    runtime._last_heartbeat_ts = 0.0
    runtime._daily_reference_cache = {}
    runtime._daily_reference_refresh_interval_seconds = 3600.0
    runtime._last_daily_reference_refresh_ts = time.time()
    runtime._next_market_data_missing_log_ts_by_symbol = {}
    runtime._protection_blocked_symbols = set()
    runtime._pending_protection_updates = {}
    runtime._active_opportunity_candidates = []
    runtime._halt_for_broker_fatal = lambda: False
    runtime._reload_config_if_changed = lambda: None
    runtime._run_heartbeat = lambda: None
    runtime._write_watchdog_heartbeat = lambda: None
    runtime._refresh_daily_reference_levels = lambda: None
    runtime._detect_and_record_broker_closed_positions = lambda broker_positions: None
    runtime._reconcile_strategy_states = lambda broker_positions: None
    runtime._run_pending_protection_cycle = lambda broker_positions, excluded_symbols: set()
    runtime._consume_manual_entry_request = lambda ctrl: False
    runtime._evaluate_profit_lock_for_position = lambda position: None
    runtime._m5_reverse_confirmed_for_exit = lambda symbol, position: False
    runtime._run_auto_tuning_cycle = lambda bars_by_symbol: None
    from core.no_trade_guard import NoTradeBiasGuard
    from execution.daily_bleed_guard import DailyBleedGuard
    from strategies.entry_filter import FeeAwareEntryFilter
    from strategies.opportunity_scanner import TradeOpportunityScanner

    runtime.daily_bleed_guard = DailyBleedGuard(config={"enabled": True})
    runtime.trade_opportunity_scanner = TradeOpportunityScanner(**runtime._trade_opportunity_scanner_config())
    runtime.fee_aware_entry_filter = FeeAwareEntryFilter(runtime._fee_aware_entry_filter_config())
    runtime.no_trade_guard = NoTradeBiasGuard(config=runtime.config.get("no_trade_bias_guard", {}))
    return runtime


def test_runtime_scanner_can_drive_entry_and_write_reports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime = _runtime(Path(tmp))

        runtime.run_cycle()

        buy_decisions = [item for item in runtime.order_manager.calls if item.action == DecisionAction.BUY]
        assert buy_decisions
        decision = buy_decisions[0]
        assert decision.strategy == "v4_opportunity_scanner"
        assert decision.metadata["v4_opportunity_scanner"] is True
        assert any(event.get("event") == "v4_opportunity_scanner_entry_selected" for event in runtime.store.events)

        active_report = json.loads((Path(tmp) / "active_opportunities.json").read_text(encoding="utf-8"))
        assert active_report["eligible_count"] >= 1
        assert active_report["best_eligible_candidate"]["symbol"] == "XAUUSD"

        no_trade_report = json.loads((Path(tmp) / "no_trade_diagnostics.json").read_text(encoding="utf-8"))
        assert no_trade_report["raw_signal_count"] >= 1
        assert no_trade_report["executed_trade_count"] == 1
