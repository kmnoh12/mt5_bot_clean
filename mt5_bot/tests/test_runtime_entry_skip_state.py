import unittest
import time
import pandas as pd
from types import SimpleNamespace

from core.models import BotMode, DecisionAction, StrategyDecision, StrategyState, StrategySymbolState
from core.runtime import TradingRuntime


class _FakeStore:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, payload):
        self.events.append(dict(payload))


class _FakeManualPositionGuard:
    enabled = False
    block_strategy_for_protected_symbols = False

    @staticmethod
    def run_cycle(positions, broker, now_ts):
        return {"protected_symbols": set(), "events": []}


class _FakeControl:
    @staticmethod
    def clear_manual_entry():
        return None

    @staticmethod
    def clear_flatten():
        return None


class _FakeBroker:
    def __init__(self) -> None:
        self.connected = True
        self._intraday_bars = pd.DataFrame(
            {
                "time": pd.date_range(start="2026-01-01T00:00:00Z", periods=8, freq="min", tz="UTC"),
                "open": [100, 101, 102, 103, 104, 105, 106, 107],
                "high": [101, 102, 103, 104, 105, 106, 107, 108],
                "low": [99, 100, 101, 102, 103, 104, 105, 106],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
            }
        )

    def fetch_bars(self, symbol: str, timeframe: str, bars: int):  # noqa: ARG002
        return self._intraday_bars.copy()

    def get_positions(self, symbol=None):  # noqa: ARG002
        return []

    def account_info(self):
        return {"equity": 10000.0, "balance": 10000.0}


class _StatefulPendingStrategy:
    enabled = True
    min_cooldown_bars = 2

    def __init__(self) -> None:
        self._states = {}

    def get_symbol_state(self, symbol: str) -> StrategySymbolState:
        key = str(symbol).upper()
        state = self._states.get(key)
        if state is None:
            state = StrategySymbolState()
            self._states[key] = state
        return state

    def evaluate(self, symbol, bars, position, external_signal=None, context=None):  # noqa: ARG002
        del bars, position, external_signal, context
        return StrategyDecision(
            action=DecisionAction.BUY,
            strategy="entry_skip",
            reason="UNIT_TEST_SKIP_CASE",
            confidence=1.0,
            metadata={},
        )


class _MtfConfirm:
    def __init__(self) -> None:
        self.enabled = True
        self.symbols = {"BTCUSD"}
        self.confirm_timeframe = "TIMEFRAME_M5"

    def is_symbol_enabled(self, symbol: str) -> bool:
        return symbol in self.symbols

    def allow_entry(self, symbol: str, action, bars):  # noqa: ARG002
        return False


class _NoopExitEngine:
    @staticmethod
    def choose(position, strategy_decision, trailing_signal):  # noqa: ARG002
        return strategy_decision


class _NoopChurnGuard:
    def __init__(self) -> None:
        pass

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
    def __call__(self, *args, **kwargs):
        return {}

    @staticmethod
    def evaluate_entry(*args, **kwargs):  # noqa: ARG002
        return {"allow": True}

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

    @staticmethod
    def snapshot():
        return {}


class _NoopExitRetryGuard:
    def __init__(self) -> None:
        pass

    @staticmethod
    def should_allow(ticket, reason, now_ts):  # noqa: ARG002
        return True, 0.0, 0

    @staticmethod
    def on_attempt(ticket, reason, now_ts, success):  # noqa: ARG002
        return {"event": "position_exit_retry_backoff", "ticket": ticket}

    @staticmethod
    def snapshot():
        return {}


class _NoopTrailingGuard:
    @staticmethod
    def snapshot():
        return {}

    @staticmethod
    def drop_closed_positions(positions):  # noqa: ARG002
        return None


class _FakeNotifier:
    @staticmethod
    def send_error(message: str) -> None:
        pass

    @staticmethod
    def send_trade(message: str) -> None:
        pass


class _FakeRiskEngine:
    @staticmethod
    def evaluate_limits(account):  # noqa: ARG002
        return None

    @staticmethod
    def resume():
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


class _FakeOrderManager:
    def __init__(self):
        self.default_volume = 0.01

    def process_decision(self, *args, **kwargs):  # noqa: ARG002
        raise AssertionError("order_manager should not be called after M5_CONFIRM_BLOCK")


class _FakeLifecycle:
    def __init__(self) -> None:
        self.stop_requested = False
        self.stop_reason = None

    def request_stop(self, reason):
        self.stop_reason = reason
        self.stop_requested = True

    def execute_shutdown(self, reason):
        return None


class RuntimeEntrySkipStateTests(unittest.TestCase):
    def test_buy_skip_by_m5_confirm_forces_strategy_cooldown(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        strategy = _StatefulStrategy = _StatefulPendingStrategy()
        runtime.broker = _FakeBroker()
        runtime.config = {"universe": [{"symbol": "BTCUSD", "strategy": "entry_skip", "timeframe": "TIMEFRAME_M1"}]}
        runtime.strategies = {"entry_skip": strategy}
        runtime.store = _FakeStore()
        runtime.state = {}
        runtime.mode = BotMode.LIVE
        runtime.dashboard_enabled = False
        runtime.control_channel = None
        runtime.bar_gate = SimpleNamespace(should_evaluate=lambda symbol, bars: (True, None))
        runtime.lifecycle = _FakeLifecycle()
        runtime.notifier = _FakeNotifier()
        runtime.manual_position_guard = _FakeManualPositionGuard()
        runtime.risk_engine = _FakeRiskEngine()
        runtime.exit_engine = _NoopExitEngine()
        runtime.execution_churn_guard = _NoopChurnGuard()
        runtime.entry_quality_guard = _NoopEntryQualityGuard()
        runtime.cost_edge_guard = _NoopCostEdgeGuard()
        runtime.exit_quality_guard = _NoopExitQualityGuard()
        runtime.exit_retry_guard = _NoopExitRetryGuard()
        runtime.mtf_confirm = _MtfConfirm()
        runtime.order_manager = _FakeOrderManager()
        runtime.trailing_guard = _NoopTrailingGuard()
        runtime._halt_for_broker_fatal = lambda: False
        runtime._reload_config_if_changed = lambda: None
        runtime._run_heartbeat = lambda: None
        runtime._reconcile_strategy_states = lambda broker_positions: None
        runtime._reset_strategy_state_for_protected_symbol = lambda symbol: None
        runtime._run_pending_protection_cycle = lambda broker_positions, excluded_symbols: set()
        runtime._consume_manual_entry_request = lambda ctrl: False
        runtime._run_auto_tuning_cycle = lambda bars_by_symbol: None
        runtime._save_runtime_state = lambda: None
        runtime._refresh_daily_reference_levels = lambda: None
        runtime._write_watchdog_heartbeat = lambda: None
        runtime._protection_blocked_symbols = set()
        runtime._pending_protection_updates = {}
        runtime._daily_reference_cache = {}
        runtime._daily_reference_refresh_interval_seconds = 3600.0
        runtime._last_daily_reference_refresh_ts = time.time()
        runtime._active_symbols = ["BTCUSD"]
        runtime._last_risk_guard_reason = ""
        runtime.bars_per_request = 20
        runtime._active_symbols = ["BTCUSD"]
        runtime._last_heartbeat_ts = 0.0
        runtime._watchdog_heartbeat_path = None
        runtime._last_risk_guard_reason = ""

        state = strategy.get_symbol_state("BTCUSD")
        state.state = StrategyState.ENTRY_PENDING
        state.pending_order = True
        state.cooldown_bars_remaining = 0

        runtime.run_cycle()

        updated = strategy.get_symbol_state("BTCUSD")
        self.assertEqual(updated.state, StrategyState.COOLDOWN)
        self.assertFalse(updated.pending_order)
        self.assertEqual(updated.last_reason, "ENTRY_BLOCKED:M5_CONFIRM_BLOCK")
        self.assertGreaterEqual(updated.cooldown_bars_remaining, strategy.min_cooldown_bars)
        self.assertTrue(any(
            event.get("event") == "strategy_state_forced_cooldown"
            and event.get("symbol") == "BTCUSD"
            and event.get("reason") == "ENTRY_BLOCKED:M5_CONFIRM_BLOCK"
            for event in runtime.store.events
        ))


if __name__ == "__main__":
    unittest.main()
