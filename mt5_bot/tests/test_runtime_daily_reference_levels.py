import unittest
from types import SimpleNamespace
import time

import pandas as pd

from core.models import BotMode, DecisionAction, StrategyDecision, StrategyState, StrategySymbolState
from core.runtime import TradingRuntime


class _FakeStore:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, payload):
        self.events.append(dict(payload))


class _FakeBroker:
    def __init__(self, bars: pd.DataFrame, intraday_bars: pd.DataFrame | None = None) -> None:
        self.connected = True
        self._bars = bars
        self._intraday_bars = intraday_bars if intraday_bars is not None else bars
        self.calls = []

    def fetch_bars(self, symbol: str, timeframe: str, bars: int):  # noqa: A003
        self.calls.append((symbol, timeframe, bars))
        if timeframe == "TIMEFRAME_D1":
            return self._bars.copy()
        return self._intraday_bars.copy()

    def get_positions(self, symbol=None):
        return []

    def account_info(self):
        return {"balance": 1000.0, "equity": 1000.0}


class _FakeControlChannel:
    @staticmethod
    def load():
        return {
            "manual_halt": False,
            "resume_requested": False,
            "paused": False,
            "flatten_requested": False,
        }

    @staticmethod
    def save(_ctrl):
        return None

    @staticmethod
    def clear_flatten():
        return None


class _FailOnCallControlChannel:
    def __init__(self) -> None:
        self.load_calls = 0
        self.save_calls = 0
        self.clear_flatten_calls = 0
        self.clear_manual_entry_calls = 0

    def load(self):
        self.load_calls += 1
        raise AssertionError("control_channel.load() must not be called when dashboard is disabled")

    def save(self, _ctrl):
        self.save_calls += 1
        raise AssertionError("control_channel.save() must not be called when dashboard is disabled")

    def clear_flatten(self):
        self.clear_flatten_calls += 1
        raise AssertionError("control_channel.clear_flatten() must not be called when dashboard is disabled")

    def clear_manual_entry(self):
        self.clear_manual_entry_calls += 1
        raise AssertionError("control_channel.clear_manual_entry() must not be called when dashboard is disabled")


class _FakeManualPositionGuard:
    enabled = False
    block_strategy_for_protected_symbols = False

    @staticmethod
    def run_cycle(positions, broker, now_ts):
        return {"protected_symbols": set(), "events": []}


class _CaptureStrategy:
    enabled = True

    def __init__(self) -> None:
        self.last_context = None

    def evaluate(self, symbol, bars, position, external_signal=None, context=None):
        self.last_context = context
        return StrategyDecision(action=DecisionAction.HOLD, reason="NOOP", strategy="capture", metadata={})


class _StatefulHoldStrategy:
    enabled = True
    min_cooldown_bars = 3

    def __init__(self) -> None:
        self._states = {}

    def get_symbol_state(self, symbol: str):
        key = str(symbol).upper()
        if key not in self._states:
            self._states[key] = StrategySymbolState()
        return self._states[key]

    def evaluate(self, symbol, bars, position, external_signal=None, context=None):
        return StrategyDecision(action=DecisionAction.HOLD, reason="NOOP", strategy="stateful_hold", metadata={})


class RuntimeDailyReferenceTests(unittest.TestCase):
    def test_run_cycle_risk_guard_halt_forces_cooldown_without_stop_request(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        strategy = _StatefulHoldStrategy()
        runtime.broker = _FakeBroker(
            bars=pd.DataFrame({"high": [10.0, 11.0], "low": [7.0, 8.0]}),
            intraday_bars=pd.DataFrame(
                {
                    "time": pd.date_range(start="2026-01-01T00:00:00Z", periods=3, freq="min", tz="UTC"),
                    "open": [1, 2, 3],
                    "high": [2, 3, 4],
                    "low": [0, 1, 2],
                    "close": [1.5, 2.5, 3.5],
                }
            ),
        )
        runtime.config = {"dashboard": {"enabled": False}, "universe": [{"symbol": "BTCUSD", "strategy": "stateful_hold"}]}
        runtime.strategies = {"stateful_hold": strategy}
        runtime.store = _FakeStore()
        runtime.state = {}
        runtime.mode = BotMode.LIVE
        runtime.dashboard_enabled = False
        runtime.control_channel = None
        runtime.bar_gate = SimpleNamespace(
            should_evaluate=lambda symbol, bars: (True, None),
            snapshot=lambda: {},
        )
        stop_reasons = []
        runtime.lifecycle = SimpleNamespace(request_stop=lambda reason: stop_reasons.append(reason))
        runtime.notifier = SimpleNamespace(send_error=lambda message: None, send_trade=lambda message: None)
        runtime.manual_position_guard = _FakeManualPositionGuard()
        runtime.risk_engine = SimpleNamespace(
            evaluate_limits=lambda account: "MAX_CONSECUTIVE_LOSSES_5",
            resume=lambda: None,
            status=lambda: SimpleNamespace(
                halted=True,
                reason="MAX_CONSECUTIVE_LOSSES_5",
                session_start_equity=1000.0,
                daily_start_equity=1000.0,
                daily_date_utc="2026-01-01",
                consecutive_losses=5,
                equity_peak=1000.0,
            ),
            snapshot=lambda: {},
        )
        runtime.exit_engine = SimpleNamespace(choose=lambda position, strategy_decision, trailing_signal: strategy_decision)
        runtime.execution_churn_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.entry_quality_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.cost_edge_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.exit_quality_guard = SimpleNamespace(should_block_exit=lambda **kwargs: {"allow": True})
        runtime.exit_retry_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.mtf_confirm = SimpleNamespace(
            is_symbol_enabled=lambda symbol: False,
            confirm_timeframe="TIMEFRAME_M5",
            allow_entry=lambda **kwargs: True,
        )
        runtime.order_manager = SimpleNamespace(default_volume=0.01, process_decision=lambda **kwargs: None)
        runtime.trailing_guard = SimpleNamespace(snapshot=lambda: {}, drop_closed_positions=lambda positions: None)
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
        runtime._daily_reference_cache = {}
        runtime._daily_reference_refresh_interval_seconds = 3600.0
        runtime._last_daily_reference_refresh_ts = time.time()
        runtime._protection_blocked_symbols = set()
        runtime._last_risk_guard_reason = ""
        runtime.bars_per_request = 50

        runtime.run_cycle()
        runtime.run_cycle()

        self.assertEqual(stop_reasons, [])
        state = strategy.get_symbol_state("BTCUSD")
        self.assertEqual(state.state, StrategyState.COOLDOWN)
        self.assertGreaterEqual(int(state.cooldown_bars_remaining), 3)
        self.assertEqual(sum(1 for e in runtime.store.events if e.get("event") == "risk_guard_halt"), 1)
        self.assertGreaterEqual(
            sum(1 for e in runtime.store.events if e.get("event") == "strategy_state_forced_cooldown"),
            1,
        )

    def test_run_cycle_skips_dashboard_control_channel_when_disabled(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        runtime.broker = _FakeBroker(
            bars=pd.DataFrame({"high": [10.0, 11.0], "low": [7.0, 8.0]}),
            intraday_bars=pd.DataFrame(
                {
                    "time": pd.date_range(start="2026-01-01T00:00:00Z", periods=3, freq="min", tz="UTC"),
                    "open": [1, 2, 3],
                    "high": [2, 3, 4],
                    "low": [0, 1, 2],
                    "close": [1.5, 2.5, 3.5],
                }
            ),
        )
        runtime.config = {"dashboard": {"enabled": False}, "universe": []}
        runtime.strategies = {}
        runtime.store = _FakeStore()
        runtime.state = {}
        runtime.mode = BotMode.LIVE
        runtime.dashboard_enabled = False
        strict_control = _FailOnCallControlChannel()
        runtime.control_channel = strict_control
        runtime.bar_gate = SimpleNamespace(should_evaluate=lambda symbol, bars: (True, None))
        runtime.lifecycle = SimpleNamespace(request_stop=lambda reason: None)
        runtime.notifier = SimpleNamespace(send_error=lambda message: None, send_trade=lambda message: None)
        runtime.manual_position_guard = _FakeManualPositionGuard()
        runtime.risk_engine = SimpleNamespace(
            evaluate_limits=lambda account: None,
            resume=lambda: None,
            status=lambda: SimpleNamespace(
                halted=False,
                reason="",
                session_start_equity=None,
                daily_start_equity=None,
                daily_date_utc="",
                consecutive_losses=0,
                equity_peak=None,
            ),
        )
        runtime.exit_engine = SimpleNamespace(choose=lambda position, strategy_decision, trailing_signal: strategy_decision)
        runtime.execution_churn_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.entry_quality_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.cost_edge_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.exit_quality_guard = SimpleNamespace(should_block_exit=lambda **kwargs: {"allow": True})
        runtime.exit_retry_guard = SimpleNamespace(snapshot=lambda: {})
        runtime.mtf_confirm = SimpleNamespace(
            is_symbol_enabled=lambda symbol: False,
            confirm_timeframe="TIMEFRAME_M5",
            allow_entry=lambda **kwargs: True,
        )
        runtime.order_manager = SimpleNamespace(default_volume=0.01, process_decision=lambda **kwargs: None)
        runtime.trailing_guard = SimpleNamespace(snapshot=lambda: {}, drop_closed_positions=lambda positions: None)
        runtime._halt_for_broker_fatal = lambda: False
        runtime._reload_config_if_changed = lambda: None
        runtime._run_heartbeat = lambda: None
        runtime._reconcile_strategy_states = lambda broker_positions: None
        runtime._run_pending_protection_cycle = lambda broker_positions, excluded_symbols: set()
        runtime._consume_manual_entry_request = lambda ctrl: False
        runtime._run_auto_tuning_cycle = lambda bars_by_symbol: None
        runtime._save_runtime_state = lambda: None
        runtime._refresh_daily_reference_levels = lambda: None
        runtime._daily_reference_cache = {}
        runtime._daily_reference_refresh_interval_seconds = 3600.0
        runtime._last_daily_reference_refresh_ts = time.time()
        runtime._protection_blocked_symbols = set()
        runtime.bars_per_request = 50

        runtime.run_cycle()

        self.assertEqual(strict_control.load_calls, 0)
        self.assertEqual(strict_control.save_calls, 0)
        self.assertEqual(strict_control.clear_flatten_calls, 0)
        self.assertEqual(strict_control.clear_manual_entry_calls, 0)

    def test_refresh_daily_reference_levels_updates_cache(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        runtime.config = {
            "universe": [
                {"symbol": "BTCUSD", "strategy": "trend_regime_sm"},
                {"symbol": "ETHUSD", "strategy": "trend_regime_sm"},
            ]
        }
        runtime.strategies = {"trend_regime_sm": SimpleNamespace(enabled=True)}
        runtime._daily_reference_cache = {}
        runtime.store = _FakeStore()
        runtime._halt_for_broker_fatal = lambda: False
        runtime.broker = _FakeBroker(
            pd.DataFrame(
                {
                    "high": [100.0, 101.0, 102.0, 103.0, 104.0],
                    "low": [90.0, 91.0, 92.0, 93.0, 94.0],
                }
            )
        )

        runtime._refresh_daily_reference_levels()

        btc = runtime._daily_reference_cache["BTCUSD"]
        eth = runtime._daily_reference_cache["ETHUSD"]
        self.assertEqual(float(btc["pdh"]), 103.0)
        self.assertEqual(float(btc["pdl"]), 93.0)
        self.assertEqual(float(eth["pdh"]), 103.0)
        self.assertEqual(float(eth["pdl"]), 93.0)
        self.assertGreater(float(runtime._last_daily_reference_refresh_ts), 0.0)
        self.assertTrue(any(e.get("event") == "daily_reference_levels_refreshed" for e in runtime.store.events))
        self.assertTrue(all(call[1] == "TIMEFRAME_D1" for call in runtime.broker.calls))

    def test_run_cycle_passes_daily_levels_via_context(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        strategy = _CaptureStrategy()
        intraday = pd.DataFrame(
            {
                "time": pd.date_range(start="2026-01-01T00:00:00Z", periods=6, freq="min", tz="UTC"),
                "open": [1, 2, 3, 4, 5, 6],
                "high": [2, 3, 4, 5, 6, 7],
                "low": [0, 1, 2, 3, 4, 5],
                "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            }
        )
        runtime.broker = _FakeBroker(
            bars=pd.DataFrame({"high": [10.0, 11.0], "low": [7.0, 8.0]}),
            intraday_bars=intraday,
        )
        runtime.config = {"universe": [{"symbol": "BTCUSD", "strategy": "capture", "timeframe": "TIMEFRAME_M1"}]}
        runtime.strategies = {"capture": strategy}
        runtime.store = _FakeStore()
        runtime.state = {}
        runtime.mode = BotMode.LIVE
        runtime.dashboard_enabled = True
        runtime.bar_gate = SimpleNamespace(should_evaluate=lambda symbol, bars: (True, None))
        runtime.control_channel = _FakeControlChannel()
        runtime.lifecycle = SimpleNamespace(request_stop=lambda reason: None)
        runtime.notifier = SimpleNamespace(send_error=lambda message: None, send_trade=lambda message: None)
        runtime.manual_position_guard = _FakeManualPositionGuard()
        runtime.risk_engine = SimpleNamespace(
            evaluate_limits=lambda account: None,
            resume=lambda: None,
            status=lambda: SimpleNamespace(
                halted=False,
                reason="",
                session_start_equity=None,
                daily_start_equity=None,
                daily_date_utc="",
                consecutive_losses=0,
                equity_peak=None,
            ),
        )
        runtime.exit_engine = SimpleNamespace(choose=lambda position, strategy_decision, trailing_signal: strategy_decision)
        runtime.execution_churn_guard = SimpleNamespace(
            enforce_min_hold=lambda min_hold, symbol: min_hold,
            should_block_entry=lambda symbol, now_ts, is_flip: None,
            record_entry=lambda symbol, now_ts: None,
            snapshot=lambda: {},
        )
        runtime.entry_quality_guard = SimpleNamespace(
            snapshot=lambda: {},
            evaluate_entry=lambda **kwargs: {"allow": True, "score": 1.0, "threshold": 0.0, "risk_mode": "normal", "features": {}},
            record_entry_context=lambda **kwargs: None,
        )
        runtime.cost_edge_guard = SimpleNamespace(
            snapshot=lambda: {},
            evaluate_entry=lambda **kwargs: {"allow": True},
        )
        runtime.exit_quality_guard = SimpleNamespace(should_block_exit=lambda **kwargs: {"allow": True})
        runtime.exit_retry_guard = SimpleNamespace(
            snapshot=lambda: {},
            should_allow=lambda **kwargs: (True, 0.0, 0),
            on_attempt=lambda **kwargs: {},
        )
        runtime.mtf_confirm = SimpleNamespace(
            is_symbol_enabled=lambda symbol: False,
            confirm_timeframe="TIMEFRAME_M5",
            allow_entry=lambda **kwargs: True,
        )
        runtime.order_manager = SimpleNamespace(default_volume=0.01, process_decision=lambda **kwargs: None)
        runtime.trailing_guard = SimpleNamespace(snapshot=lambda: {}, drop_closed_positions=lambda positions: None)
        runtime._halt_for_broker_fatal = lambda: False
        runtime._reload_config_if_changed = lambda: None
        runtime._run_heartbeat = lambda: None
        runtime._reconcile_strategy_states = lambda broker_positions: None
        runtime._reset_strategy_state_for_protected_symbol = lambda symbol: None
        runtime._run_pending_protection_cycle = lambda broker_positions, excluded_symbols: set()
        runtime._consume_manual_entry_request = lambda ctrl: False
        runtime._run_auto_tuning_cycle = lambda bars_by_symbol: None
        runtime._evaluate_profit_lock_for_position = lambda position: None
        runtime._m5_reverse_confirmed_for_exit = lambda symbol, position: False
        runtime._save_runtime_state = lambda: None
        runtime._daily_reference_cache = {
            "BTCUSD": {"pdh": 11.0, "pdl": 7.0, "updated_at_utc": "2026-01-01T00:00:00+00:00", "timeframe": "TIMEFRAME_D1"}
        }
        runtime._daily_reference_refresh_interval_seconds = 3600.0
        runtime._last_daily_reference_refresh_ts = time.time()
        runtime._protection_blocked_symbols = set()
        runtime.bars_per_request = 50

        runtime.run_cycle()

        self.assertIsNotNone(strategy.last_context)
        mtf_info = strategy.last_context.mtf_info
        self.assertEqual(float(mtf_info["daily_reference"]["pdh"]), 11.0)
        self.assertEqual(float(mtf_info["daily_reference"]["pdl"]), 7.0)


if __name__ == "__main__":
    unittest.main()
