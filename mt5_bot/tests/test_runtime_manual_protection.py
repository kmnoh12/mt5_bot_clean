import unittest
from types import SimpleNamespace

from core.models import StrategyState
from core.runtime import TradingRuntime


class _FakeStateStrategy:
    def __init__(self, state: StrategyState) -> None:
        self._state = SimpleNamespace(
            state=state,
            pending_order=True,
            cooldown_bars_remaining=3,
            last_reason="OLD",
            metadata={"bars_in_trade": 9, "last_manage_bar_time": "x"},
        )

    def get_symbol_state(self, _symbol: str):
        return self._state


class _FakeStore:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, payload):
        self.events.append(dict(payload))


class RuntimeManualProtectionTests(unittest.TestCase):
    def test_reset_strategy_state_for_protected_symbol(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        runtime.config = {
            "universe": [
                {"symbol": "GOLD", "strategy": "liquidity_sweep_reversal"},
            ]
        }
        runtime.strategies = {"liquidity_sweep_reversal": _FakeStateStrategy(StrategyState.EXIT_READY)}
        runtime._reset_strategy_state_for_protected_symbol("GOLD")
        st = runtime.strategies["liquidity_sweep_reversal"].get_symbol_state("GOLD")
        self.assertEqual(st.state, StrategyState.IDLE)
        self.assertFalse(st.pending_order)
        self.assertEqual(st.cooldown_bars_remaining, 0)
        self.assertEqual(st.last_reason, "MANUAL_POSITION_GUARD_PROTECTED")
        self.assertEqual(st.metadata["bars_in_trade"], 0)

    def test_reconcile_strategy_states_handles_lsr_ghost_positions(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        runtime.config = {
            "universe": [
                {"symbol": "GOLD", "strategy": "liquidity_sweep_reversal"},
            ]
        }
        runtime.strategies = {"liquidity_sweep_reversal": _FakeStateStrategy(StrategyState.IN_POSITION)}
        runtime.store = _FakeStore()

        runtime._reconcile_strategy_states([])

        st = runtime.strategies["liquidity_sweep_reversal"].get_symbol_state("GOLD")
        self.assertEqual(st.state, StrategyState.IDLE)
        self.assertFalse(st.pending_order)
        self.assertTrue(any(event.get("event") == "ghost_reconciliation" for event in runtime.store.events))


if __name__ == "__main__":
    unittest.main()
