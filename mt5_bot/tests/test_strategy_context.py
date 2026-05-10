import unittest

import pandas as pd

from core.models import DecisionAction, StrategyDecision, StrategyEvaluationContext
from strategies.base import BaseStateMachineStrategy


class _ContextEchoStrategy(BaseStateMachineStrategy):
    def __init__(self) -> None:
        super().__init__(name="context_echo", config={})

    def _evaluate_impl(self, symbol, bars, position, st) -> StrategyDecision:  # type: ignore[override]
        return StrategyDecision(
            action=DecisionAction.HOLD,
            reason="OK",
            strategy=self.name,
            metadata={
                "seen_mtf_info": dict(bars.attrs.get("mtf_info", {})),
                "seen_risk_context": dict(bars.attrs.get("risk_context", {})),
            },
        )


class StrategyContextTests(unittest.TestCase):
    def test_evaluate_exposes_context_to_bars_and_decision_metadata(self) -> None:
        strategy = _ContextEchoStrategy()
        bars = pd.DataFrame(
            {
                "time": pd.date_range(start="2026-01-01T00:00:00Z", periods=3, freq="min", tz="UTC"),
                "open": [100.0, 101.0, 101.5],
                "high": [101.0, 102.0, 102.5],
                "low": [99.5, 100.5, 101.0],
                "close": [100.5, 101.5, 101.7],
            }
        )
        context = StrategyEvaluationContext(
            mtf_info={
                "daily_reference": {
                    "pdh": 2500.0,
                    "pdl": 2450.0,
                    "timeframe": "TIMEFRAME_D1",
                }
            },
            equity=1234.5,
            equity_peak=1300.0,
            loss_streak=2,
            daily_pnl=-12.25,
        )

        decision = strategy.evaluate(symbol="GOLD", bars=bars, position=None, context=context)

        self.assertEqual(bars.attrs["mtf_info"]["daily_reference"]["pdh"], 2500.0)
        self.assertEqual(bars.attrs["mtf_info"]["daily_reference"]["pdl"], 2450.0)
        self.assertEqual(float(bars.attrs["risk_context"]["equity"]), 1234.5)
        self.assertEqual(int(bars.attrs["risk_context"]["loss_streak"]), 2)
        self.assertEqual(decision.metadata["seen_mtf_info"]["daily_reference"]["pdh"], 2500.0)
        self.assertEqual(decision.metadata["mtf_info"]["daily_reference"]["pdl"], 2450.0)
        self.assertEqual(float(decision.metadata["seen_risk_context"]["equity_peak"]), 1300.0)
        self.assertEqual(float(decision.metadata["risk_context"]["daily_pnl"]), -12.25)
        self.assertEqual(int(decision.metadata["risk_context"]["loss_streak"]), 2)


if __name__ == "__main__":
    unittest.main()
