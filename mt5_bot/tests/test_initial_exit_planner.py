import unittest
from dataclasses import dataclass

from core.models import SymbolConstraints
from execution.exit_planner import InitialExitPlanner


@dataclass
class Opportunity:
    direction: str
    entry_price: float
    invalidation_price: float
    target_reference_price: float


class InitialExitPlannerTests(unittest.TestCase):
    def test_long_plan_passes_fee_aware_thresholds(self) -> None:
        plan = InitialExitPlanner().plan(
            opportunity={
                "direction": "LONG",
                "entry_price": 100.0,
                "invalidation_price": 99.0,
                "target_reference_price": 105.0,
            },
            position_size=1.0,
            estimated_round_trip_cost=0.5,
            hard_max_loss=2.0,
            min_tp_net_profit=3.0,
            min_reward_to_net_risk_ratio=3.0,
        )

        self.assertTrue(plan.passed)
        self.assertAlmostEqual(plan.sl_price or 0.0, 99.0)
        self.assertAlmostEqual(plan.tp_price or 0.0, 105.0)
        self.assertAlmostEqual(plan.expected_net_loss_at_sl, 1.5)
        self.assertAlmostEqual(plan.expected_net_profit_at_tp, 4.5)
        self.assertGreaterEqual(plan.fee_adjusted_rr, 3.0)

    def test_short_plan_passes_from_object(self) -> None:
        plan = InitialExitPlanner().plan(
            opportunity=Opportunity("SHORT", 100.0, 101.0, 96.0),
            lot=2.0,
            contract_size=1.0,
            estimated_entry_cost=0.25,
            estimated_exit_cost=0.25,
            hard_max_loss=3.0,
            min_tp_net_profit=6.0,
        )

        self.assertTrue(plan.passed)
        self.assertAlmostEqual(plan.expected_net_loss_at_sl, 2.5)
        self.assertAlmostEqual(plan.expected_net_profit_at_tp, 7.5)

    def test_hard_max_loss_rejects(self) -> None:
        plan = InitialExitPlanner().plan(
            direction="BUY",
            entry_price=100.0,
            invalidation_price=98.0,
            target_reference_price=107.0,
            position_size=1.0,
            estimated_round_trip_cost=0.25,
            hard_max_loss=2.0,
        )

        self.assertFalse(plan.passed)
        self.assertEqual(plan.reason, "HARD_MAX_LOSS_EXCEEDED")

    def test_min_tp_net_profit_rejects(self) -> None:
        plan = InitialExitPlanner().plan(
            direction="BUY",
            entry_price=100.0,
            invalidation_price=99.5,
            target_reference_price=101.0,
            position_size=1.0,
            estimated_round_trip_cost=0.25,
            min_tp_net_profit=1.0,
            min_reward_to_net_risk_ratio=1.0,
        )

        self.assertFalse(plan.passed)
        self.assertEqual(plan.reason, "TP_NET_PROFIT_TOO_LOW")

    def test_rr_below_three_rejects(self) -> None:
        plan = InitialExitPlanner().plan(
            direction="LONG",
            entry_price=100.0,
            invalidation_price=99.0,
            target_reference_price=102.9,
            position_size=1.0,
            estimated_round_trip_cost=0.0,
            min_reward_to_net_risk_ratio=3.0,
        )

        self.assertFalse(plan.passed)
        self.assertEqual(plan.reason, "RR_TOO_LOW")

    def test_broker_stop_level_rejects_sl_too_close(self) -> None:
        plan = InitialExitPlanner().plan(
            direction="LONG",
            entry_price=100.0,
            invalidation_price=99.8,
            target_reference_price=107.0,
            position_size=1.0,
            symbol_spec=SymbolConstraints(point=0.01, trade_stops_level=50.0),
        )

        self.assertFalse(plan.passed)
        self.assertEqual(plan.reason, "BROKER_STOP_LEVEL_SL")

    def test_broker_stop_level_rejects_tp_too_close(self) -> None:
        plan = InitialExitPlanner().plan(
            direction="SHORT",
            entry_price=100.0,
            invalidation_price=101.0,
            target_reference_price=99.8,
            position_size=1.0,
            symbol_spec={"point": 0.01, "trade_stops_level": 50.0},
        )

        self.assertFalse(plan.passed)
        self.assertEqual(plan.reason, "BROKER_STOP_LEVEL_TP")

    def test_risk_model_costs_are_used(self) -> None:
        def model(**_kwargs):
            return {"round_trip_cost_usd": 1.0}

        plan = InitialExitPlanner().plan(
            direction="LONG",
            entry_price=100.0,
            invalidation_price=99.0,
            target_reference_price=107.0,
            position_size=1.0,
            estimated_round_trip_cost=0.0,
            risk_model=model,
            hard_max_loss=2.0,
            min_tp_net_profit=4.0,
        )

        self.assertTrue(plan.passed)
        self.assertAlmostEqual(plan.expected_net_loss_at_sl, 2.0)
        self.assertAlmostEqual(plan.expected_net_profit_at_tp, 6.0)

    def test_derives_tp_when_reference_missing(self) -> None:
        plan = InitialExitPlanner().plan(
            direction="BUY",
            entry_price=100.0,
            invalidation_price=99.5,
            position_size=2.0,
            estimated_round_trip_cost=0.5,
            min_tp_net_profit=3.0,
            min_reward_to_net_risk_ratio=3.0,
        )

        self.assertTrue(plan.passed)
        self.assertAlmostEqual(plan.tp_price or 0.0, 102.5)
        self.assertAlmostEqual(plan.expected_net_profit_at_tp, 4.5)


if __name__ == "__main__":
    unittest.main()
