import unittest

from core.models import SymbolConstraints
from execution.cost_edge_guard import CostEdgeGuard


class CostEdgeGuardTests(unittest.TestCase):
    def test_blocks_low_edge_ratio(self) -> None:
        guard = CostEdgeGuard({"enabled": True, "min_edge_to_cost_ratio_default": 3.0})
        guard.record_cost("BTCUSD", 5.0)
        out = guard.evaluate_entry(
            symbol="BTCUSD",
            decision_metadata={"risk_per_unit": 0.1, "expected_rr": 1.5},
            requested_volume=0.01,
            constraints=SymbolConstraints(contract_size=1.0),
        )
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "EDGE_TOO_LOW")

    def test_allows_high_edge_ratio(self) -> None:
        guard = CostEdgeGuard({"enabled": True, "min_edge_to_cost_ratio_default": 3.0})
        out = guard.evaluate_entry(
            symbol="BTCUSD",
            decision_metadata={"risk_per_unit": 100.0, "expected_rr": 2.0},
            requested_volume=0.01,
            constraints=SymbolConstraints(contract_size=1.0),
        )
        self.assertTrue(out["allow"])


if __name__ == "__main__":
    unittest.main()
