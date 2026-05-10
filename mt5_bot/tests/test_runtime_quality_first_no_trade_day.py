import unittest

from core.models import DecisionAction
from execution.entry_quality_guard import EntryQualityGuard


class RuntimeQualityFirstNoTradeDayTests(unittest.TestCase):
    def test_quality_first_can_block_all_entries(self) -> None:
        guard = EntryQualityGuard(
            {
                "enabled": True,
                "min_score": 0.95,
                "min_score_risk_off": 0.95,
                "min_score_risk_on": 0.95,
            }
        )
        out = guard.evaluate_entry(
            symbol="BTCUSD",
            decision_action=DecisionAction.BUY,
            decision_metadata={
                "entry_style": "trend_follow",
                "indicator_snapshot": {
                    "trend_strength": 0.3,
                    "adx_norm": 0.3,
                    "ema_gap_atr": 0.2,
                },
            },
            m5_aligned=False,
        )
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "ENTRY_QUALITY_BLOCK")


if __name__ == "__main__":
    unittest.main()
