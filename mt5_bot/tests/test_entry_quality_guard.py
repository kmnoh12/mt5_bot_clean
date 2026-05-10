import unittest

from core.models import DecisionAction
from execution.entry_quality_guard import EntryQualityGuard


class EntryQualityGuardTests(unittest.TestCase):
    def test_blocks_mean_reversion_for_trend_only_symbols(self) -> None:
        guard = EntryQualityGuard(
            {
                "enabled": True,
                "trend_only_symbols": ["BTCUSD", "ETHUSD"],
            }
        )
        out = guard.evaluate_entry(
            symbol="BTCUSD",
            decision_action=DecisionAction.BUY,
            decision_metadata={"entry_style": "mean_reversion", "indicator_snapshot": {}},
            m5_aligned=True,
        )
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "ENTRY_QUALITY_BLOCK")

    def test_allows_high_quality_trend_entry(self) -> None:
        guard = EntryQualityGuard({"enabled": True})
        out = guard.evaluate_entry(
            symbol="BTCUSD",
            decision_action=DecisionAction.BUY,
            decision_metadata={
                "entry_style": "trend_follow",
                "indicator_snapshot": {
                    "trend_strength": 0.9,
                    "adx_norm": 0.9,
                    "ema_gap_atr": 1.5,
                },
            },
            m5_aligned=True,
        )
        self.assertTrue(out["allow"])
        self.assertGreaterEqual(out["score"], out["threshold"])

    def test_winner_profile_updates_from_closed_trades(self) -> None:
        guard = EntryQualityGuard({"enabled": True, "min_winner_pnl_usd": 5.0})
        guard.record_entry_context(
            ticket=1,
            symbol="BTCUSD",
            metadata={
                "entry_style": "trend_follow",
                "m5_align": True,
                "indicator_snapshot": {
                    "trend_strength": 0.8,
                    "adx_norm": 0.7,
                    "ema_gap_atr": 1.2,
                },
            },
        )
        out = guard.record_closed_trade(ticket=1, symbol="BTCUSD", pnl=10.0, hold_seconds=500)
        self.assertTrue(out["updated"])
        self.assertIn("BTCUSD", out["winner_profile"])


if __name__ == "__main__":
    unittest.main()
