import unittest

from execution.entry_quality_guard import EntryQualityGuard


class RuntimeWinnerProfileUpdateTests(unittest.TestCase):
    def test_winner_profile_updates_after_closed_winner(self) -> None:
        guard = EntryQualityGuard({"enabled": True, "min_winner_pnl_usd": 5.0})
        guard.record_entry_context(
            ticket=1001,
            symbol="ETHUSD",
            metadata={
                "entry_style": "trend_follow",
                "m5_align": True,
                "indicator_snapshot": {
                    "trend_strength": 0.92,
                    "adx_norm": 0.88,
                    "ema_gap_atr": 1.3,
                },
            },
        )
        report = guard.record_closed_trade(ticket=1001, symbol="ETHUSD", pnl=7.5, hold_seconds=800)
        self.assertTrue(report["updated"])
        self.assertIn("ETHUSD", report["winner_profile"])


if __name__ == "__main__":
    unittest.main()
