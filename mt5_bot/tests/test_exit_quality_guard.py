import unittest
from datetime import datetime, timedelta, timezone

from core.models import Position, Side
from execution.exit_quality_guard import ExitQualityGuard


class ExitQualityGuardTests(unittest.TestCase):
    def test_blocks_soft_exit_for_tiny_quick_pnl(self) -> None:
        guard = ExitQualityGuard(
            {
                "enabled": True,
                "tiny_profit_block_usd": 2.0,
                "min_hold_seconds_for_soft_exit": 300,
            }
        )
        position = Position(
            ticket=1,
            symbol="BTCUSD",
            side=Side.BUY,
            volume=0.01,
            price_open=100.0,
            time_open_utc=datetime.now(timezone.utc) - timedelta(seconds=120),
            metadata={"floating_pnl": 1.0, "swap": 0.0, "commission": 0.0},
        )
        out = guard.should_block_exit(position=position, reason="TREND_REGIME_EXIT:BUY_REGIME_FLIP_EXIT")
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "SOFT_EXIT_BLOCKED")

    def test_allows_non_soft_exit(self) -> None:
        guard = ExitQualityGuard({"enabled": True})
        position = Position(ticket=1, symbol="BTCUSD", side=Side.BUY, volume=0.01, price_open=100.0)
        out = guard.should_block_exit(position=position, reason="HARD_STOP_LOSS")
        self.assertTrue(out["allow"])

    def test_allows_soft_exit_when_m5_reverse_confirmed(self) -> None:
        guard = ExitQualityGuard({"enabled": True})
        position = Position(
            ticket=1,
            symbol="BTCUSD",
            side=Side.BUY,
            volume=0.01,
            price_open=100.0,
            time_open_utc=datetime.now(timezone.utc),
            metadata={"floating_pnl": 0.1, "swap": 0.0, "commission": 0.0},
        )
        out = guard.should_block_exit(
            position=position,
            reason="TREND_REGIME_EXIT:BUY_TRAIL_BREACH",
            m5_reverse_confirmed=True,
        )
        self.assertTrue(out["allow"])


if __name__ == "__main__":
    unittest.main()
