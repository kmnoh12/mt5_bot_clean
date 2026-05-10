import unittest

from execution.execution_churn_guard import ExecutionChurnGuard


class ExecutionChurnGuardTests(unittest.TestCase):
    def test_reentry_cooldown_blocks_fast_reentry(self) -> None:
        guard = ExecutionChurnGuard(
            config={
                "enabled": True,
                "reentry_cooldown_seconds": 180,
                "max_entries_per_symbol_per_hour": 4,
                "min_hold_bars_floor": 2,
            }
        )
        guard.record_entry("BTCUSD", 1000.0)
        self.assertEqual(guard.should_block_entry("BTCUSD", 1050.0), "CHURN_COOLDOWN")
        self.assertIsNone(guard.should_block_entry("BTCUSD", 1181.0))

    def test_hourly_limit_blocks_overtrading(self) -> None:
        guard = ExecutionChurnGuard(
            config={
                "enabled": True,
                "reentry_cooldown_seconds": 0,
                "max_entries_per_symbol_per_hour": 4,
                "min_hold_bars_floor": 2,
            }
        )
        for ts in (1000.0, 1200.0, 1400.0, 1600.0):
            guard.record_entry("ETHUSD", ts)
        self.assertEqual(guard.should_block_entry("ETHUSD", 1700.0), "CHURN_HOURLY_LIMIT")

    def test_enforce_min_hold_floor(self) -> None:
        guard = ExecutionChurnGuard(config={"min_hold_bars_floor": 2})
        self.assertEqual(guard.enforce_min_hold(1), 2)
        self.assertEqual(guard.enforce_min_hold(4), 4)

    def test_daily_limit_blocks_after_cap(self) -> None:
        guard = ExecutionChurnGuard(
            config={
                "enabled": True,
                "reentry_cooldown_seconds": 0,
                "max_entries_per_symbol_per_hour": 99,
                "max_entries_per_symbol_per_day": 2,
                "daily_reset_timezone": "Asia/Seoul",
                "min_hold_bars_floor": 2,
            }
        )
        guard.record_entry("BTCUSD", 1700000000.0)
        guard.record_entry("BTCUSD", 1700000600.0)
        self.assertEqual(guard.should_block_entry("BTCUSD", 1700001200.0), "CHURN_DAILY_LIMIT")

    def test_eth_specific_daily_limit(self) -> None:
        guard = ExecutionChurnGuard(
            config={
                "enabled": True,
                "reentry_cooldown_seconds": 0,
                "max_entries_per_symbol_per_hour": 99,
                "max_entries_per_symbol_per_day": 12,
                "max_entries_per_symbol_per_day_eth": 1,
                "daily_reset_timezone": "Asia/Seoul",
                "min_hold_bars_floor": 2,
            }
        )
        guard.record_entry("ETHUSD", 1700000000.0)
        self.assertEqual(guard.should_block_entry("ETHUSD", 1700000600.0), "CHURN_DAILY_LIMIT")

    def test_tiny_pnl_quick_exit_triggers_cooldown(self) -> None:
        guard = ExecutionChurnGuard(
            config={
                "enabled": True,
                "reentry_cooldown_seconds": 0,
                "max_entries_per_symbol_per_hour": 99,
                "max_entries_per_symbol_per_day": 99,
                "daily_reset_timezone": "Asia/Seoul",
                "tiny_pnl_threshold_usd": 2.0,
                "quick_exit_window_seconds": 300,
                "tiny_pnl_max_count_per_hour": 2,
                "tiny_pnl_cooldown_seconds": 3600,
                "min_hold_bars_floor": 2,
            }
        )
        self.assertIsNone(guard.record_close("BTCUSD", 1700000000.0, realized_pnl=1.0, hold_seconds=120))
        report = guard.record_close("BTCUSD", 1700000100.0, realized_pnl=-0.5, hold_seconds=240)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["reason"], "CHURN_TINY_PNL_COOLDOWN")
        self.assertEqual(guard.should_block_entry("BTCUSD", 1700000200.0), "CHURN_TINY_PNL_COOLDOWN")


if __name__ == "__main__":
    unittest.main()
