import unittest

from execution.execution_churn_guard import ExecutionChurnGuard


class ChurnGuardGlobalDailyTests(unittest.TestCase):
    def test_global_daily_limit_blocks_after_cap(self) -> None:
        guard = ExecutionChurnGuard(
            {
                "enabled": True,
                "reentry_cooldown_seconds": 0,
                "max_entries_per_symbol_per_hour": 99,
                "max_entries_per_symbol_per_day": 99,
                "max_entries_global_per_day": 2,
                "daily_reset_timezone": "Asia/Seoul",
            }
        )
        guard.record_entry("BTCUSD", 1700000000.0)
        guard.record_entry("ETHUSD", 1700000100.0)
        self.assertEqual(guard.should_block_entry("GOLD", 1700000200.0), "CHURN_DAILY_LIMIT")

    def test_symbol_specific_min_hold_floor(self) -> None:
        guard = ExecutionChurnGuard(
            {
                "min_hold_bars_floor": 2,
                "min_hold_bars_floor_by_symbol": {"BTCUSD": 3},
            }
        )
        self.assertEqual(guard.enforce_min_hold(1, symbol="BTCUSD"), 3)
        self.assertEqual(guard.enforce_min_hold(1, symbol="GOLD"), 2)


if __name__ == "__main__":
    unittest.main()
