import unittest
from pathlib import Path

from core.config import load_config
from execution.execution_churn_guard import ExecutionChurnGuard


class ConservativeProfileDefaultsTests(unittest.TestCase):
    def test_checked_in_config_defaults_to_quality_first_guards(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")

        self.assertTrue(cfg["general"]["dry_run"])
        self.assertFalse(cfg["execution"]["live_trading_enabled"])
        self.assertTrue(cfg["entry_quality_guard"]["enabled"])
        self.assertTrue(cfg["cost_edge_guard"]["enabled"])
        self.assertTrue(cfg["mtf_confirm"]["enabled"])
        self.assertLessEqual(float(cfg["risk_guard"]["risk_per_trade_pct"]), 0.003)
        self.assertLessEqual(float(cfg["risk_guard"]["max_risk_per_trade_pct"]), 0.005)
        self.assertLessEqual(int(cfg["execution_churn_guard"]["max_entries_global_per_day"]), 3)
        self.assertLessEqual(int(cfg["execution_churn_guard"]["per_symbol_daily_limits"]["BTCUSD"]), 1)

    def test_zero_symbol_daily_limit_blocks_entries(self) -> None:
        guard = ExecutionChurnGuard(
            {
                "enabled": True,
                "reentry_cooldown_seconds": 0,
                "max_entries_per_symbol_per_hour": 99,
                "max_entries_per_symbol_per_day": 99,
                "per_symbol_daily_limits": {"ETHUSD": 0},
                "daily_reset_timezone": "Asia/Seoul",
            }
        )

        self.assertEqual(guard.should_block_entry("ETHUSD", 1700000000.0), "CHURN_DAILY_LIMIT")


if __name__ == "__main__":
    unittest.main()
