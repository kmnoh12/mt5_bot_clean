import unittest

from execution.daily_bleed_guard import DailyBleedGuard, utc_ts


class DailyBleedGuardTests(unittest.TestCase):
    def test_default_config_values(self) -> None:
        guard = DailyBleedGuard()
        self.assertTrue(guard.enabled)
        self.assertEqual(guard.max_daily_net_loss_usd, 3.0)
        self.assertEqual(guard.stop_after_consecutive_losses, 3)
        self.assertEqual(guard.cooldown_after_loss_minutes, 30.0)
        self.assertEqual(guard.cooldown_after_same_setup_loss_minutes, 60.0)
        self.assertEqual(guard.same_direction_loss_limit_per_day, 2)
        self.assertEqual(guard.same_symbol_loss_limit_per_day, 3)

    def test_three_consecutive_losses_blocks_new_entries(self) -> None:
        guard = DailyBleedGuard({"cooldown_after_loss_minutes": 0, "max_daily_net_loss_usd": 99})
        ts = utc_ts(2026, 1, 1, 1)
        for idx in range(3):
            guard.record_trade_close("BTCUSD", -0.25, ts + idx, direction="BUY")
        self.assertEqual(guard.should_block_entry("ETHUSD", ts + 10), "DAILY_BLEED_CONSECUTIVE_LOSSES")

    def test_win_resets_consecutive_losses(self) -> None:
        guard = DailyBleedGuard({"cooldown_after_loss_minutes": 0, "max_daily_net_loss_usd": 99})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -0.25, ts)
        guard.record_trade_close("BTCUSD", -0.25, ts + 1)
        guard.record_trade_close("BTCUSD", 0.50, ts + 2)
        self.assertEqual(guard.consecutive_losses, 0)
        self.assertIsNone(guard.should_block_entry("ETHUSD", ts + 3))

    def test_daily_three_dollar_loss_blocks_new_entries(self) -> None:
        guard = DailyBleedGuard({"cooldown_after_loss_minutes": 0})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -1.25, ts, direction="BUY")
        guard.record_trade_close("ETHUSD", -1.75, ts + 1, direction="SELL")
        self.assertEqual(guard.should_block_entry("XAUUSD", ts + 2), "DAILY_BLEED_NET_LOSS_LIMIT")

    def test_daily_net_pnl_counts_wins_against_losses(self) -> None:
        guard = DailyBleedGuard({"cooldown_after_loss_minutes": 0})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -2.50, ts)
        guard.record_trade_close("ETHUSD", 1.00, ts + 1)
        self.assertEqual(guard.daily_net_pnl_usd, -1.50)
        self.assertIsNone(guard.should_block_entry("XAUUSD", ts + 2))

    def test_loss_cooldown_applies_to_same_symbol(self) -> None:
        guard = DailyBleedGuard({"max_daily_net_loss_usd": 99, "stop_after_consecutive_losses": 99})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -0.25, ts)
        self.assertEqual(guard.should_block_entry("BTCUSD", ts + 60), "DAILY_BLEED_LOSS_COOLDOWN")

    def test_loss_cooldown_expires(self) -> None:
        guard = DailyBleedGuard({"max_daily_net_loss_usd": 99, "stop_after_consecutive_losses": 99})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -0.25, ts)
        self.assertIsNone(guard.should_block_entry("BTCUSD", ts + 30 * 60 + 1))

    def test_same_setup_cooldown_applies(self) -> None:
        guard = DailyBleedGuard(
            {"cooldown_after_loss_minutes": 0, "max_daily_net_loss_usd": 99, "stop_after_consecutive_losses": 99}
        )
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -0.25, ts, setup_key="lsr_reclaim")
        self.assertEqual(
            guard.should_block_entry("ETHUSD", ts + 60, setup_key="lsr_reclaim"),
            "DAILY_BLEED_SAME_SETUP_COOLDOWN",
        )

    def test_same_setup_cooldown_is_setup_specific(self) -> None:
        guard = DailyBleedGuard(
            {"cooldown_after_loss_minutes": 0, "max_daily_net_loss_usd": 99, "stop_after_consecutive_losses": 99}
        )
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -0.25, ts, setup_key="lsr_reclaim")
        self.assertIsNone(guard.should_block_entry("ETHUSD", ts + 60, setup_key="breakout"))

    def test_same_direction_day_limit_applies(self) -> None:
        guard = DailyBleedGuard(
            {"cooldown_after_loss_minutes": 0, "max_daily_net_loss_usd": 99, "stop_after_consecutive_losses": 99}
        )
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -0.25, ts, direction="BUY")
        guard.record_trade_close("BTCUSD", -0.25, ts + 1, direction="BUY")
        self.assertEqual(
            guard.should_block_entry("BTCUSD", ts + 2, direction="BUY"),
            "DAILY_BLEED_SAME_DIRECTION_LIMIT",
        )

    def test_same_direction_day_limit_is_direction_specific(self) -> None:
        guard = DailyBleedGuard(
            {"cooldown_after_loss_minutes": 0, "max_daily_net_loss_usd": 99, "stop_after_consecutive_losses": 99}
        )
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -0.25, ts, direction="BUY")
        guard.record_trade_close("BTCUSD", -0.25, ts + 1, direction="BUY")
        self.assertIsNone(guard.should_block_entry("BTCUSD", ts + 2, direction="SELL"))

    def test_same_symbol_day_limit_applies(self) -> None:
        guard = DailyBleedGuard(
            {
                "cooldown_after_loss_minutes": 0,
                "max_daily_net_loss_usd": 99,
                "stop_after_consecutive_losses": 99,
                "same_direction_loss_limit_per_day": 99,
            }
        )
        ts = utc_ts(2026, 1, 1, 1)
        for idx, direction in enumerate(("BUY", "SELL", "BUY")):
            guard.record_trade_close("BTCUSD", -0.25, ts + idx, direction=direction)
        self.assertEqual(guard.should_block_entry("BTCUSD", ts + 4), "DAILY_BLEED_SAME_SYMBOL_LIMIT")

    def test_reset_on_date_change_clears_blocks(self) -> None:
        guard = DailyBleedGuard({"cooldown_after_loss_minutes": 0})
        ts = utc_ts(2026, 1, 1, 23, 59)
        guard.record_trade_close("BTCUSD", -3.0, ts)
        self.assertEqual(guard.should_block_entry("BTCUSD", ts + 1), "DAILY_BLEED_NET_LOSS_LIMIT")
        self.assertIsNone(guard.should_block_entry("BTCUSD", utc_ts(2026, 1, 2, 0, 0, 1)))

    def test_disabled_guard_never_blocks(self) -> None:
        guard = DailyBleedGuard({"enabled": False})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -99.0, ts)
        self.assertIsNone(guard.should_block_entry("BTCUSD", ts + 1))

    def test_protection_modify_allowed_when_new_entry_blocked(self) -> None:
        guard = DailyBleedGuard({"cooldown_after_loss_minutes": 0})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -3.0, ts)
        state = guard.permission_state("BTCUSD", ts + 1)
        self.assertFalse(state.allow_new_entries)
        self.assertTrue(state.allow_protection_modify)

    def test_snapshot_restores_blocking_state(self) -> None:
        guard = DailyBleedGuard({"cooldown_after_loss_minutes": 0})
        ts = utc_ts(2026, 1, 1, 1)
        guard.record_trade_close("BTCUSD", -3.0, ts)
        restored = DailyBleedGuard(snapshot=guard.snapshot())
        self.assertEqual(restored.should_block_entry("ETHUSD", ts + 1), "DAILY_BLEED_NET_LOSS_LIMIT")


if __name__ == "__main__":
    unittest.main()
