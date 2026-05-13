import unittest
from dataclasses import dataclass

from core.models import Side, SymbolConstraints
from execution.profit_lock import ProfitLockTrailingManager


@dataclass
class PositionLike:
    ticket: int
    side: object
    volume: float
    price_open: float
    sl: float | None = None
    tp: float | None = None


class ProfitLockTrailingManagerTests(unittest.TestCase):
    def test_no_modify_below_first_threshold(self) -> None:
        decision = ProfitLockTrailingManager().evaluate(
            position=PositionLike(1, Side.BUY, 1.0, 100.0),
            current_price=101.99,
            now=0.0,
        )

        self.assertFalse(decision.should_modify)
        self.assertEqual(decision.reason, "NO_THRESHOLD")

    def test_two_dollars_locks_long_to_net_zero(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(1, "LONG", 1.0, 100.0),
            current_price=102.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 100.0)
        self.assertIsNone(decision.tp_price)
        self.assertEqual(decision.lock_net_profit, 0.0)

    def test_three_dollars_locks_long_to_one(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(1, "BUY", 1.0, 100.0),
            current_price=103.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 101.0)

    def test_five_dollars_locks_long_to_two(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(1, "BUY", 1.0, 100.0),
            current_price=105.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 102.0)
        self.assertIsNone(decision.tp_price)

    def test_ten_dollars_sets_sl_five_and_tp_twenty(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(1, "BUY", 1.0, 100.0),
            current_price=110.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 105.0)
        self.assertAlmostEqual(decision.tp_price or 0.0, 120.0)

    def test_twenty_dollars_sets_sl_ten_and_tp_thirty(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(1, "BUY", 1.0, 100.0),
            current_price=120.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 110.0)
        self.assertAlmostEqual(decision.tp_price or 0.0, 130.0)

    def test_thirty_dollars_sets_sl_fifteen_and_tp_forty_five(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(1, "BUY", 1.0, 100.0),
            current_price=130.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 115.0)
        self.assertAlmostEqual(decision.tp_price or 0.0, 145.0)

    def test_short_two_dollars_locks_to_net_zero(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(2, "SHORT", 1.0, 100.0),
            current_price=98.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 100.0)

    def test_short_ten_dollars_sets_lower_tp(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(2, "SELL", 1.0, 100.0),
            current_price=90.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 95.0)
        self.assertAlmostEqual(decision.tp_price or 0.0, 80.0)

    def test_long_sl_never_moves_backward(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(3, "BUY", 1.0, 100.0, sl=106.0),
            current_price=110.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 106.0)
        self.assertAlmostEqual(decision.tp_price or 0.0, 120.0)

    def test_short_sl_never_moves_backward(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(4, "SELL", 1.0, 100.0, sl=94.0),
            current_price=90.0,
            now=0.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.sl_price or 0.0, 94.0)
        self.assertAlmostEqual(decision.tp_price or 0.0, 80.0)

    def test_no_forward_progress_when_sl_and_tp_already_match(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(5, "BUY", 1.0, 100.0, sl=105.0, tp=120.0),
            current_price=110.0,
            now=0.0,
        )

        self.assertFalse(decision.should_modify)
        self.assertEqual(decision.reason, "NO_FORWARD_PROGRESS")

    def test_default_update_interval_blocks_second_modify(self) -> None:
        manager = ProfitLockTrailingManager()
        first = manager.evaluate(position=PositionLike(6, "BUY", 1.0, 100.0), current_price=110.0, now=100.0)
        second = manager.evaluate(position=PositionLike(6, "BUY", 1.0, 100.0), current_price=120.0, now=105.0)

        self.assertTrue(first.should_modify)
        self.assertFalse(second.should_modify)
        self.assertEqual(second.reason, "MIN_UPDATE_INTERVAL")

    def test_update_allowed_after_interval(self) -> None:
        manager = ProfitLockTrailingManager()
        first = manager.evaluate(position=PositionLike(7, "BUY", 1.0, 100.0), current_price=110.0, now=100.0)
        second = manager.evaluate(position=PositionLike(7, "BUY", 1.0, 100.0), current_price=120.0, now=110.0)

        self.assertTrue(first.should_modify)
        self.assertTrue(second.should_modify)
        self.assertAlmostEqual(second.sl_price or 0.0, 110.0)

    def test_stop_level_violation_blocks_sl(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(8, "BUY", 1.0, 100.0),
            current_price=102.0,
            now=0.0,
            symbol_spec=SymbolConstraints(point=0.01, trade_stops_level=250.0),
        )

        self.assertFalse(decision.should_modify)
        self.assertEqual(decision.reason, "BROKER_STOP_LEVEL_SL")

    def test_stop_level_violation_blocks_tp(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(9, "BUY", 1.0, 100.0),
            current_price=119.0,
            now=0.0,
            symbol_spec={"point": 1.0, "trade_stops_level": 2.0},
        )

        self.assertFalse(decision.should_modify)
        self.assertEqual(decision.reason, "BROKER_STOP_LEVEL_TP")

    def test_freeze_level_is_respected(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(10, "SELL", 1.0, 100.0),
            current_price=98.0,
            now=0.0,
            symbol_spec=SymbolConstraints(point=0.01, trade_freeze_level=250.0),
        )

        self.assertFalse(decision.should_modify)
        self.assertEqual(decision.reason, "BROKER_STOP_LEVEL_SL")

    def test_exit_costs_make_net_zero_lock_price_fee_aware(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(11, "BUY", 1.0, 100.0),
            current_price=103.0,
            now=0.0,
            estimated_exit_cost=1.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.net_unrealized_pnl or 0.0, 2.0)
        self.assertAlmostEqual(decision.sl_price or 0.0, 101.0)

    def test_explicit_net_unrealized_pnl_drives_threshold(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(12, "BUY", 1.0, 100.0),
            current_price=106.0,
            now=0.0,
            net_unrealized_pnl=5.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertEqual(decision.trigger_net_profit, 5.0)
        self.assertAlmostEqual(decision.sl_price or 0.0, 102.0)

    def test_contract_size_and_volume_convert_profit_to_price_distance(self) -> None:
        decision = ProfitLockTrailingManager(min_seconds_between_sltp_updates=0).evaluate(
            position=PositionLike(13, "BUY", 0.5, 100.0),
            current_price=106.0,
            now=0.0,
            contract_size=2.0,
        )

        self.assertTrue(decision.should_modify)
        self.assertAlmostEqual(decision.net_unrealized_pnl or 0.0, 6.0)
        self.assertAlmostEqual(decision.sl_price or 0.0, 102.0)


if __name__ == "__main__":
    unittest.main()
