import unittest

from core.models import Position, Side
from execution.trailing_guard import DynamicTrailingProfitGuard


def _position(ticket: int, pnl: float) -> Position:
    return Position(
        ticket=ticket,
        symbol="BTCUSD",
        side=Side.BUY,
        volume=0.01,
        price_open=100.0,
        metadata={"floating_pnl": pnl},
    )


def _position_with_sl(ticket: int, pnl: float, side: Side, sl: float | None) -> Position:
    return Position(
        ticket=ticket,
        symbol="BTCUSD",
        side=side,
        volume=0.01,
        price_open=100.0,
        sl=sl,
        metadata={"floating_pnl": pnl},
    )


class DynamicTrailingProfitGuardTests(unittest.TestCase):
    def test_trigger_when_profit_drops_below_retain_ratio(self) -> None:
        guard = DynamicTrailingProfitGuard(
            config={
                "enabled": True, 
                "stage_b_retain_ratio": 0.8, 
                "min_activation_profit_usd": 5.0, 
                "min_breach_count_for_exit": 1,
                "min_drawdown_usd_for_exit": 0.0
            }
        )

        self.assertIsNone(guard.evaluate_position(_position(ticket=1, pnl=4.0)))
        self.assertIsNone(guard.evaluate_position(_position(ticket=1, pnl=10.0)))
        self.assertIsNone(guard.evaluate_position(_position(ticket=1, pnl=8.0)))

        signal = guard.evaluate_position(_position(ticket=1, pnl=7.99))
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.ticket, 1)
        self.assertAlmostEqual(signal.peak_pnl_usd, 10.0)
        self.assertAlmostEqual(signal.trigger_pnl_usd, 8.0)
        self.assertAlmostEqual(signal.current_pnl_usd, 7.99)
        self.assertAlmostEqual(signal.threshold_usd, 2.0)

    def test_min_activation_profit_must_be_met(self) -> None:
        guard = DynamicTrailingProfitGuard(
            config={"enabled": True, "retain_ratio": 0.8, "min_activation_profit_usd": 5.0}
        )

        self.assertIsNone(guard.evaluate_position(_position(ticket=2, pnl=4.99)))
        self.assertIsNone(guard.evaluate_position(_position(ticket=2, pnl=0.1)))

    def test_snapshot_and_drop_closed_positions(self) -> None:
        guard = DynamicTrailingProfitGuard(
            config={"enabled": True, "retain_ratio": 0.8, "min_activation_profit_usd": 5.0}
        )
        guard.evaluate_position(_position(ticket=1, pnl=15.0))
        guard.evaluate_position(_position(ticket=2, pnl=25.0))

        snapshot = guard.snapshot()
        restored = DynamicTrailingProfitGuard(
            config={"enabled": True, "retain_ratio": 0.8, "min_activation_profit_usd": 5.0},
            snapshot=snapshot,
        )

        self.assertIn("1", restored.snapshot()["peak_by_ticket"])
        self.assertIn("2", restored.snapshot()["peak_by_ticket"])

        restored.drop_closed_positions([_position(ticket=2, pnl=20.0)])
        self.assertNotIn("1", restored.snapshot()["peak_by_ticket"])
        self.assertIn("2", restored.snapshot()["peak_by_ticket"])

    def test_break_even_sync_signal_generated_after_activation(self) -> None:
        guard = DynamicTrailingProfitGuard(
            config={
                "enabled": True,
                "retain_ratio": 0.5,
                "min_activation_profit_usd": 25.0,
                "break_even_enabled": True,
                "break_even_activation_profit_usd": 3.0,
                "break_even_lock_profit_usd": 0.5,
                "break_even_sync_sl": True,
            }
        )
        pos = _position_with_sl(ticket=11, pnl=3.1, side=Side.BUY, sl=99.7)
        signal = guard.evaluate_break_even_sl(position=pos, contract_size=1.0)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertAlmostEqual(signal.desired_sl, 150.0)
        self.assertEqual(signal.reason, "profit_lock_stage_a_be")

    def test_break_even_does_not_lower_existing_sl(self) -> None:
        guard = DynamicTrailingProfitGuard(
            config={
                "break_even_enabled": True,
                "break_even_activation_profit_usd": 3.0,
                "break_even_lock_profit_usd": 0.5,
            }
        )
        pos = _position_with_sl(ticket=12, pnl=4.0, side=Side.BUY, sl=160.0)
        signal = guard.evaluate_break_even_sl(position=pos, contract_size=1.0)
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()
