import unittest
from typing import Tuple

import pandas as pd

from core.models import DecisionAction, OrderResult, Position, Side, StrategyDecision, StrategyEvaluationContext, StrategyState
from strategies.factory import build_strategies
from strategies.liquidity_sweep_reversal import LiquiditySweepReversalStrategy


class LiquiditySweepReversalStrategyTests(unittest.TestCase):
    def _build_strategy(self) -> LiquiditySweepReversalStrategy:
        return LiquiditySweepReversalStrategy(
            config={
                "enabled": True,
                "atr_period": 3,
                "pivot_lookback_sec": 900,
                "swing_window": 5,
                "sweep_buffer_atr": 0.15,
                "reclaim_buffer_atr": 0.05,
                "reclaim_window_sec": 120,
                "displacement_mult": 1.2,
                "displacement_lookback": 5,
                "sl_atr_mult": 0.6,
                "stop_buffer_atr": 0.05,
                "tp_R1": 1.2,
                "tp_R2": 2.5,
                "be_at_R": 1.0,
                "max_hold_bars": 100,
                "min_hold_bars": 1,
                "min_cooldown_bars": 1,
                "fvg_enabled": False,
                "retest_enabled": False,
                "trail_tp_enabled": False,
            }
        )

    @staticmethod
    def _to_live_frame(closed: pd.DataFrame) -> pd.DataFrame:
        out = closed.copy()
        last_time = pd.to_datetime(out.iloc[-1]["time"], utc=True)
        last_close = float(out.iloc[-1]["close"])
        forming = pd.DataFrame(
            {
                "time": [last_time + pd.Timedelta(minutes=1)],
                "open": [last_close],
                "high": [last_close + 0.2],
                "low": [last_close - 0.2],
                "close": [last_close],
            }
        )
        return pd.concat([out, forming], ignore_index=True)

    @staticmethod
    def _base_closed(periods: int, start: str, start_price: float = 100.0, step: float = 0.03) -> pd.DataFrame:
        closes = [start_price + (i * step) for i in range(periods)]
        opens = [closes[0] - 0.05]
        opens.extend(closes[:-1])
        highs = [max(o, c) + 0.20 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.20 for o, c in zip(opens, closes)]
        times = pd.date_range(start=start, periods=periods, freq="min", tz="UTC")
        return pd.DataFrame(
            {
                "time": times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            }
        )

    def _build_sweep_then_reclaim_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        closed = self._base_closed(periods=30, start="2026-01-01T00:00:00Z")

        # Sweep bar: takes prior low, but fails to reclaim.
        closed.loc[29, "open"] = 100.6
        closed.loc[29, "high"] = 100.9
        closed.loc[29, "low"] = 98.2
        closed.loc[29, "close"] = 98.6

        frame_sweep = self._to_live_frame(closed)

        # Reclaim bar: bullish displacement back above liquidity level.
        reclaim_time = pd.to_datetime(closed.iloc[-1]["time"], utc=True) + pd.Timedelta(minutes=1)
        reclaim = pd.DataFrame(
            {
                "time": [reclaim_time],
                "open": [98.7],
                "high": [102.2],
                "low": [98.5],
                "close": [101.7],
            }
        )
        closed_reclaim = pd.concat([closed, reclaim], ignore_index=True)
        frame_reclaim = self._to_live_frame(closed_reclaim)
        return frame_sweep, frame_reclaim

    @staticmethod
    def _shift_frame_time(frame: pd.DataFrame, *, minutes: int) -> pd.DataFrame:
        shifted = frame.copy()
        shifted["time"] = pd.to_datetime(shifted["time"], utc=True) + pd.Timedelta(minutes=minutes)
        return shifted

    def _enter_long(self, strategy: LiquiditySweepReversalStrategy) -> Tuple[StrategyDecision, float, float]:
        frame_sweep, frame_reclaim = self._build_sweep_then_reclaim_frames()
        first = strategy.evaluate(symbol="GOLD", bars=frame_sweep, position=None)
        self.assertEqual(first.action, DecisionAction.HOLD)
        self.assertEqual(strategy.get_symbol_state("GOLD").state, StrategyState.SETUP)

        context = StrategyEvaluationContext(
            equity=1000.0,
            equity_peak=1200.0,
            loss_streak=2,
            daily_pnl=-25.0,
        )
        entry = strategy.evaluate(symbol="GOLD", bars=frame_reclaim, position=None, context=context)
        self.assertEqual(entry.action, DecisionAction.BUY)
        self.assertIsNotNone(entry.sl)
        self.assertIsNotNone(entry.tp)
        self.assertIn("win_probability", entry.metadata)
        self.assertIn("payoff_ratio", entry.metadata)
        self.assertIn("expected_rr", entry.metadata)
        self.assertIn("volume_scale", entry.metadata)
        self.assertIn("risk_per_unit", entry.metadata)
        entry_price = float(entry.metadata["signal_close"])
        risk = float(entry.metadata["risk_per_unit"])
        return entry, entry_price, risk

    def _management_frame(self, entry_price: float, final_close: float, start: str) -> pd.DataFrame:
        periods = 30
        closes = [entry_price + ((final_close - entry_price) * (i / float(periods - 1))) for i in range(periods)]
        opens = [closes[0] - 0.1]
        opens.extend(closes[:-1])
        highs = [max(o, c) + 0.25 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.25 for o, c in zip(opens, closes)]
        times = pd.date_range(start=start, periods=periods, freq="min", tz="UTC")
        closed = pd.DataFrame(
            {
                "time": times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            }
        )
        return self._to_live_frame(closed)

    def test_sweep_reclaim_entry_emits_risk_engine_hints(self) -> None:
        strategy = self._build_strategy()
        frame_sweep, frame_reclaim = self._build_sweep_then_reclaim_frames()

        first = strategy.evaluate(symbol="GOLD", bars=frame_sweep, position=None)
        self.assertEqual(first.action, DecisionAction.HOLD)
        self.assertEqual(first.reason, "LSR_WAIT_BUY_RECLAIM")
        self.assertEqual(strategy.get_symbol_state("GOLD").state, StrategyState.SETUP)

        context = StrategyEvaluationContext(equity=1000.0, equity_peak=1200.0, loss_streak=3, daily_pnl=-30.0)
        entry = strategy.evaluate(symbol="GOLD", bars=frame_reclaim, position=None, context=context)

        self.assertEqual(entry.action, DecisionAction.BUY)
        self.assertEqual(entry.reason, "LSR_BUY_ENTRY")
        self.assertLess(float(entry.sl), float(entry.metadata["signal_close"]))
        self.assertGreater(float(entry.tp), float(entry.metadata["signal_close"]))
        self.assertIn("win_probability", entry.metadata)
        self.assertIn("payoff_ratio", entry.metadata)
        self.assertIn("expected_rr", entry.metadata)
        self.assertIn("volume_scale", entry.metadata)
        self.assertGreater(float(entry.metadata["expected_rr"]), 1.0)
        self.assertGreater(float(entry.metadata["win_probability"]), 0.0)
        self.assertEqual(entry.metadata["reclaim_quality"]["confirmation_path"], "reclaim_only")
        self.assertFalse(entry.metadata["reclaim_quality"]["retest_confirmed"])
        self.assertEqual(entry.metadata["lsr_confirmation_flags"]["review_bucket"], "unconfirmed_reclaim_chase")
        self.assertTrue(entry.metadata["lsr_confirmation_flags"]["lsr_unconfirmed_reclaim"])
        self.assertTrue(entry.metadata["lsr_confirmation_flags"]["weak_reclaim_after_deep_sweep"])
        self.assertIn("confirmation_score", entry.metadata["lsr_confirmation_flags"])
        self.assertEqual(
            entry.metadata["reclaim_quality"]["confirmation_flags"]["confirmation_score"],
            entry.metadata["lsr_confirmation_flags"]["confirmation_score"],
        )
        self.assertGreaterEqual(float(entry.metadata["time_from_sweep_to_reclaim_sec"]), 0.0)
        self.assertGreater(float(entry.metadata["reclaim_quality"]["reclaim_distance_atr"]), 0.0)
        self.assertGreater(float(entry.metadata["reclaim_quality"]["sweep_depth_atr"]), 0.0)
        self.assertIsNotNone(entry.metadata.get("signal_reclaim_time_utc"))

    def test_blocks_reentry_for_same_sweep_event_key(self) -> None:
        strategy = self._build_strategy()
        frame_sweep, frame_reclaim = self._build_sweep_then_reclaim_frames()

        first = strategy.evaluate(symbol="GOLD", bars=frame_sweep, position=None)
        self.assertEqual(first.action, DecisionAction.HOLD)
        st = strategy.get_symbol_state("GOLD")
        pending_snapshot = dict(st.metadata.get("pending_sweep", {}))
        self.assertTrue(bool(pending_snapshot))

        entry = strategy.evaluate(symbol="GOLD", bars=frame_reclaim, position=None)
        self.assertEqual(entry.action, DecisionAction.BUY)
        self.assertEqual(entry.reason, "LSR_BUY_ENTRY")

        st = strategy.get_symbol_state("GOLD")
        st.pending_order = False
        st.state = StrategyState.SETUP
        st.metadata["pending_sweep"] = dict(pending_snapshot)

        blocked = strategy.evaluate(symbol="GOLD", bars=frame_reclaim, position=None)
        self.assertEqual(blocked.action, DecisionAction.HOLD)
        self.assertEqual(blocked.reason, "LSR_DUPLICATE_GHOST_RECONCILIATION_SWEEP_EVENT")
        self.assertIn("sweep_event_key", blocked.metadata)

    def test_blocks_same_side_reentry_within_2_bars_after_close(self) -> None:
        strategy = self._build_strategy()
        entry_decision, entry_price, _ = self._enter_long(strategy)
        strategy.apply_order_result(
            symbol="GOLD",
            decision=entry_decision,
            result=OrderResult(ok=True, status="FILLED", filled_price=entry_price),
        )

        exit_decision = StrategyDecision(
            action=DecisionAction.EXIT,
            reason="TEST_EXIT",
            strategy=strategy.name,
        )
        strategy.apply_order_result(
            symbol="GOLD",
            decision=exit_decision,
            result=OrderResult(ok=True, status="FILLED", filled_price=entry_price),
        )

        st = strategy.get_symbol_state("GOLD")
        self.assertEqual(st.metadata.get("last_close_side"), Side.BUY.value)
        self.assertIn("last_close_bar_time_utc", st.metadata)

        st.state = StrategyState.IDLE
        st.cooldown_bars_remaining = 0
        st.pending_order = False

        frame_sweep, frame_reclaim = self._build_sweep_then_reclaim_frames()
        frame_sweep_shifted = self._shift_frame_time(frame_sweep, minutes=1)
        frame_reclaim_shifted = self._shift_frame_time(frame_reclaim, minutes=1)

        setup = strategy.evaluate(symbol="GOLD", bars=frame_sweep_shifted, position=None)
        self.assertEqual(setup.action, DecisionAction.HOLD)
        self.assertEqual(strategy.get_symbol_state("GOLD").state, StrategyState.SETUP)

        blocked = strategy.evaluate(symbol="GOLD", bars=frame_reclaim_shifted, position=None)
        self.assertEqual(blocked.action, DecisionAction.HOLD)
        self.assertEqual(blocked.reason, "LSR_SAME_SIDE_REENTRY_LOCK_ACTIVE")
        self.assertEqual(int(blocked.metadata.get("reentry_lock_bars", -1)), 2)
        self.assertEqual(int(blocked.metadata.get("bars_since_close", -1)), 1)

    def test_stage_a_partial_close_then_break_even_hold(self) -> None:
        strategy = self._build_strategy()
        entry_decision, entry_price, risk = self._enter_long(strategy)
        strategy.apply_order_result(
            symbol="GOLD",
            decision=entry_decision,
            result=OrderResult(ok=True, status="FILLED", filled_price=entry_price),
        )

        position = Position(
            ticket=1,
            symbol="GOLD",
            side=Side.BUY,
            volume=0.10,
            price_open=entry_price,
            sl=entry_decision.sl,
            tp=entry_decision.tp,
            metadata={"min_volume": 0.01, "volume_step": 0.01},
        )

        frame_partial = self._management_frame(
            entry_price=entry_price,
            final_close=entry_price + (risk * 1.35),
            start="2026-01-01T03:00:00Z",
        )
        partial = strategy.evaluate(symbol="GOLD", bars=frame_partial, position=position)
        self.assertEqual(partial.action, DecisionAction.EXIT)
        self.assertEqual(partial.reason, "LSR_STAGE_A_PARTIAL_CLOSE")
        self.assertTrue(bool(partial.metadata.get("is_partial", False)))
        self.assertIsNotNone(partial.volume)
        assert partial.volume is not None
        self.assertGreater(float(partial.volume), 0.0)
        self.assertLess(float(partial.volume), float(position.volume))

        strategy.apply_order_result(
            symbol="GOLD",
            decision=partial,
            result=OrderResult(ok=True, status="CLOSED_PARTIAL", filled_price=entry_price + (risk * 1.2)),
        )
        st = strategy.get_symbol_state("GOLD")
        self.assertEqual(st.state, StrategyState.IN_POSITION)

        remaining = float(position.volume) - float(partial.volume)
        position_after_partial = Position(
            ticket=1,
            symbol="GOLD",
            side=Side.BUY,
            volume=max(0.01, remaining),
            price_open=entry_price,
            sl=position.sl,
            tp=position.tp,
            metadata={"min_volume": 0.01, "volume_step": 0.01},
        )

        frame_be = self._management_frame(
            entry_price=entry_price,
            final_close=entry_price + (risk * 1.1),
            start="2026-01-01T04:00:00Z",
        )
        managed = strategy.evaluate(symbol="GOLD", bars=frame_be, position=position_after_partial)
        self.assertEqual(managed.action, DecisionAction.HOLD)
        self.assertIsNotNone(managed.sl)
        assert managed.sl is not None
        self.assertGreaterEqual(float(managed.sl), float(entry_price))

    def test_exit_ready_emits_exit_at_r2(self) -> None:
        strategy = self._build_strategy()
        entry_decision, entry_price, risk = self._enter_long(strategy)
        strategy.apply_order_result(
            symbol="GOLD",
            decision=entry_decision,
            result=OrderResult(ok=True, status="FILLED", filled_price=entry_price),
        )

        position = Position(
            ticket=2,
            symbol="GOLD",
            side=Side.BUY,
            volume=0.01,
            price_open=entry_price,
            sl=entry_decision.sl,
            tp=entry_decision.tp,
            metadata={"min_volume": 0.01, "volume_step": 0.01},
        )
        frame_r2 = self._management_frame(
            entry_price=entry_price,
            final_close=entry_price + (risk * 2.7),
            start="2026-01-01T05:00:00Z",
        )
        exit_decision = strategy.evaluate(symbol="GOLD", bars=frame_r2, position=position)
        self.assertEqual(exit_decision.action, DecisionAction.EXIT)
        self.assertIn("LSR_EXIT", exit_decision.reason)
        self.assertIn("TP_R2", exit_decision.reason)

    def test_symbol_params_override_switches_per_symbol(self) -> None:
        strategy = LiquiditySweepReversalStrategy(
            config={
                "enabled": True,
                "reclaim_window_sec": 30,
                "displacement_mult": 1.5,
                "symbol_params": {
                    "BTCUSD": {
                        "reclaim_window_sec": 120,
                        "displacement_mult": 1.2,
                    }
                },
            }
        )
        frame_sweep, _ = self._build_sweep_then_reclaim_frames()

        strategy.evaluate(symbol="BTCUSD", bars=frame_sweep, position=None)
        self.assertEqual(strategy.reclaim_window_sec, 120)
        self.assertEqual(float(strategy.displacement_mult), 1.2)

        strategy.evaluate(symbol="GOLD", bars=frame_sweep, position=None)
        self.assertEqual(strategy.reclaim_window_sec, 30)
        self.assertEqual(float(strategy.displacement_mult), 1.5)

    def test_factory_registers_lsr_strategy(self) -> None:
        built = build_strategies(
            config={"strategies": {"liquidity_sweep_reversal": {"enabled": True}}},
            state_snapshot={},
        )
        self.assertIn("liquidity_sweep_reversal", built)
        self.assertIsInstance(built["liquidity_sweep_reversal"], LiquiditySweepReversalStrategy)


if __name__ == "__main__":
    unittest.main()
