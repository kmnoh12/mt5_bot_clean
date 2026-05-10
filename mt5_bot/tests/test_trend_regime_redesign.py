import unittest
from typing import List

import pandas as pd

from core.models import DecisionAction, OrderResult, Position, Side, StrategyDecision, StrategyState
from strategies.trend_regime_sm import TrendRegimeStateMachine


class TrendRegimeRedesignTests(unittest.TestCase):
    def _build_strategy(
        self,
        fixed_tp_on_entry: bool = True,
        liquidity_grab_enabled: bool = False,
    ) -> TrendRegimeStateMachine:
        return TrendRegimeStateMachine(
            config={
                "enabled": True,
                "fast_ema_period": 3,
                "slow_ema_period": 6,
                "atr_period": 3,
                "adx_period": 3,
                "rsi_period": 3,
                "slope_lookback": 3,
                "breakout_lookback": 4,
                "meanrev_lookback": 6,
                "adx_floor": 8.0,
                "adx_ceiling": 30.0,
                "atr_pct_floor": 0.0001,
                "atr_pct_ceiling": 0.03,
                "trend_score_threshold": 0.12,
                "trend_strength_threshold": 0.12,
                "meanrev_max_strength": 0.35,
                "meanrev_entry_distance_atr": 0.6,
                "meanrev_rsi_oversold": 35.0,
                "meanrev_rsi_overbought": 65.0,
                "trend_sl_atr_mult": 1.0,
                "trend_tp_r_multiple": 2.0,
                "meanrev_sl_atr_mult": 1.0,
                "meanrev_tp_r_multiple": 1.4,
                "break_even_rr": 0.8,
                "break_even_offset_atr": 0.05,
                "trailing_atr_mult": 1.0,
                "trailing_start_rr": 0.5,
                "time_stop_bars": 50,
                "regime_flip_exit_threshold": 0.10,
                "min_hold_bars": 1,
                "min_cooldown_bars": 1,
                "fixed_tp_on_entry": fixed_tp_on_entry,
                "liquidity_grab_enabled": liquidity_grab_enabled,
            }
        )

    @staticmethod
    def _bars_from_closes(closes: List[float], start_time: str) -> pd.DataFrame:
        opens = [closes[0] - 0.4]
        opens.extend(closes[:-1])
        highs = [max(o, c) + 0.35 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.35 for o, c in zip(opens, closes)]

        # Append one still-forming bar so strategy must use only closed bars.
        opens.append(closes[-1])
        highs.append(closes[-1] + 0.2)
        lows.append(closes[-1] - 0.2)
        closes_ext = list(closes) + [closes[-1]]

        times = pd.date_range(start=start_time, periods=len(closes_ext), freq="min", tz="UTC")
        return pd.DataFrame(
            {
                "time": times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes_ext,
            }
        )

    def test_trend_breakout_generates_rich_entry(self) -> None:
        strategy = self._build_strategy()
        rising_closes = [100.0 + (i * 1.4) for i in range(24)]
        bars = self._bars_from_closes(rising_closes, "2026-01-01T00:00:00Z")

        decision = strategy.evaluate(symbol="BTCUSD", bars=bars, position=None)

        self.assertEqual(decision.action, DecisionAction.BUY)
        self.assertIsNotNone(decision.sl)
        self.assertIsNotNone(decision.tp)
        self.assertLess(float(decision.sl), float(decision.metadata["signal_close"]))
        self.assertGreater(float(decision.tp), float(decision.metadata["signal_close"]))

        self.assertIn("risk_per_unit", decision.metadata)
        self.assertGreater(float(decision.metadata["risk_per_unit"]), 0.0)
        self.assertIn("regime", decision.metadata)
        self.assertEqual(decision.metadata["regime"], "TREND_UP")
        self.assertIn("regime_score", decision.metadata)

        snapshot = decision.metadata.get("indicator_snapshot")
        self.assertIsInstance(snapshot, dict)
        self.assertIn("ema_fast", snapshot)
        self.assertIn("ema_slow", snapshot)
        self.assertIn("adx", snapshot)
        self.assertIn("atr", snapshot)

    def test_trend_breakout_can_disable_fixed_tp(self) -> None:
        strategy = self._build_strategy(fixed_tp_on_entry=False)
        rising_closes = [100.0 + (i * 1.4) for i in range(24)]
        bars = self._bars_from_closes(rising_closes, "2026-01-01T00:00:00Z")

        decision = strategy.evaluate(symbol="BTCUSD", bars=bars, position=None)

        self.assertEqual(decision.action, DecisionAction.BUY)
        self.assertIsNotNone(decision.sl)
        self.assertIsNone(decision.tp)

    def test_in_position_exits_on_regime_flip_context(self) -> None:
        strategy = self._build_strategy()

        entry_bars = self._bars_from_closes([100.0 + (i * 1.5) for i in range(24)], "2026-01-01T00:00:00Z")
        entry_decision = strategy.evaluate(symbol="BTCUSD", bars=entry_bars, position=None)
        self.assertEqual(entry_decision.action, DecisionAction.BUY)

        entry_price = float(entry_decision.metadata["signal_close"])
        position = Position(
            ticket=42,
            symbol="BTCUSD",
            side=Side.BUY,
            volume=0.1,
            price_open=entry_price,
            sl=entry_decision.sl,
            tp=entry_decision.tp,
        )

        strategy.apply_order_result(
            symbol="BTCUSD",
            decision=entry_decision,
            result=OrderResult(ok=True, status="FILLED", filled_price=entry_price),
        )

        reversal_bars = self._bars_from_closes([150.0 - (i * 2.2) for i in range(24)], "2026-01-01T01:00:00Z")
        exit_decision = strategy.evaluate(symbol="BTCUSD", bars=reversal_bars, position=position)

        self.assertEqual(exit_decision.action, DecisionAction.EXIT)
        self.assertIn("TREND_REGIME_EXIT", exit_decision.reason)
        self.assertTrue(
            (
                "REGIME_FLIP" in exit_decision.reason
                or "TIME_STOP" in exit_decision.reason
                or "TRAIL_BREACH" in exit_decision.reason
            ),
            msg=f"unexpected exit reason: {exit_decision.reason}",
        )

    def test_stage_a_partial_close_fires_once_and_keeps_in_position(self) -> None:
        strategy = self._build_strategy()
        entry_bars = self._bars_from_closes([100.0 + (i * 1.5) for i in range(24)], "2026-01-01T00:00:00Z")
        entry_decision = strategy.evaluate(symbol="BTCUSD", bars=entry_bars, position=None)
        self.assertEqual(entry_decision.action, DecisionAction.BUY)

        entry_price = float(entry_decision.metadata["signal_close"])
        position = Position(
            ticket=99,
            symbol="BTCUSD",
            side=Side.BUY,
            volume=0.1,
            price_open=entry_price,
            sl=entry_decision.sl,
            tp=entry_decision.tp,
        )
        strategy.apply_order_result(
            symbol="BTCUSD",
            decision=entry_decision,
            result=OrderResult(ok=True, status="FILLED", filled_price=entry_price),
        )

        manage_bars = self._bars_from_closes([entry_price + (i * 1.6) for i in range(24)], "2026-01-01T01:00:00Z")
        partial_decision = strategy.evaluate(symbol="BTCUSD", bars=manage_bars, position=position)
        self.assertEqual(partial_decision.action, DecisionAction.EXIT)
        self.assertEqual(partial_decision.reason, "STAGE_A_PARTIAL_CLOSE")
        self.assertIsNotNone(partial_decision.volume)
        assert partial_decision.volume is not None
        self.assertGreater(float(partial_decision.volume), 0.0)
        self.assertLess(float(partial_decision.volume), float(position.volume))
        self.assertTrue(bool(partial_decision.metadata.get("is_partial", False)))
        self.assertEqual(float(partial_decision.metadata.get("position_volume_before", 0.0)), float(position.volume))

        strategy.apply_order_result(
            symbol="BTCUSD",
            decision=partial_decision,
            result=OrderResult(ok=True, status="CLOSED_PARTIAL", filled_price=entry_price + 2.0),
        )
        st = strategy.get_symbol_state("BTCUSD")
        self.assertEqual(st.state, StrategyState.IN_POSITION)

        remaining = max(0.01, float(position.volume) - float(partial_decision.volume))
        position_after_partial = Position(
            ticket=99,
            symbol="BTCUSD",
            side=Side.BUY,
            volume=remaining,
            price_open=entry_price,
            sl=position.sl,
            tp=position.tp,
        )
        next_bars = self._bars_from_closes([entry_price + (i * 1.7) for i in range(24)], "2026-01-01T02:00:00Z")
        follow_up_decision = strategy.evaluate(symbol="BTCUSD", bars=next_bars, position=position_after_partial)
        self.assertNotEqual(follow_up_decision.reason, "STAGE_A_PARTIAL_CLOSE")

    def test_apply_order_result_partial_exit_detected_by_volume_hint(self) -> None:
        strategy = self._build_strategy()
        entry_bars = self._bars_from_closes([100.0 + (i * 1.5) for i in range(24)], "2026-01-01T00:00:00Z")
        entry_decision = strategy.evaluate(symbol="BTCUSD", bars=entry_bars, position=None)
        self.assertEqual(entry_decision.action, DecisionAction.BUY)
        entry_price = float(entry_decision.metadata["signal_close"])
        strategy.apply_order_result(
            symbol="BTCUSD",
            decision=entry_decision,
            result=OrderResult(ok=True, status="FILLED", filled_price=entry_price),
        )

        volume_hint_partial = StrategyDecision(
            action=DecisionAction.EXIT,
            reason="STAGE_A_PARTIAL_CLOSE",
            strategy="trend_regime_sm",
            volume=0.05,
            metadata={"position_volume_before": 0.1},
        )
        strategy.apply_order_result(
            symbol="BTCUSD",
            decision=volume_hint_partial,
            result=OrderResult(ok=True, status="CLOSED_PARTIAL", filled_price=entry_price + 1.0),
        )
        st = strategy.get_symbol_state("BTCUSD")
        self.assertEqual(st.state, StrategyState.IN_POSITION)

    def test_liquidity_grab_buy_sweep_generates_rich_entry(self) -> None:
        strategy = self._build_strategy(liquidity_grab_enabled=True)
        bars = self._bars_from_closes([100.0 + (i * 0.9) for i in range(24)], "2026-01-01T00:00:00Z")
        sweep_idx = len(bars) - 2
        bars.loc[sweep_idx, "open"] = 120.0
        bars.loc[sweep_idx, "high"] = 122.0
        bars.loc[sweep_idx, "low"] = 118.0
        bars.loc[sweep_idx, "close"] = 121.0
        bars.attrs["mtf_info"] = {"daily_reference": {"pdh": 130.0, "pdl": 118.5}}

        decision = strategy.evaluate(symbol="BTCUSD", bars=bars, position=None)

        self.assertEqual(decision.action, DecisionAction.BUY)
        self.assertEqual(decision.reason, "PDL_SWEEP_REJECTION")
        self.assertAlmostEqual(float(decision.confidence), 0.9, places=6)
        self.assertEqual(decision.metadata.get("entry_style"), "liquidity_grab")
        self.assertEqual(decision.metadata.get("liquidity_sweep", {}).get("buy_sweep"), True)
        self.assertLess(float(decision.sl), float(decision.metadata["signal_close"]))

    def test_liquidity_grab_sell_sweep_generates_rich_entry(self) -> None:
        strategy = self._build_strategy(liquidity_grab_enabled=True)
        bars = self._bars_from_closes([140.0 - (i * 0.8) for i in range(24)], "2026-01-01T00:00:00Z")
        sweep_idx = len(bars) - 2
        bars.loc[sweep_idx, "open"] = 121.0
        bars.loc[sweep_idx, "high"] = 124.0
        bars.loc[sweep_idx, "low"] = 119.0
        bars.loc[sweep_idx, "close"] = 120.0
        bars.attrs["mtf_info"] = {"daily_reference": {"pdh": 123.5, "pdl": 110.0}}

        decision = strategy.evaluate(symbol="BTCUSD", bars=bars, position=None)

        self.assertEqual(decision.action, DecisionAction.SELL)
        self.assertEqual(decision.reason, "PDH_SWEEP_REJECTION")
        self.assertAlmostEqual(float(decision.confidence), 0.9, places=6)
        self.assertEqual(decision.metadata.get("entry_style"), "liquidity_grab")
        self.assertEqual(decision.metadata.get("liquidity_sweep", {}).get("sell_sweep"), True)
        self.assertGreater(float(decision.sl), float(decision.metadata["signal_close"]))

    def test_runtime_override_can_enable_liquidity_grab(self) -> None:
        strategy = self._build_strategy(liquidity_grab_enabled=False)
        self.assertFalse(strategy.liquidity_grab_enabled)

        applied = strategy.apply_runtime_overrides({"liquidity_grab_enabled": "true"})

        self.assertEqual(applied.get("liquidity_grab_enabled"), True)
        self.assertTrue(strategy.liquidity_grab_enabled)


if __name__ == "__main__":
    unittest.main()
