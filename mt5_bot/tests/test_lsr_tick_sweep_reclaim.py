import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from core.models import DecisionAction, MarketTick, StrategyEvaluationContext
from strategies.liquidity_sweep_reversal_tick import LiquiditySweepReversalTickStrategy


def _make_bars(start_utc: datetime, n: int) -> pd.DataFrame:
    rows = []
    price = 100.0
    for i in range(n):
        t = start_utc + timedelta(minutes=i)
        rows.append(
            {
                "time": t,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
            }
        )
    return pd.DataFrame(rows)


class LsrTickSweepReclaimTests(unittest.TestCase):
    def test_sweep_reclaim_displacement_triggers_entry(self) -> None:
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        bars = _make_bars(base_time - timedelta(minutes=30), 40)
        ctx = StrategyEvaluationContext(mtf_info={"daily_reference": {"pdh": 100.0, "pdl": 90.0}})

        strat = LiquiditySweepReversalTickStrategy(
            config={
                "enabled": True,
                "atr_period": 3,
                "sweep_buffer_atr": 0.0,
                "reclaim_buffer_atr": 0.0,
                "stop_buffer_atr": 0.0,
                "reclaim_window_sec": 20,
                "reclaim_extension_sec": 0,
                "displacement_mult": 1.0,
                "displacement_window_sec": 1.0,
                "displacement_lookback_sec": 12.0,
                "tp_R1": 1.5,
                "tp_R2": 4.5,
                "be_at_R": 1.0,
                "tick_buffer_seconds": 60,
                "min_hold_bars": 1,
                "min_cooldown_bars": 1,
            }
        )

        history = []
        for i in range(-12, -2):
            history.append(MarketTick(time_utc=base_time + timedelta(seconds=i), last=100.0 + (i * 0.01)))
        strat.ingest_ticks("BTCUSD", history)
        decision = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)
        self.assertEqual(decision.action, DecisionAction.HOLD)

        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time + timedelta(seconds=-2), last=101.0)])
        decision = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)
        self.assertEqual(decision.action, DecisionAction.HOLD)
        self.assertEqual(decision.reason, "LSR_TICK_SWEEP_DETECTED")

        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time + timedelta(seconds=-1), last=101.0)])
        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time, last=99.0)])
        decision = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)
        self.assertEqual(decision.action, DecisionAction.SELL)
        self.assertIsNotNone(decision.sl)
        self.assertIsNotNone(decision.tp)
        self.assertGreater(float(decision.sl or 0.0), 99.0)
        self.assertLess(float(decision.tp or 0.0), 99.0)
        self.assertEqual(str(decision.metadata.get("sweep_level_name")), "PDH")
        self.assertGreaterEqual(float(decision.metadata.get("displacement_ratio", 0.0) or 0.0), 1.0)

    def test_reclaim_after_window_expires_does_not_enter(self) -> None:
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        bars = _make_bars(base_time - timedelta(minutes=30), 40)
        ctx = StrategyEvaluationContext(mtf_info={"daily_reference": {"pdh": 100.0, "pdl": 90.0}})

        strat = LiquiditySweepReversalTickStrategy(
            config={
                "enabled": True,
                "atr_period": 3,
                "sweep_buffer_atr": 0.0,
                "reclaim_buffer_atr": 0.0,
                "stop_buffer_atr": 0.0,
                "reclaim_window_sec": 20,
                "reclaim_extension_sec": 0,
                "displacement_mult": 1.0,
                "displacement_window_sec": 1.0,
                "displacement_lookback_sec": 12.0,
                "tick_buffer_seconds": 120,
            }
        )

        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time, last=101.0)])
        _ = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)

        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time + timedelta(seconds=21), last=99.0)])
        decision = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)
        self.assertEqual(decision.action, DecisionAction.HOLD)
        self.assertEqual(decision.reason, "RECLAIM_EXPIRED")

    def test_extension_allows_reclaim_until_45s(self) -> None:
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        bars = _make_bars(base_time - timedelta(minutes=30), 40)
        ctx = StrategyEvaluationContext(mtf_info={"daily_reference": {"pdh": 100.0, "pdl": 90.0}})

        strat = LiquiditySweepReversalTickStrategy(
            config={
                "enabled": True,
                "atr_period": 3,
                "sweep_buffer_atr": 0.0,
                "reclaim_buffer_atr": 0.0,
                "stop_buffer_atr": 0.0,
                "reclaim_window_sec": 20,
                "reclaim_extension_sec": 25,
                "displacement_mult": 1.0,
                "displacement_window_sec": 1.0,
                "displacement_lookback_sec": 12.0,
                "tick_buffer_seconds": 120,
            }
        )

        # Pre-fill small moves for displacement baseline.
        history = []
        for i in range(-15, 0):
            history.append(MarketTick(time_utc=base_time + timedelta(seconds=i), last=100.0 + (i * 0.01)))
        strat.ingest_ticks("BTCUSD", history)

        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time, last=101.0)])
        _ = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)

        # At +21s, extension should be granted.
        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time + timedelta(seconds=21), last=101.0)])
        decision = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)
        self.assertEqual(decision.action, DecisionAction.HOLD)
        self.assertEqual(decision.reason, "LSR_TICK_RECLAIM_WINDOW_EXTENDED")

        # Reclaim at +30s -> entry allowed.
        # Ingest small-moving ticks so the last-12s displacement baseline is non-zero.
        drift = []
        for sec in range(22, 30):
            drift.append(
                MarketTick(time_utc=base_time + timedelta(seconds=sec), last=101.0 + (sec - 22) * 0.01)
            )
        strat.ingest_ticks("BTCUSD", drift)
        strat.ingest_ticks("BTCUSD", [MarketTick(time_utc=base_time + timedelta(seconds=30), last=99.0)])
        decision = strat.evaluate(symbol="BTCUSD", bars=bars, position=None, context=ctx)
        self.assertEqual(decision.action, DecisionAction.SELL)


if __name__ == "__main__":
    unittest.main()
