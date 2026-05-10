import math
import unittest
from typing import Dict

import pandas as pd

from core.auto_tuning import ParameterAutoTuningLoop


def _build_bars(start: str, count: int, base: float, trend_step: float) -> pd.DataFrame:
    closes = []
    for idx in range(count):
        wave = math.sin(idx / 4.5) * 0.35
        closes.append(base + (idx * trend_step) + wave)

    opens = [closes[0]]
    opens.extend(closes[:-1])
    highs = [max(o, c) + 0.55 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.55 for o, c in zip(opens, closes)]

    opens.append(closes[-1])
    highs.append(closes[-1] + 0.25)
    lows.append(closes[-1] - 0.25)
    closes_ext = list(closes) + [closes[-1] + 0.05]

    times = pd.date_range(start=start, periods=len(closes_ext), freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes_ext,
        }
    )


def _make_config() -> Dict[str, object]:
    return {
        "auto_tuning": {
            "enabled": True,
            "target_symbols": ["BTCUSD", "ETHUSD"],
            "tune_interval_seconds": 120,
            "lookback_bars": 90,
            "min_bars": 50,
            "smoothing_alpha": 0.4,
            "parameter_bounds": {
                "trend_score_threshold": {"min": 0.1, "max": 0.8},
                "trend_strength_threshold": {"min": 0.15, "max": 0.85},
                "meanrev_max_strength": {"min": 0.15, "max": 0.8},
                "breakout_lookback": {"min": 3, "max": 20},
                "trend_sl_atr_mult": {"min": 0.7, "max": 3.0},
                "trend_tp_r_multiple": {"min": 0.9, "max": 4.5},
                "trailing_atr_mult": {"min": 0.5, "max": 2.5},
                "trailing_start_rr": {"min": 0.1, "max": 2.5},
                "regime_flip_exit_threshold": {"min": 0.05, "max": 0.45},
            },
        },
        "strategies": {
            "trend_regime_sm": {
                "atr_period": 14,
                "adx_period": 14,
                "trend_score_threshold": 0.32,
                "trend_strength_threshold": 0.42,
                "meanrev_max_strength": 0.40,
                "breakout_lookback": 5,
                "trend_sl_atr_mult": 1.2,
                "trend_tp_r_multiple": 2.1,
                "trailing_atr_mult": 1.0,
                "trailing_start_rr": 0.8,
                "regime_flip_exit_threshold": 0.18,
            }
        },
    }


class AutoTuningLoopTests(unittest.TestCase):
    def test_successful_update_produces_bounded_overrides(self) -> None:
        loop = ParameterAutoTuningLoop(config=_make_config())
        loop.ingest_symbol_bars("BTCUSD", _build_bars("2026-01-01T00:00:00Z", 120, 45000.0, 15.0))
        loop.ingest_symbol_bars("ETHUSD", _build_bars("2026-01-01T00:00:00Z", 120, 2600.0, 0.8))

        result = loop.step(now_ts=10_000.0)

        self.assertTrue(result["updated"])
        self.assertEqual(result["reason"], "updated")
        self.assertIn("metrics", result)
        self.assertIn("symbol_metrics", result)

        metrics = result["metrics"]
        self.assertIn("atr_percent", metrics)
        self.assertIn("return_volatility", metrics)
        self.assertIn("trend_persistence", metrics)
        self.assertIn("adx", metrics)
        self.assertIn("whipsaw_proxy", metrics)
        self.assertIn("chop_proxy", metrics)

        overrides = result["overrides"]
        self.assertEqual(set(overrides.keys()), set(ParameterAutoTuningLoop.TUNABLE_PARAMETERS))
        for name in ParameterAutoTuningLoop.TUNABLE_PARAMETERS:
            lo, hi = loop.bounds[name]
            value = float(overrides[name])
            self.assertGreaterEqual(value, lo, msg=f"{name} below min")
            self.assertLessEqual(value, hi, msg=f"{name} above max")

        snapshot = loop.snapshot()
        self.assertEqual(snapshot["update_count"], 1)
        self.assertEqual(snapshot["last_skip_reason"], "")

    def test_interval_gate_and_insufficient_bars_skip(self) -> None:
        loop = ParameterAutoTuningLoop(config=_make_config())
        btc_bars = _build_bars("2026-01-01T00:00:00Z", 120, 45000.0, 10.0)
        eth_bars = _build_bars("2026-01-01T00:00:00Z", 120, 2600.0, 0.5)
        loop.ingest_bars({"BTCUSD": btc_bars, "ETHUSD": eth_bars})

        first = loop.step(now_ts=20_000.0)
        self.assertTrue(first["updated"])

        gated = loop.step(now_ts=20_050.0)
        self.assertFalse(gated["updated"])
        self.assertEqual(gated["reason"], "interval_not_elapsed")
        self.assertGreater(float(gated.get("seconds_until_next", 0.0)), 0.0)

        insufficient_loop = ParameterAutoTuningLoop(config=_make_config())
        insufficient_loop.ingest_bars(
            {
                "BTCUSD": _build_bars("2026-01-02T00:00:00Z", 30, 47000.0, 8.0),
                "ETHUSD": _build_bars("2026-01-02T00:00:00Z", 30, 2800.0, 0.3),
            }
        )

        skipped = insufficient_loop.step(now_ts=30_000.0)
        self.assertFalse(skipped["updated"])
        self.assertEqual(skipped["reason"], "insufficient_bars")
        self.assertIn("missing_symbols", skipped)
        self.assertEqual(set(skipped["missing_symbols"]), {"BTCUSD", "ETHUSD"})


if __name__ == "__main__":
    unittest.main()
