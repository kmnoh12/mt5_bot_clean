import unittest

import pandas as pd

from core.models import DecisionAction
from execution.mtf_confirm import MtfDirectionConfirm


class MtfConfirmTests(unittest.TestCase):
    @staticmethod
    def _bars(closes: list[float]) -> pd.DataFrame:
        opens = [closes[0]]
        opens.extend(closes[:-1])
        highs = [max(o, c) + 0.1 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.1 for o, c in zip(opens, closes)]
        # Append still-forming bar
        closes2 = list(closes) + [closes[-1]]
        opens.append(closes[-1])
        highs.append(closes[-1] + 0.1)
        lows.append(closes[-1] - 0.1)
        times = pd.date_range(start="2026-01-01T00:00:00Z", periods=len(closes2), freq="min", tz="UTC")
        return pd.DataFrame({"time": times, "open": opens, "high": highs, "low": lows, "close": closes2})

    def test_buy_blocked_when_m5_trend_down(self) -> None:
        confirm = MtfDirectionConfirm(
            {
                "enabled": True,
                "symbols": ["BTCUSD"],
                "confirm_timeframe": "TIMEFRAME_M5",
                "fast_ema": 5,
                "slow_ema": 10,
            }
        )
        bars = self._bars([200.0 - (i * 1.0) for i in range(30)])
        self.assertFalse(confirm.allow_entry("BTCUSD", DecisionAction.BUY, bars))

    def test_sell_allowed_when_m5_trend_down(self) -> None:
        confirm = MtfDirectionConfirm(
            {
                "enabled": True,
                "symbols": ["BTCUSD"],
                "confirm_timeframe": "TIMEFRAME_M5",
                "fast_ema": 5,
                "slow_ema": 10,
            }
        )
        bars = self._bars([200.0 - (i * 1.0) for i in range(30)])
        self.assertTrue(confirm.allow_entry("BTCUSD", DecisionAction.SELL, bars))


if __name__ == "__main__":
    unittest.main()
