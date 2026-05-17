from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from indicators import compute_atr, last_close, rolling_high_low


class VolBreakoutStrategy:
    name = "vol_breakout"

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config or {}
        self.lookback = int(cfg.get("lookback", 20))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.breakout_atr_multiple = float(cfg.get("breakout_atr_multiple", 1.0))

    @staticmethod
    def _hold(reason: str, atr: float | None = None, metrics: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "side": "HOLD",
            "reason": reason,
            "atr": atr,
            "metrics": metrics or {},
        }

    def generate(self, rates: pd.DataFrame) -> Dict[str, Any]:
        if rates is None or rates.empty:
            return self._hold("NO_DATA")

        # Skip current forming candle when possible.
        closed = rates.iloc[:-1].copy() if len(rates) > 1 else rates.copy()
        if len(closed) < self.lookback + 2:
            return self._hold("INSUFFICIENT_CANDLES")

        signal_candle = closed.iloc[-1:]
        reference = closed.iloc[-(self.lookback + 1) : -1]
        atr_value = compute_atr(closed, self.atr_period)
        close_price = last_close(signal_candle)
        range_high, range_low = rolling_high_low(reference, self.lookback)

        if atr_value is None:
            return self._hold("ATR_UNAVAILABLE")
        if None in (close_price, range_high, range_low):
            return self._hold("INSUFFICIENT_BREAKOUT_LEVELS", atr=atr_value)

        upper_breakout = range_high + (atr_value * self.breakout_atr_multiple)
        lower_breakout = range_low - (atr_value * self.breakout_atr_multiple)
        metrics = {
            "close": close_price,
            "range_high": range_high,
            "range_low": range_low,
            "upper_breakout": upper_breakout,
            "lower_breakout": lower_breakout,
        }

        if close_price > upper_breakout:
            return {
                "side": "BUY",
                "reason": "VOL_BREAKOUT_UP",
                "atr": atr_value,
                "metrics": metrics,
            }

        if close_price < lower_breakout:
            return {
                "side": "SELL",
                "reason": "VOL_BREAKOUT_DOWN",
                "atr": atr_value,
                "metrics": metrics,
            }

        return self._hold("BREAKOUT_NOT_CONFIRMED", atr=atr_value, metrics=metrics)
