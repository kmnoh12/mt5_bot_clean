from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from indicators import compute_atr, compute_bollinger_bands, compute_rsi, last_close


class MeanReversionStrategy:
    name = "mean_reversion"

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config or {}
        self.rsi_period = int(cfg.get("rsi_period", 14))
        self.rsi_buy_threshold = float(cfg.get("rsi_buy_threshold", 30))
        self.rsi_sell_threshold = float(cfg.get("rsi_sell_threshold", 70))
        self.bollinger_period = int(cfg.get("bollinger_period", 20))
        self.bollinger_stddev = float(cfg.get("bollinger_stddev", 2.0))
        self.atr_period = int(cfg.get("atr_period", 14))

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

        # Skip the current forming candle when possible.
        closed = rates.iloc[:-1].copy() if len(rates) > 1 else rates.copy()
        if closed.empty:
            return self._hold("NO_CLOSED_CANDLE")

        close_price = last_close(closed)
        rsi_value = compute_rsi(closed, self.rsi_period)
        middle, upper, lower = compute_bollinger_bands(closed, self.bollinger_period, self.bollinger_stddev)
        atr_value = compute_atr(closed, self.atr_period)

        metrics = {
            "close": close_price,
            "rsi": rsi_value,
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
        }

        if None in (close_price, rsi_value, middle, upper, lower):
            return self._hold("INSUFFICIENT_INDICATORS", atr=atr_value, metrics=metrics)

        if close_price <= lower and rsi_value <= self.rsi_buy_threshold:
            return {
                "side": "BUY",
                "reason": "PRICE_BELOW_LOWER_BB_AND_RSI_OVERSOLD",
                "atr": atr_value,
                "metrics": metrics,
            }

        if close_price >= upper and rsi_value >= self.rsi_sell_threshold:
            return {
                "side": "SELL",
                "reason": "PRICE_ABOVE_UPPER_BB_AND_RSI_OVERBOUGHT",
                "atr": atr_value,
                "metrics": metrics,
            }

        return self._hold("MEAN_REVERSION_CONDITIONS_NOT_MET", atr=atr_value, metrics=metrics)
