from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.models import DecisionAction, Position, StrategyDecision, Side
from strategies.base import BaseStateMachineStrategy
from utils.indicators import compute_rsi, compute_ema, compute_atr, last_close, sanitize_ohlc

LOGGER = logging.getLogger(__name__)

class DeepWaveStrategy(BaseStateMachineStrategy):
    """
    DeepWave Strategy: Focuses on Macro/Micro wave alignment and mean-reversion at extremes.
    - Long: EMA uptrend + Micro-dip (RSI < 30 on small frame) + Reversal confirmation.
    - Short: EMA downtrend + Micro-pump (RSI > 70 on small frame) + Reversal confirmation.
    - News: Placeholder for external news impact scaling.
    """
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="deep_wave_sm", config=config, snapshot=snapshot)
        cfg = config or {}
        self.ema_period = int(cfg.get("ema_period", 50))
        self.rsi_period = int(cfg.get("rsi_period", 14))
        self.oversold = float(cfg.get("oversold", 30.0))
        self.overbought = float(cfg.get("overbought", 70.0))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.sl_mult = float(cfg.get("sl_mult", 2.0))
        self.tp_mult = float(cfg.get("tp_mult", 4.0))

    def _evaluate_impl(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: Optional[Position],
        st: Any,
    ) -> StrategyDecision:
        clean = sanitize_ohlc(bars)
        if clean is None or len(clean) < self.ema_period + 5:
            return self._hold("INSUFFICIENT_DATA")

        # Current Price & Indicators
        price = last_close(clean)
        ema = compute_ema(clean, self.ema_period)
        rsi = compute_rsi(clean, self.rsi_period)
        atr = compute_atr(clean, self.atr_period)

        if any(v is None for v in [price, ema, rsi, atr]):
            return self._hold("INDICATORS_PENDING")

        # Logic: Buy when "Hooks Down" (Dip) in an "Uptrend" (Macro)
        # Sell when "Hooks Up" (Pump) in a "Downtrend" (Macro)
        
        trend_up = price > ema
        trend_down = price < ema

        if position is None:
            # Long Setup: Uptrend + RSI Oversold (The Hook Down)
            if trend_up and rsi < self.oversold:
                sl = price - (atr * self.sl_mult)
                tp = price + (atr * self.tp_mult)
                return self._emit_entry(Side.BUY, "DEEP_WAVE_LONG_DIP", sl=sl, tp=tp)
            
            # Short Setup: Downtrend + RSI Overbought (The Hook Up)
            if trend_down and rsi > self.overbought:
                sl = price + (atr * self.sl_mult)
                tp = price - (atr * self.tp_mult)
                return self._emit_entry(Side.SELL, "DEEP_WAVE_SHORT_PUMP", sl=sl, tp=tp)
        
        return self._hold("WAITING_WAVE_ALIGNMENT")
