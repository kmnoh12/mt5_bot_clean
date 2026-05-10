from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd


def find_daily_levels(daily_bars: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (PDH, PDL) from daily bars.
    Assumes last bar is current day, second to last is previous day.
    """
    if daily_bars is None or len(daily_bars) < 2:
        return None, None
    
    # Second to last bar is the complete 'previous day'
    prev_day = daily_bars.iloc[-2]
    return float(prev_day["high"]), float(prev_day["low"])

def is_liquidity_sweep(
    current_bar: pd.Series, 
    pdh: Optional[float], 
    pdl: Optional[float]
) -> Tuple[bool, bool]:
    """
    Checks if the current bar swept liquidity.
    Returns (swept_high, swept_low)
    """
    swept_high = False
    swept_low = False
    
    if pdh is not None:
        if current_bar["high"] > pdh and current_bar["close"] < pdh:
            swept_high = True
            
    if pdl is not None:
        if current_bar["low"] < pdl and current_bar["close"] > pdl:
            swept_low = True
            
    return swept_high, swept_low

def detect_institutional_sweep(
    current_bar: pd.Series,
    pdh: Optional[float],
    pdl: Optional[float]
) -> Tuple[bool, bool]:
    """
    Detects institutional liquidity sweep + rejection.
    BUY: Wick below PDL, close above PDL + Bullish candle.
    SELL: Wick above PDH, close below PDH + Bearish candle.
    """
    buy_sweep = False
    sell_sweep = False
    
    is_bullish = current_bar["close"] > current_bar["open"]
    is_bearish = current_bar["close"] < current_bar["open"]
    
    if pdl is not None:
        if current_bar["low"] < pdl and current_bar["close"] > pdl and is_bullish:
            buy_sweep = True
            
    if pdh is not None:
        if current_bar["high"] > pdh and current_bar["close"] < pdh and is_bearish:
            sell_sweep = True
            
    return buy_sweep, sell_sweep
