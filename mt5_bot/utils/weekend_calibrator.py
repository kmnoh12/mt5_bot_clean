# weekend_calibrator.py
from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5

def _is_crypto_symbol(symbol: str) -> bool:
    # Heuristic: XM symbol naming varies; keep a compact token list.
    crypto_tokens = [
        "BTC",
        "ETH",
        "SOL",
        "XBT",
        "BNB",
        "XRP",
        "ADA",
        "DOGE",
        "AVAX",
        "MATIC",
        "DOT",
        "LTC",
        "LINK",
    ]
    s = symbol.upper()
    return any(token in s for token in crypto_tokens)

def _atr14(high, low, close):
    # simple ATR(14) using SMA of TR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    if len(tr) < 15:
        return np.nan
    atr = np.convolve(tr, np.ones(14)/14, mode="valid")
    return np.concatenate([np.full(13, np.nan), atr])

def compute_weekend_corrections(symbol: str, lookback_days: int = 56):
    """
    Compute L/S/V factors and return multipliers for LSR strategy.
    """
    # MT5 rates: fields include time, high, low, close, tick_volume, spread
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    if rates is None or len(rates) < 5000:
        return None
    
    t = np.array([datetime.fromtimestamp(int(x), timezone.utc) for x in rates["time"]])
    dow = np.array([x.weekday() for x in t]) # 0=Mon ... 5=Sat 6=Sun
    hours = np.array([x.hour for x in t])
    # Treat Mon 00:00-02:00 UTC as "weekend risk" (thin liquidity/wide spreads)
    is_weekend = (dow >= 5) | ((dow == 0) & (hours < 2))
    
    spread = rates["spread"].astype(float) # points
    tv = rates["tick_volume"].astype(float)
    high = rates["high"].astype(float)
    low = rates["low"].astype(float)
    close = rates["close"].astype(float)
    
    atr = _atr14(high, low, close)
    
    # medians
    def med(x, mask):
        v = x[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.nan
        return float(np.median(v))
        
    L = med(tv, is_weekend) / med(tv, ~is_weekend)
    S = med(spread, is_weekend) / med(spread, ~is_weekend)
    V = med(atr, is_weekend) / med(atr, ~is_weekend)
    
    # clamps
    def clamp(a, b, x):
        return max(a, min(b, x))
        
    # Apply a single conservative correction model across symbols.
    # If crypto should be treated differently, handle it in strategy-layer gating.
    _ = _is_crypto_symbol(symbol)

    k_pen = clamp(1.10, 1.60, 1 + 0.6*(S-1) + 0.8*(1-L))

    t_rev = int(clamp(240, 720, 300 * (1/max(L, 0.35))**0.5))

    k_risk = clamp(0.10, 1.00, 0.85 * (L / max(S, 0.8)))
    k_sl_mult = 1.0

    k_disp = clamp(1.0, 2.0, 1.0 + 0.6 * (1.0 - L))

    return {
        "L_liquidity": L,
        "S_spread": S,
        "V_atr": V,
        "k_penetration": k_pen,
        "t_reversion_sec": t_rev,
        "k_risk": k_risk,
        "k_displacement": k_disp,
        "k_sl_mult": k_sl_mult,
    }


def is_chop_market(symbol: str, lookback_days: int = 56, atr_ratio_threshold: float = 0.50) -> bool:
    """
    Return True when weekend volatility is too low versus weekdays.
    Restriction rule: weekend ATR / weekday ATR < 50%.
    """
    metrics = compute_weekend_corrections(symbol=symbol, lookback_days=lookback_days)
    if not metrics:
        return False

    atr_ratio = metrics.get("V_atr")
    try:
        atr_ratio = float(atr_ratio)
    except Exception:
        return False

    if not np.isfinite(atr_ratio):
        return False
    return atr_ratio < float(atr_ratio_threshold)
