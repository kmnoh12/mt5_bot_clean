from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _sanitize_ohlc(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty:
        return None

    required = ("open", "high", "low", "close")
    missing = [col for col in required if col not in frame.columns]
    if missing:
        LOGGER.warning("Indicator input missing OHLC columns: %s", missing)
        return None

    clean = frame.copy()
    for col in required:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean.dropna(subset=list(required), inplace=True)
    if clean.empty:
        return None
    return clean.reset_index(drop=True)


def last_close(frame: pd.DataFrame) -> Optional[float]:
    clean = _sanitize_ohlc(frame)
    if clean is None or clean.empty:
        return None
    value = clean["close"].iloc[-1]
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) else None


def compute_atr(frame: pd.DataFrame, period: int = 14) -> Optional[float]:
    clean = _sanitize_ohlc(frame)
    period = max(1, int(period))
    if clean is None or len(clean) < period + 1:
        return None

    high = clean["high"]
    low = clean["low"]
    prev_close = clean["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    try:
        value = float(atr)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def compute_rsi(frame: pd.DataFrame, period: int = 14) -> Optional[float]:
    clean = _sanitize_ohlc(frame)
    period = max(1, int(period))
    if clean is None or len(clean) < period + 1:
        return None

    delta = clean["close"].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return None

    if avg_loss == 0:
        if avg_gain == 0:
            return 50.0
        return 100.0

    rs = float(avg_gain) / float(avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi if math.isfinite(rsi) else None


def compute_bollinger_bands(
    frame: pd.DataFrame,
    period: int = 20,
    stddev: float = 2.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    clean = _sanitize_ohlc(frame)
    period = max(1, int(period))
    stddev = max(0.1, float(stddev))
    if clean is None or len(clean) < period:
        return None, None, None

    rolling = clean["close"].rolling(period)
    middle = rolling.mean().iloc[-1]
    sigma = rolling.std(ddof=0).iloc[-1]
    if pd.isna(middle) or pd.isna(sigma):
        return None, None, None

    upper = float(middle) + (stddev * float(sigma))
    lower = float(middle) - (stddev * float(sigma))
    return float(middle), upper, lower


def rolling_high_low(frame: pd.DataFrame, lookback: int) -> Tuple[Optional[float], Optional[float]]:
    clean = _sanitize_ohlc(frame)
    lookback = max(1, int(lookback))
    if clean is None or len(clean) < lookback:
        return None, None

    window = clean.tail(lookback)
    high_value = window["high"].max()
    low_value = window["low"].min()
    try:
        high = float(high_value)
        low = float(low_value)
    except (TypeError, ValueError):
        return None, None
    if not (math.isfinite(high) and math.isfinite(low)):
        return None, None
    return high, low
