from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


def sanitize_ohlc(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty:
        return None
    required = ("open", "high", "low", "close")
    if any(col not in frame.columns for col in required):
        LOGGER.warning("OHLC frame missing required columns.")
        return None
    out = frame.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.dropna(subset=list(required), inplace=True)
    if out.empty:
        return None
    return out.reset_index(drop=True)


def parse_bar_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime().astimezone(timezone.utc)
    return None


def closed_bar_time(frame: pd.DataFrame) -> Optional[datetime]:
    if frame is None or frame.empty or len(frame) < 2:
        return None
    if "time" not in frame.columns:
        return None
    return parse_bar_time(frame.iloc[-2].get("time"))


def last_close(frame: pd.DataFrame) -> Optional[float]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None
    try:
        value = float(clean["close"].iloc[-1])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def compute_rsi(frame: pd.DataFrame, period: int = 14) -> Optional[float]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None
    period = max(1, int(period))
    if len(clean) < period + 1:
        return None
    delta = clean["close"].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]
    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return None
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = float(avg_gain) / float(avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi if math.isfinite(rsi) else None


def compute_ema(frame: pd.DataFrame, period: int = 20) -> Optional[float]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None
    period = max(1, int(period))
    if len(clean) < period:
        return None
    value = clean["close"].ewm(span=period, adjust=False).mean().iloc[-1]
    try:
        ema = float(value)
    except (TypeError, ValueError):
        return None
    return ema if math.isfinite(ema) else None


def compute_adx(frame: pd.DataFrame, period: int = 14) -> Optional[float]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None
    period = max(2, int(period))
    if len(clean) < (period * 2):
        return None

    high = clean["high"]
    low = clean["low"]
    close = clean["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    safe_atr = atr.replace(0.0, float("nan"))
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / safe_atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / safe_atr

    denom = (plus_di + minus_di).replace(0.0, float("nan"))
    dx = ((plus_di - minus_di).abs() / denom) * 100.0
    adx_series = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    value = adx_series.iloc[-1]
    try:
        adx = float(value)
    except (TypeError, ValueError):
        return None
    return adx if math.isfinite(adx) else None


def compute_atr(frame: pd.DataFrame, period: int = 14) -> Optional[float]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None
    period = max(1, int(period))
    if len(clean) < period + 1:
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
    value = tr.rolling(period).mean().iloc[-1]
    try:
        atr = float(value)
    except (TypeError, ValueError):
        return None
    return atr if math.isfinite(atr) and atr > 0 else None


def compute_bollinger_bands(
    frame: pd.DataFrame,
    period: int = 20,
    stddev: float = 2.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None, None, None
    period = max(1, int(period))
    stddev = max(0.1, float(stddev))
    if len(clean) < period:
        return None, None, None
    close = clean["close"]
    middle = close.rolling(period).mean().iloc[-1]
    sigma = close.rolling(period).std(ddof=0).iloc[-1]
    if pd.isna(middle) or pd.isna(sigma):
        return None, None, None
    upper = float(middle) + (stddev * float(sigma))
    lower = float(middle) - (stddev * float(sigma))
    return float(middle), upper, lower


def compute_vwap_bands(
    frame: pd.DataFrame,
    std_multiplier: float = 1.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None, None, None

    volume_col = next((col for col in ("volume", "tick_volume", "real_volume") if col in clean.columns), None)
    if volume_col is None:
        LOGGER.warning("VWAP frame missing volume column.")
        return None, None, None

    try:
        multiplier = float(std_multiplier)
        if not math.isfinite(multiplier):
            return None, None, None
        multiplier = max(0.0, multiplier)

        work = clean.copy()
        work["typical_price"] = (work["high"] + work["low"] + work["close"]) / 3.0
        work[volume_col] = pd.to_numeric(work[volume_col], errors="coerce")
        work[volume_col] = work[volume_col].where(work[volume_col] >= 0.0, float("nan"))
        work = work[work["typical_price"].notna()].copy()
        if work.empty:
            return None, None, None
        if (work[volume_col].fillna(0.0) <= 0.0).all():
            LOGGER.warning("VWAP frame has no positive volume values.")
            return None, None, None
        work[volume_col] = work[volume_col].fillna(0.0)

        session_key: pd.Series
        if "time" in work.columns:
            session_key = pd.to_datetime(work["time"], utc=True, errors="coerce").dt.date
        else:
            session_key = pd.Series(0, index=work.index)

        volume = work[volume_col].astype(float)
        typical = work["typical_price"].astype(float)
        weighted_price = volume * typical
        weighted_price_sq = volume * (typical * typical)

        cum_volume = volume.groupby(session_key, dropna=False).cumsum()
        safe_volume = cum_volume.replace(0.0, float("nan"))
        cum_weighted_price = weighted_price.groupby(session_key, dropna=False).cumsum()
        cum_weighted_price_sq = weighted_price_sq.groupby(session_key, dropna=False).cumsum()

        vwap_series = cum_weighted_price / safe_volume
        variance = (cum_weighted_price_sq / safe_volume) - (vwap_series * vwap_series)
        sigma_series = variance.clip(lower=0.0).pow(0.5)

        vwap_raw = vwap_series.iloc[-1]
        sigma_raw = sigma_series.iloc[-1]
        if pd.isna(vwap_raw) or pd.isna(sigma_raw):
            return None, None, None

        vwap = float(vwap_raw)
        sigma = float(sigma_raw)
        if not (math.isfinite(vwap) and math.isfinite(sigma)):
            return None, None, None

        upper = vwap + (multiplier * sigma)
        lower = vwap - (multiplier * sigma)
        return vwap, upper, lower
    except Exception:
        LOGGER.exception("Failed to compute VWAP bands.")
        return None, None, None


def rolling_high_low(frame: pd.DataFrame, lookback: int) -> Tuple[Optional[float], Optional[float]]:
    clean = sanitize_ohlc(frame)
    if clean is None:
        return None, None
    lookback = max(1, int(lookback))
    if len(clean) < lookback:
        return None, None
    window = clean.tail(lookback)
    high = window["high"].max()
    low = window["low"].min()
    try:
        high_value = float(high)
        low_value = float(low)
    except (TypeError, ValueError):
        return None, None
    if not (math.isfinite(high_value) and math.isfinite(low_value)):
        return None, None
    return high_value, low_value
