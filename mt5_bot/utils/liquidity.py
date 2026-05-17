from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd


def _finite_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def classify_lsr_confirmation_quality(
    *,
    confirmation_path: Any,
    retest_confirmed: bool,
    reclaim_distance_atr: Any,
    sweep_depth_atr: Any,
    reclaim_to_sweep_depth_ratio: Any,
    displacement_ratio: Any,
    time_from_sweep_to_reclaim_sec: Any = None,
    reclaim_window_sec: Any = None,
    entered_into_exhaustion: bool = False,
    is_lsr: bool = True,
) -> Dict[str, Any]:
    """
    Review-only LSR confirmation diagnostics.

    This does not decide whether to trade. It separates clean retest/confirmation
    paths from unconfirmed reclaim entries so analytics do not tune them together.
    """
    path_key = str(confirmation_path or "").strip().lower()
    reclaim_atr = _finite_float(reclaim_distance_atr)
    depth_atr = _finite_float(sweep_depth_atr)
    reclaim_depth_ratio = _finite_float(reclaim_to_sweep_depth_ratio)
    displacement = _finite_float(displacement_ratio)
    reclaim_age_sec = _finite_float(time_from_sweep_to_reclaim_sec)
    reclaim_window = _finite_float(reclaim_window_sec)
    reclaim_window_elapsed_ratio = None
    invalid_reclaim_timing = bool(reclaim_age_sec is not None and reclaim_age_sec < 0.0)
    if (
        reclaim_age_sec is not None
        and reclaim_age_sec >= 0.0
        and reclaim_window is not None
        and reclaim_window > 0.0
    ):
        reclaim_window_elapsed_ratio = float(reclaim_age_sec / reclaim_window)

    lsr_unconfirmed_reclaim = bool(
        is_lsr
        and not bool(retest_confirmed)
        and path_key in {"", "reclaim_only", "tick_reclaim"}
    )
    shallow_reclaim_threshold_atr = 0.25
    shallow_reclaim_confirmation = bool(
        lsr_unconfirmed_reclaim
        and reclaim_atr is not None
        and reclaim_atr <= shallow_reclaim_threshold_atr
    )
    weak_reclaim_after_deep_sweep = bool(
        lsr_unconfirmed_reclaim
        and depth_atr is not None
        and depth_atr >= 0.5
        and (reclaim_depth_ratio is None or reclaim_depth_ratio < 1.0)
    )
    entry_chased_extension = bool(displacement is not None and displacement >= 2.0)
    late_window_reclaim = bool(
        lsr_unconfirmed_reclaim
        and reclaim_window_elapsed_ratio is not None
        and reclaim_window_elapsed_ratio >= 0.75
    )
    lsr_unconfirmed_reclaim_chase = bool(
        lsr_unconfirmed_reclaim
        and (
            entry_chased_extension
            or bool(entered_into_exhaustion)
            or shallow_reclaim_confirmation
            or weak_reclaim_after_deep_sweep
            or late_window_reclaim
        )
    )
    if bool(retest_confirmed):
        review_bucket = "confirmed_retest"
    elif lsr_unconfirmed_reclaim_chase:
        review_bucket = "unconfirmed_reclaim_chase"
    elif weak_reclaim_after_deep_sweep:
        review_bucket = "weak_reclaim_after_deep_sweep"
    elif lsr_unconfirmed_reclaim:
        review_bucket = "unconfirmed_reclaim"
    else:
        review_bucket = "unknown"

    confirmation_score = 0.50
    score_components: Dict[str, float] = {"base": confirmation_score}
    if bool(retest_confirmed):
        score_components["confirmed_retest"] = 0.30
    if reclaim_atr is not None:
        reclaim_distance_component = min(0.15, max(0.0, reclaim_atr) * 0.15)
        score_components["reclaim_distance_atr"] = reclaim_distance_component
    if reclaim_depth_ratio is not None:
        ratio_component = min(0.15, max(0.0, reclaim_depth_ratio) * 0.15)
        score_components["reclaim_to_sweep_depth_ratio"] = ratio_component
    if shallow_reclaim_confirmation:
        score_components["shallow_reclaim_confirmation"] = -0.15
    if weak_reclaim_after_deep_sweep:
        score_components["weak_reclaim_after_deep_sweep"] = -0.20
    if late_window_reclaim:
        score_components["late_window_reclaim"] = -0.15
    if entry_chased_extension:
        score_components["entry_chased_extension"] = -0.10
    if bool(entered_into_exhaustion):
        score_components["entered_into_exhaustion"] = -0.10
    if invalid_reclaim_timing:
        score_components["invalid_reclaim_timing"] = -0.25
    confirmation_score = min(1.0, max(0.0, sum(score_components.values())))
    if confirmation_score >= 0.70:
        confirmation_band = "clean"
    elif confirmation_score >= 0.40:
        confirmation_band = "mixed"
    else:
        confirmation_band = "weak"

    return {
        "confirmation_path_key": path_key or None,
        "confirmation_score": confirmation_score,
        "confirmation_band": confirmation_band,
        "confirmation_score_components": score_components,
        "reclaim_distance_atr": reclaim_atr,
        "sweep_depth_atr": depth_atr,
        "reclaim_to_sweep_depth_ratio": reclaim_depth_ratio,
        "time_from_sweep_to_reclaim_sec": reclaim_age_sec,
        "reclaim_window_sec": reclaim_window,
        "reclaim_window_elapsed_ratio": reclaim_window_elapsed_ratio,
        "invalid_reclaim_timing": invalid_reclaim_timing,
        "entry_chased_extension": entry_chased_extension,
        "late_window_reclaim": late_window_reclaim,
        "entered_into_exhaustion": bool(entered_into_exhaustion),
        "lsr_unconfirmed_reclaim": lsr_unconfirmed_reclaim,
        "shallow_reclaim_threshold_atr": shallow_reclaim_threshold_atr,
        "shallow_reclaim_confirmation": shallow_reclaim_confirmation,
        "weak_reclaim_after_deep_sweep": weak_reclaim_after_deep_sweep,
        "lsr_unconfirmed_reclaim_chase": lsr_unconfirmed_reclaim_chase,
        "review_bucket": review_bucket,
        "review_only": True,
    }


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
