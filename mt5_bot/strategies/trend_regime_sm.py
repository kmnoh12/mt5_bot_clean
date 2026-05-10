from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.models import DecisionAction, Position, Side, StrategyDecision, StrategyState, StrategySymbolState
from strategies.base import BaseStateMachineStrategy
from utils.indicators import compute_adx, compute_atr, compute_ema, compute_rsi, last_close, parse_bar_time, sanitize_ohlc
from utils.liquidity import detect_institutional_sweep


class TrendRegimeStateMachine(BaseStateMachineStrategy):
    RUNTIME_OVERRIDE_KEYS = (
        "trend_score_threshold",
        "trend_strength_threshold",
        "meanrev_max_strength",
        "breakout_lookback",
        "trend_sl_atr_mult",
        "trend_tp_r_multiple",
        "trailing_atr_mult",
        "trailing_start_rr",
        "regime_flip_exit_threshold",
        "partial_close_rr",
        "liquidity_grab_enabled",
        "min_volume_ratio",
    )

    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="trend_regime_sm", config=config, snapshot=snapshot)
        cfg = config or {}

        self.fast_ema_period = max(2, int(cfg.get("fast_ema_period", 20)))
        self.slow_ema_period = max(self.fast_ema_period + 1, int(cfg.get("slow_ema_period", 80)))
        self.atr_period = max(5, int(cfg.get("atr_period", 14)))
        self.adx_period = max(5, int(cfg.get("adx_period", 14)))
        self.rsi_period = max(2, int(cfg.get("rsi_period", 14)))
        self.slope_lookback = max(2, int(cfg.get("slope_lookback", 8)))
        self.breakout_lookback = max(3, int(cfg.get("breakout_lookback", 5)))
        self.meanrev_lookback = max(4, int(cfg.get("meanrev_lookback", 10)))

        self.adx_floor = float(cfg.get("adx_floor", 14.0))
        self.adx_ceiling = max(self.adx_floor + 1.0, float(cfg.get("adx_ceiling", 35.0)))
        self.atr_pct_floor = float(cfg.get("atr_pct_floor", 0.0006))
        self.atr_pct_ceiling = max(self.atr_pct_floor + 1e-6, float(cfg.get("atr_pct_ceiling", 0.02)))
        self.ema_gap_norm_divisor = max(0.1, float(cfg.get("ema_gap_norm_divisor", 3.0)))
        self.slope_norm_divisor = max(0.05, float(cfg.get("slope_norm_divisor", 0.5)))

        self.trend_score_threshold = max(0.05, float(cfg.get("trend_score_threshold", 0.32)))
        self.trend_strength_threshold = max(0.05, float(cfg.get("trend_strength_threshold", 0.42)))
        self.meanrev_max_strength = max(0.05, float(cfg.get("meanrev_max_strength", 0.40)))
        self.meanrev_entry_distance_atr = max(0.2, float(cfg.get("meanrev_entry_distance_atr", 0.85)))
        self.meanrev_rsi_oversold = min(50.0, float(cfg.get("meanrev_rsi_oversold", 35.0)))
        self.meanrev_rsi_overbought = max(50.0, float(cfg.get("meanrev_rsi_overbought", 65.0)))

        self.trend_sl_atr_mult = max(0.2, float(cfg.get("trend_sl_atr_mult", cfg.get("sl_atr_mult", 1.2))))
        self.trend_tp_r_multiple = max(0.5, float(cfg.get("trend_tp_r_multiple", cfg.get("tp_r_multiple", 2.1))))
        self.meanrev_sl_atr_mult = max(0.2, float(cfg.get("meanrev_sl_atr_mult", 1.0)))
        self.meanrev_tp_r_multiple = max(0.5, float(cfg.get("meanrev_tp_r_multiple", 1.4)))
        self.fixed_tp_on_entry = bool(cfg.get("fixed_tp_on_entry", False))
        self.liquidity_grab_enabled = self._coerce_bool(cfg.get("liquidity_grab_enabled", False))
        self.min_volume_ratio = max(0.0, float(cfg.get("min_volume_ratio", 0.0)))

        self.break_even_rr = max(0.5, float(cfg.get("break_even_rr", 1.0)))
        self.break_even_offset_atr = max(0.0, float(cfg.get("break_even_offset_atr", 0.05)))
        self.trailing_atr_mult = max(0.3, float(cfg.get("trailing_atr_mult", 1.0)))
        self.trailing_start_rr = max(0.0, float(cfg.get("trailing_start_rr", 0.8)))
        self.partial_close_rr = max(0.0, float(cfg.get("partial_close_rr", 1.0)))
        self.time_stop_bars = max(2, int(cfg.get("time_stop_bars", 60)))
        self.regime_flip_exit_threshold = max(0.05, float(cfg.get("regime_flip_exit_threshold", 0.18)))
        self.min_hold_bars = max(1, int(cfg.get("min_hold_bars", 1)))
        self.runtime_overrides: Dict[str, Any] = {}

    def apply_runtime_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(overrides, dict):
            return {}

        applied: Dict[str, Any] = {}
        for key in self.RUNTIME_OVERRIDE_KEYS:
            if key not in overrides:
                continue
            try:
                raw = overrides[key]
                if key == "trend_score_threshold":
                    value = max(0.05, float(raw))
                    self.trend_score_threshold = value
                elif key == "trend_strength_threshold":
                    value = max(0.05, float(raw))
                    self.trend_strength_threshold = value
                elif key == "meanrev_max_strength":
                    value = max(0.05, float(raw))
                    self.meanrev_max_strength = value
                elif key == "breakout_lookback":
                    value = max(3, int(round(float(raw))))
                    self.breakout_lookback = value
                elif key == "trend_sl_atr_mult":
                    value = max(0.2, float(raw))
                    self.trend_sl_atr_mult = value
                elif key == "trend_tp_r_multiple":
                    value = max(0.5, float(raw))
                    self.trend_tp_r_multiple = value
                elif key == "trailing_atr_mult":
                    value = max(0.3, float(raw))
                    self.trailing_atr_mult = value
                elif key == "trailing_start_rr":
                    value = max(0.0, float(raw))
                    self.trailing_start_rr = value
                elif key == "partial_close_rr":
                    value = max(0.0, float(raw))
                    self.partial_close_rr = value
                elif key == "liquidity_grab_enabled":
                    value = self._coerce_bool(raw)
                    self.liquidity_grab_enabled = value
                elif key == "min_volume_ratio":
                    value = max(0.0, float(raw))
                    self.min_volume_ratio = value
                else:  # regime_flip_exit_threshold
                    value = max(0.05, float(raw))
                    self.regime_flip_exit_threshold = value
            except (TypeError, ValueError):
                continue

            self.config[key] = value
            applied[key] = value

        if applied:
            self.runtime_overrides.update(applied)
        return applied

    def _signal_bar_time(self, closed: pd.DataFrame) -> Optional[datetime]:
        if "time" not in closed.columns or closed.empty:
            return None
        return parse_bar_time(closed.iloc[-1].get("time"))

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
                return True
            if lowered in {"0", "false", "no", "n", "off", "disabled", "disable"}:
                return False
        return bool(value)

    @staticmethod
    def _volume_precision(step: float) -> int:
        text = f"{step:.12f}".rstrip("0")
        return len(text.split(".")[1]) if "." in text else 0

    @classmethod
    def _quantize_volume_floor(cls, raw_volume: float, min_volume: float, step: float) -> Optional[float]:
        if raw_volume < (min_volume - 1e-12):
            return None
        units = math.floor(((raw_volume - min_volume) / step) + 1e-12)
        quantized = min_volume + (units * step)
        return round(max(min_volume, quantized), cls._volume_precision(step))

    def _resolve_partial_close_volume(self, position: Position, st: StrategySymbolState) -> Optional[float]:
        total_volume = self._finite_float(position.volume)
        if total_volume is None or total_volume <= 0:
            return None

        position_meta = position.metadata if isinstance(position.metadata, dict) else {}
        step = self._finite_float(position_meta.get("volume_step"))
        if step is None:
            step = self._finite_float(st.metadata.get("volume_step"))
        if step is None:
            step = self._finite_float(self.config.get("volume_step"))
        if step is None or step <= 0:
            step = 0.01

        min_volume = self._finite_float(position_meta.get("min_volume"))
        if min_volume is None:
            min_volume = self._finite_float(st.metadata.get("min_volume"))
        if min_volume is None:
            min_volume = self._finite_float(self.config.get("min_volume"))
        if min_volume is None or min_volume <= 0:
            min_volume = step
        min_volume = max(step, min_volume)

        tolerance = max(1e-9, step * 0.25)
        if total_volume <= (min_volume + tolerance):
            return None

        target_volume = total_volume * 0.5
        partial_volume = self._quantize_volume_floor(target_volume, min_volume=min_volume, step=step)
        if partial_volume is None:
            return None

        max_partial = self._quantize_volume_floor(total_volume - min_volume, min_volume=min_volume, step=step)
        if max_partial is not None:
            partial_volume = min(partial_volume, max_partial)

        if partial_volume <= 0:
            return None
        if partial_volume >= (total_volume - tolerance):
            return None
        return float(partial_volume)

    def _maybe_emit_stage_a_partial_close(
        self,
        position: Position,
        st: StrategySymbolState,
        move_rr: float,
        regime: str,
        regime_score: float,
    ) -> Optional[StrategyDecision]:
        if move_rr < self.partial_close_rr:
            return None
        if bool(st.metadata.get("stage_a_hit", False)):
            return None

        partial_volume = self._resolve_partial_close_volume(position=position, st=st)
        if partial_volume is None:
            return None

        position_volume = float(position.volume)
        st.metadata["stage_a_hit"] = True
        st.metadata["stage_a_volume"] = partial_volume
        st.metadata["stage_a_rr"] = float(move_rr)

        return StrategyDecision(
            action=DecisionAction.EXIT,
            reason="STAGE_A_PARTIAL_CLOSE",
            strategy=self.name,
            confidence=0.82,
            volume=partial_volume,
            metadata={
                "is_partial": True,
                "partial_stage": "A",
                "position_volume_before": position_volume,
                "partial_volume_requested": partial_volume,
                "stage_a_rr": float(move_rr),
                "regime": regime,
                "regime_score": regime_score,
            },
        )

    def _build_entry_levels(
        self,
        side: Side,
        close_price: float,
        atr: float,
        sl_mult: float,
        tp_r_multiple: float,
        fixed_tp_on_entry: bool = False,
    ) -> Dict[str, Any]:
        stop_distance = max(atr * max(0.1, sl_mult), atr * 0.2)
        if side == Side.BUY:
            sl = close_price - stop_distance
            tp = close_price + (stop_distance * max(0.1, tp_r_multiple)) if fixed_tp_on_entry else None
        else:
            sl = close_price + stop_distance
            tp = close_price - (stop_distance * max(0.1, tp_r_multiple)) if fixed_tp_on_entry else None
        return {
            "sl": float(sl),
            "tp": float(tp) if tp is not None else None,
            "stop_distance": float(stop_distance),
        }

    def _regime_context(self, closed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        close_price = last_close(closed)
        atr = compute_atr(closed, period=self.atr_period)
        ema_fast = compute_ema(closed, period=self.fast_ema_period)
        ema_slow = compute_ema(closed, period=self.slow_ema_period)
        adx = compute_adx(closed, period=self.adx_period)
        rsi = compute_rsi(closed, period=self.rsi_period)
        if None in (close_price, atr, ema_fast, ema_slow, adx, rsi):
            return None
        if atr is None or atr <= 0:
            return None

        ema_series = closed["close"].ewm(span=self.slow_ema_period, adjust=False).mean()
        if len(ema_series) <= self.slope_lookback:
            return None

        ema_slow_now = float(ema_series.iloc[-1])
        ema_slow_prev = float(ema_series.iloc[-1 - self.slope_lookback])
        slope_per_bar = (ema_slow_now - ema_slow_prev) / float(self.slope_lookback)

        atr_pct = float(atr) / max(float(close_price), 1e-9)
        atr_norm = self._clamp((atr_pct - self.atr_pct_floor) / (self.atr_pct_ceiling - self.atr_pct_floor), 0.0, 1.0)
        adx_norm = self._clamp((float(adx) - self.adx_floor) / (self.adx_ceiling - self.adx_floor), 0.0, 1.0)
        ema_gap_atr = (float(ema_fast) - float(ema_slow)) / max(float(atr), 1e-9)
        ema_alignment = self._clamp(ema_gap_atr / self.ema_gap_norm_divisor, -1.0, 1.0)
        slope_norm = self._clamp((slope_per_bar / max(float(atr), 1e-9)) / self.slope_norm_divisor, -1.0, 1.0)

        directional_score = (ema_alignment * 0.65) + (slope_norm * 0.35)
        trend_strength = (adx_norm * 0.65) + (atr_norm * 0.35)
        regime_score = directional_score * trend_strength

        if regime_score >= self.trend_score_threshold and trend_strength >= self.trend_strength_threshold:
            regime = "TREND_UP"
        elif regime_score <= -self.trend_score_threshold and trend_strength >= self.trend_strength_threshold:
            regime = "TREND_DOWN"
        elif trend_strength <= self.meanrev_max_strength:
            regime = "RANGE"
        else:
            regime = "TRANSITION"

        return {
            "close": float(close_price),
            "atr": float(atr),
            "ema_fast": float(ema_fast),
            "ema_slow": float(ema_slow),
            "adx": float(adx),
            "rsi": float(rsi),
            "atr_pct": float(atr_pct),
            "atr_norm": float(atr_norm),
            "adx_norm": float(adx_norm),
            "ema_gap_atr": float(ema_gap_atr),
            "slope_norm": float(slope_norm),
            "directional_score": float(directional_score),
            "trend_strength": float(trend_strength),
            "regime_score": float(regime_score),
            "regime": regime,
        }

    @staticmethod
    def _entry_confidence(style: str, regime_score: float, strength: float) -> float:
        if style == "trend_follow":
            raw = 0.52 + min(0.35, abs(regime_score) * 0.6 + strength * 0.2)
        else:
            raw = 0.5 + min(0.28, (1.0 - strength) * 0.25 + abs(regime_score) * 0.25)
        return max(0.5, min(0.95, float(raw)))

    def _trend_breakout_reference(self, closed: pd.DataFrame) -> Optional[Tuple[float, float]]:
        reference = closed.tail(self.breakout_lookback + 1).iloc[:-1]
        if reference.empty:
            return None
        return float(reference["high"].max()), float(reference["low"].min())

    def _range_reference(self, closed: pd.DataFrame) -> Optional[Tuple[float, float]]:
        reference = closed.tail(self.meanrev_lookback + 1).iloc[:-1]
        if reference.empty:
            return None
        return float(reference["high"].max()), float(reference["low"].min())

    def _emit_rich_entry(
        self,
        st: StrategySymbolState,
        side: Side,
        style: str,
        reason: str,
        levels: Dict[str, Any],
        regime_ctx: Dict[str, Any],
        signal_bar_time: Optional[datetime],
        confidence_override: Optional[float] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> StrategyDecision:
        close_price = float(regime_ctx["close"])
        st.bias = side
        st.pending_order = True
        st.entry_price = close_price
        st.entry_bar_time = signal_bar_time
        st.peak_price = close_price
        st.trough_price = close_price
        st.metadata["initial_risk"] = float(levels["stop_distance"])
        st.metadata["risk_per_unit"] = float(levels["stop_distance"])
        st.metadata["entry_style"] = style
        st.metadata["entry_regime"] = str(regime_ctx["regime"])
        st.metadata["entry_regime_score"] = float(regime_ctx["regime_score"])
        st.metadata["bars_in_trade"] = 0
        st.metadata["stage_a_hit"] = False
        if isinstance(metadata_extra, dict) and metadata_extra:
            st.metadata.update(metadata_extra)

        self._transition(st, StrategyState.ENTRY_PENDING, reason)
        confidence = (
            self._entry_confidence(
                style,
                regime_score=float(regime_ctx["regime_score"]),
                strength=float(regime_ctx["trend_strength"]),
            )
            if confidence_override is None
            else max(0.0, min(1.0, float(confidence_override)))
        )
        entry_metadata: Dict[str, Any] = {
            "entry_style": style,
            "regime": str(regime_ctx["regime"]),
            "regime_score": float(regime_ctx["regime_score"]),
            "risk_per_unit": float(levels["stop_distance"]),
            "signal_close": close_price,
            "indicator_snapshot": {
                "ema_fast": float(regime_ctx["ema_fast"]),
                "ema_slow": float(regime_ctx["ema_slow"]),
                "adx": float(regime_ctx["adx"]),
                "rsi": float(regime_ctx["rsi"]),
                "atr": float(regime_ctx["atr"]),
                "atr_pct": float(regime_ctx["atr_pct"]),
                "atr_norm": float(regime_ctx["atr_norm"]),
                "adx_norm": float(regime_ctx["adx_norm"]),
                "ema_gap_atr": float(regime_ctx["ema_gap_atr"]),
                "slope_norm": float(regime_ctx["slope_norm"]),
                "directional_score": float(regime_ctx["directional_score"]),
                "trend_strength": float(regime_ctx["trend_strength"]),
            },
        }
        if isinstance(metadata_extra, dict) and metadata_extra:
            entry_metadata.update(metadata_extra)
        return self._emit_entry(
            side=side,
            reason=reason,
            confidence=confidence,
            sl=float(levels["sl"]),
            tp=float(levels["tp"]) if levels.get("tp") is not None else None,
            signal_bar_time=signal_bar_time,
            min_hold_bars=self.min_hold_bars,
            metadata=entry_metadata,
        )

    def _evaluate_impl(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: Optional[Position],
        st: StrategySymbolState,
    ) -> StrategyDecision:
        pdh: Optional[float] = None
        pdl: Optional[float] = None
        mtf_info = bars.attrs.get("mtf_info") if hasattr(bars, "attrs") else None
        if isinstance(mtf_info, dict):
            daily_reference = mtf_info.get("daily_reference")
            if isinstance(daily_reference, dict):
                pdh = self._finite_float(daily_reference.get("pdh"))
                pdl = self._finite_float(daily_reference.get("pdl"))

        clean = sanitize_ohlc(bars)
        min_len = max(
            self.slow_ema_period + self.slope_lookback + 3,
            self.adx_period * 2 + 5,
            self.atr_period + 3,
            self.rsi_period + 3,
            self.breakout_lookback + 3,
            self.meanrev_lookback + 3,
        )
        if clean is None or len(clean) < (min_len + 1):
            return self._hold("INSUFFICIENT_BARS")

        # Strategy logic strictly uses closed bars (exclude still-forming live bar).
        closed = clean.iloc[:-1].copy()
        if len(closed) < min_len:
            return self._hold("INSUFFICIENT_CLOSED_BARS")

        regime_ctx = self._regime_context(closed)
        if regime_ctx is None:
            return self._hold("INDICATORS_UNAVAILABLE")

        close_price = float(regime_ctx["close"])
        atr = float(regime_ctx["atr"])
        regime = str(regime_ctx["regime"])
        regime_score = float(regime_ctx["regime_score"])

        signal_bar_time = self._signal_bar_time(closed)

        if st.state == StrategyState.HALTED:
            return self._hold("HALTED_MANUAL_RECOVERY_REQUIRED")

        if st.state == StrategyState.COOLDOWN:
            if st.cooldown_bars_remaining > 0:
                st.cooldown_bars_remaining -= 1
                return self._hold("COOLDOWN_ACTIVE", {"remaining": st.cooldown_bars_remaining})
            self._transition(st, StrategyState.IDLE, "COOLDOWN_COMPLETE")

        if st.state == StrategyState.ENTRY_PENDING:
            if position is not None:
                self._transition(st, StrategyState.IN_POSITION, "POSITION_DETECTED_WHILE_PENDING")
            else:
                return self._hold("ENTRY_PENDING_WAIT_FILL")

        if st.state == StrategyState.IDLE:
            if self.min_volume_ratio > 0.0 and "tick_volume" in closed.columns:
                volume_series = pd.to_numeric(closed["tick_volume"], errors="coerce")
                current_volume = self._finite_float(volume_series.iloc[-1])
                volume_ma_20 = self._finite_float(volume_series.rolling(window=20, min_periods=20).mean().iloc[-1])
                if (
                    current_volume is not None
                    and volume_ma_20 is not None
                    and volume_ma_20 > 0.0
                    and current_volume < (volume_ma_20 * self.min_volume_ratio)
                ):
                    return self._hold("LOW_VOLUME_FILTER")

            if self.liquidity_grab_enabled and (pdh is not None or pdl is not None):
                last_bar = closed.iloc[-1]
                buy_sweep, sell_sweep = detect_institutional_sweep(last_bar, pdh, pdl)

                if buy_sweep:
                    levels = self._build_entry_levels(
                        side=Side.BUY,
                        close_price=close_price,
                        atr=atr,
                        sl_mult=self.meanrev_sl_atr_mult,
                        tp_r_multiple=self.trend_tp_r_multiple,
                        fixed_tp_on_entry=self.fixed_tp_on_entry,
                    )
                    return self._emit_rich_entry(
                        st=st,
                        side=Side.BUY,
                        style="liquidity_grab",
                        reason="PDL_SWEEP_REJECTION",
                        levels=levels,
                        regime_ctx=regime_ctx,
                        signal_bar_time=signal_bar_time,
                        confidence_override=0.9,
                        metadata_extra={
                            "liquidity_levels": {"pdh": pdh, "pdl": pdl},
                            "liquidity_sweep": {"buy_sweep": True, "sell_sweep": False},
                        },
                    )

                if sell_sweep:
                    levels = self._build_entry_levels(
                        side=Side.SELL,
                        close_price=close_price,
                        atr=atr,
                        sl_mult=self.meanrev_sl_atr_mult,
                        tp_r_multiple=self.trend_tp_r_multiple,
                        fixed_tp_on_entry=self.fixed_tp_on_entry,
                    )
                    return self._emit_rich_entry(
                        st=st,
                        side=Side.SELL,
                        style="liquidity_grab",
                        reason="PDH_SWEEP_REJECTION",
                        levels=levels,
                        regime_ctx=regime_ctx,
                        signal_bar_time=signal_bar_time,
                        confidence_override=0.9,
                        metadata_extra={
                            "liquidity_levels": {"pdh": pdh, "pdl": pdl},
                            "liquidity_sweep": {"buy_sweep": False, "sell_sweep": True},
                        },
                    )

            breakout = self._trend_breakout_reference(closed)
            if breakout is None:
                return self._hold("NO_BREAKOUT_REFERENCE")
            breakout_high, breakout_low = breakout

            if regime == "TREND_UP" and close_price > breakout_high:
                levels = self._build_entry_levels(
                    side=Side.BUY,
                    close_price=close_price,
                    atr=atr,
                    sl_mult=self.trend_sl_atr_mult,
                    tp_r_multiple=self.trend_tp_r_multiple,
                    fixed_tp_on_entry=self.fixed_tp_on_entry,
                )
                return self._emit_rich_entry(
                    st=st,
                    side=Side.BUY,
                    style="trend_follow",
                    reason="TREND_LONG_BREAKOUT",
                    levels=levels,
                    regime_ctx=regime_ctx,
                    signal_bar_time=signal_bar_time,
                )

            if regime == "TREND_DOWN" and close_price < breakout_low:
                levels = self._build_entry_levels(
                    side=Side.SELL,
                    close_price=close_price,
                    atr=atr,
                    sl_mult=self.trend_sl_atr_mult,
                    tp_r_multiple=self.trend_tp_r_multiple,
                    fixed_tp_on_entry=self.fixed_tp_on_entry,
                )
                return self._emit_rich_entry(
                    st=st,
                    side=Side.SELL,
                    style="trend_follow",
                    reason="TREND_SHORT_BREAKDOWN",
                    levels=levels,
                    regime_ctx=regime_ctx,
                    signal_bar_time=signal_bar_time,
                )

            if regime == "RANGE":
                range_ref = self._range_reference(closed)
                if range_ref is None:
                    return self._hold("NO_RANGE_REFERENCE")
                range_high, range_low = range_ref
                rsi = float(regime_ctx["rsi"])
                distance_atr = (close_price - float(regime_ctx["ema_slow"])) / max(atr, 1e-9)

                long_edge = close_price <= (range_low + (atr * 0.15))
                short_edge = close_price >= (range_high - (atr * 0.15))

                if distance_atr <= -self.meanrev_entry_distance_atr and rsi <= self.meanrev_rsi_oversold and long_edge:
                    levels = self._build_entry_levels(
                        side=Side.BUY,
                        close_price=close_price,
                        atr=atr,
                        sl_mult=self.meanrev_sl_atr_mult,
                        tp_r_multiple=self.meanrev_tp_r_multiple,
                        fixed_tp_on_entry=self.fixed_tp_on_entry,
                    )
                    return self._emit_rich_entry(
                        st=st,
                        side=Side.BUY,
                        style="mean_reversion",
                        reason="RANGE_REVERT_LONG",
                        levels=levels,
                        regime_ctx=regime_ctx,
                        signal_bar_time=signal_bar_time,
                    )

                if distance_atr >= self.meanrev_entry_distance_atr and rsi >= self.meanrev_rsi_overbought and short_edge:
                    levels = self._build_entry_levels(
                        side=Side.SELL,
                        close_price=close_price,
                        atr=atr,
                        sl_mult=self.meanrev_sl_atr_mult,
                        tp_r_multiple=self.meanrev_tp_r_multiple,
                        fixed_tp_on_entry=self.fixed_tp_on_entry,
                    )
                    return self._emit_rich_entry(
                        st=st,
                        side=Side.SELL,
                        style="mean_reversion",
                        reason="RANGE_REVERT_SHORT",
                        levels=levels,
                        regime_ctx=regime_ctx,
                        signal_bar_time=signal_bar_time,
                    )

            return self._hold(
                "NO_ENTRY_SIGNAL",
                {
                    "regime": regime,
                    "regime_score": regime_score,
                    "trend_strength": float(regime_ctx["trend_strength"]),
                    "indicator_snapshot": {
                        "ema_fast": float(regime_ctx["ema_fast"]),
                        "ema_slow": float(regime_ctx["ema_slow"]),
                        "adx": float(regime_ctx["adx"]),
                        "rsi": float(regime_ctx["rsi"]),
                        "atr": float(regime_ctx["atr"]),
                    },
                },
            )

        if st.state == StrategyState.IN_POSITION:
            if position is None:
                st.cooldown_bars_remaining = self.min_cooldown_bars
                self._transition(st, StrategyState.COOLDOWN, "POSITION_NOT_FOUND")
                return self._hold("WAITING_RECONCILIATION")

            current_bar_key = signal_bar_time.isoformat() if signal_bar_time is not None else ""
            last_manage_bar_key = str(st.metadata.get("last_manage_bar_time", "") or "")
            bar_advanced = bool(current_bar_key) and current_bar_key != last_manage_bar_key

            bars_in_trade = int(st.metadata.get("bars_in_trade", 0))
            if bar_advanced:
                bars_in_trade += 1
                st.metadata["bars_in_trade"] = bars_in_trade
                st.metadata["last_manage_bar_time"] = current_bar_key

            min_hold = max(self.min_hold_bars, int(st.metadata.get("min_hold_bars", self.min_hold_bars)))
            desired_sl = position.sl
            desired_tp = position.tp

            entry_price = st.entry_price if st.entry_price is not None else float(position.price_open)
            risk_per_unit = float(st.metadata.get("initial_risk", 0.0) or 0.0)
            if risk_per_unit <= 0 and position.sl is not None:
                risk_per_unit = abs(entry_price - float(position.sl))
            if risk_per_unit <= 0:
                fallback_mult = self.trend_sl_atr_mult if st.metadata.get("entry_style") == "trend_follow" else self.meanrev_sl_atr_mult
                risk_per_unit = max(atr * fallback_mult, atr * 0.2)

            st.metadata["risk_per_unit"] = float(risk_per_unit)
            st.metadata["last_regime"] = regime
            st.metadata["last_regime_score"] = regime_score

            if position.side == Side.BUY:
                st.peak_price = max(st.peak_price or close_price, close_price)
                move_rr = (close_price - entry_price) / max(risk_per_unit, 1e-9)

                stage_a_decision = self._maybe_emit_stage_a_partial_close(
                    position=position,
                    st=st,
                    move_rr=move_rr,
                    regime=regime,
                    regime_score=regime_score,
                )
                if stage_a_decision is not None:
                    return stage_a_decision

                if move_rr >= self.break_even_rr:
                    be_sl = entry_price + (atr * self.break_even_offset_atr)
                    desired_sl = max(float(desired_sl) if desired_sl is not None else be_sl, be_sl)

                trail_candidate = None
                if move_rr >= self.trailing_start_rr:
                    trail_candidate = (st.peak_price or close_price) - (atr * self.trailing_atr_mult)
                    desired_sl = max(float(desired_sl) if desired_sl is not None else trail_candidate, trail_candidate)

                trail_exit = bool(
                    trail_candidate is not None
                    and bar_advanced
                    and bars_in_trade >= min_hold
                    and close_price <= trail_candidate
                )
                time_exit = bool(bar_advanced and bars_in_trade >= self.time_stop_bars)
                regime_flip = bool(bars_in_trade >= min_hold and regime_score <= -self.regime_flip_exit_threshold)

                if trail_exit:
                    self._transition(st, StrategyState.EXIT_READY, "BUY_TRAIL_BREACH")
                elif time_exit:
                    self._transition(st, StrategyState.EXIT_READY, "BUY_TIME_STOP")
                elif regime_flip:
                    self._transition(st, StrategyState.EXIT_READY, "BUY_REGIME_FLIP_EXIT")
                else:
                    return self._hold(
                        "BUY_POSITION_MANAGE",
                        {
                            "bars_in_trade": bars_in_trade,
                            "regime": regime,
                            "regime_score": regime_score,
                            "move_rr": move_rr,
                            "peak": st.peak_price,
                            "trail_candidate": trail_candidate,
                            "risk_per_unit": risk_per_unit,
                        },
                        sl=desired_sl,
                        tp=desired_tp,
                    )
            else:
                st.trough_price = min(st.trough_price or close_price, close_price)
                move_rr = (entry_price - close_price) / max(risk_per_unit, 1e-9)

                stage_a_decision = self._maybe_emit_stage_a_partial_close(
                    position=position,
                    st=st,
                    move_rr=move_rr,
                    regime=regime,
                    regime_score=regime_score,
                )
                if stage_a_decision is not None:
                    return stage_a_decision

                if move_rr >= self.break_even_rr:
                    be_sl = entry_price - (atr * self.break_even_offset_atr)
                    desired_sl = min(float(desired_sl) if desired_sl is not None else be_sl, be_sl)

                trail_candidate = None
                if move_rr >= self.trailing_start_rr:
                    trail_candidate = (st.trough_price or close_price) + (atr * self.trailing_atr_mult)
                    desired_sl = min(float(desired_sl) if desired_sl is not None else trail_candidate, trail_candidate)

                trail_exit = bool(
                    trail_candidate is not None
                    and bar_advanced
                    and bars_in_trade >= min_hold
                    and close_price >= trail_candidate
                )
                time_exit = bool(bar_advanced and bars_in_trade >= self.time_stop_bars)
                regime_flip = bool(bars_in_trade >= min_hold and regime_score >= self.regime_flip_exit_threshold)

                if trail_exit:
                    self._transition(st, StrategyState.EXIT_READY, "SELL_TRAIL_BREACH")
                elif time_exit:
                    self._transition(st, StrategyState.EXIT_READY, "SELL_TIME_STOP")
                elif regime_flip:
                    self._transition(st, StrategyState.EXIT_READY, "SELL_REGIME_FLIP_EXIT")
                else:
                    return self._hold(
                        "SELL_POSITION_MANAGE",
                        {
                            "bars_in_trade": bars_in_trade,
                            "regime": regime,
                            "regime_score": regime_score,
                            "move_rr": move_rr,
                            "trough": st.trough_price,
                            "trail_candidate": trail_candidate,
                            "risk_per_unit": risk_per_unit,
                        },
                        sl=desired_sl,
                        tp=desired_tp,
                    )

        if st.state == StrategyState.EXIT_READY:
            st.pending_order = False
            return self._emit_exit(
                reason=f"TREND_REGIME_EXIT:{st.last_reason}",
                confidence=0.78,
                metadata={
                    "regime": regime,
                    "regime_score": regime_score,
                },
            )

        return self._hold("NO_ACTION")
