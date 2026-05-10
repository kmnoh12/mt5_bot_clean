from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd

from core.models import (
    DecisionAction,
    MarketTick,
    Position,
    Side,
    StrategyDecision,
    StrategyState,
    StrategySymbolState,
)
from strategies.base import LOGGER, BaseStateMachineStrategy
from utils.indicators import compute_atr, sanitize_ohlc


@dataclass
class _SweepSetup:
    side: Side
    level_name: str
    level: float
    sweep_time_ms: int
    deadline_ms: int
    reclaim_window_sec: int
    reclaim_extension_sec: int
    extended: bool
    sweep_buffer: float
    reclaim_buffer: float
    stop_buffer: float
    extreme: float
    extreme_time_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side.value,
            "level_name": self.level_name,
            "level": float(self.level),
            "sweep_time_ms": int(self.sweep_time_ms),
            "deadline_ms": int(self.deadline_ms),
            "reclaim_window_sec": int(self.reclaim_window_sec),
            "reclaim_extension_sec": int(self.reclaim_extension_sec),
            "extended": bool(self.extended),
            "sweep_buffer": float(self.sweep_buffer),
            "reclaim_buffer": float(self.reclaim_buffer),
            "stop_buffer": float(self.stop_buffer),
            "extreme": float(self.extreme),
            "extreme_time_ms": int(self.extreme_time_ms),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["_SweepSetup"]:
        if not isinstance(raw, dict):
            return None
        side_text = str(raw.get("side", "")).strip().upper()
        if side_text not in Side.__members__:
            return None
        side = Side[side_text]
        try:
            level = float(raw.get("level"))
            extreme = float(raw.get("extreme"))
        except Exception:
            return None
        if not math.isfinite(level) or not math.isfinite(extreme):
            return None
        level_name = str(raw.get("level_name", "") or "").strip().upper()
        if not level_name:
            level_name = "PDH" if side == Side.SELL else "PDL"
        try:
            sweep_time_ms = int(raw.get("sweep_time_ms"))
            deadline_ms = int(raw.get("deadline_ms"))
        except Exception:
            return None
        try:
            extreme_time_ms = int(raw.get("extreme_time_ms", sweep_time_ms) or sweep_time_ms)
        except Exception:
            extreme_time_ms = int(sweep_time_ms)
        reclaim_window_sec = max(1, int(raw.get("reclaim_window_sec", 20) or 20))
        reclaim_extension_sec = max(0, int(raw.get("reclaim_extension_sec", 0) or 0))
        extended = bool(raw.get("extended", False))
        try:
            sweep_buffer = float(raw.get("sweep_buffer", 0.0) or 0.0)
            reclaim_buffer = float(raw.get("reclaim_buffer", 0.0) or 0.0)
            stop_buffer = float(raw.get("stop_buffer", 0.0) or 0.0)
        except Exception:
            return None
        return cls(
            side=side,
            level_name=level_name,
            level=float(level),
            sweep_time_ms=sweep_time_ms,
            deadline_ms=deadline_ms,
            reclaim_window_sec=reclaim_window_sec,
            reclaim_extension_sec=reclaim_extension_sec,
            extended=extended,
            sweep_buffer=float(sweep_buffer),
            reclaim_buffer=float(reclaim_buffer),
            stop_buffer=float(stop_buffer),
            extreme=float(extreme),
            extreme_time_ms=extreme_time_ms,
        )


class LiquiditySweepReversalTickStrategy(BaseStateMachineStrategy):
    """
    Legacy LSR (tick-driven):
    - v0.16 Execution Alpha: Dynamic Spread Guard
    - Daily PDH/PDL sweep
    - Reclaim within 20s (+ optional extension to 45s)
    - Displacement check (tick-based)
    - Immediate entry, TP 3.0R, stage-A partial 1.5R, BE at 1R (or after stage-A)
    """

    tick_driven = True

    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="liquidity_sweep_reversal_tick", config=config, snapshot=snapshot)
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))

        self._base_params = self._coerce_param_bundle(cfg)
        self._symbol_params = self._coerce_symbol_params(cfg.get("symbol_params"))
        self._active_param_symbol = "__BASE__"
        self._activate_param_bundle(self._base_params)

        self._tick_buffers: Dict[str, Deque[Tuple[int, float, float]]] = {}
        self._last_ingested_ms: Dict[str, int] = {}

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return float(out) if math.isfinite(out) else float(default)

    @classmethod
    def _finite_float(cls, value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out

    def _coerce_param_bundle(self, raw_cfg: Dict[str, Any], seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        defaults = dict(seed or {})

        atr_period = max(3, self._to_int(cfg.get("atr_period", defaults.get("atr_period", 14)), 14))
        sweep_buffer_atr = max(
            0.0,
            self._to_float(cfg.get("sweep_buffer_atr", defaults.get("sweep_buffer_atr", 0.012)), 0.012),
        )
        reclaim_buffer_atr = max(
            0.0,
            self._to_float(cfg.get("reclaim_buffer_atr", defaults.get("reclaim_buffer_atr", 0.007)), 0.007),
        )
        stop_buffer_atr = max(
            0.0,
            self._to_float(cfg.get("stop_buffer_atr", defaults.get("stop_buffer_atr", 0.12)), 0.12),
        )
        reclaim_window_sec = max(
            1,
            self._to_int(cfg.get("reclaim_window_sec", defaults.get("reclaim_window_sec", 20)), 20),
        )
        reclaim_extension_sec = max(
            0,
            self._to_int(cfg.get("reclaim_extension_sec", defaults.get("reclaim_extension_sec", 25)), 25),
        )
        displacement_mult = max(
            1.0,
            self._to_float(cfg.get("displacement_mult", defaults.get("displacement_mult", 1.35)), 1.35),
        )
        displacement_window_sec = max(
            0.2,
            self._to_float(cfg.get("displacement_window_sec", defaults.get("displacement_window_sec", 1.0)), 1.0),
        )
        displacement_lookback_sec = max(
            5.0,
            self._to_float(cfg.get("displacement_lookback_sec", defaults.get("displacement_lookback_sec", 60.0)), 60.0),
        )

        tp_r1 = max(0.1, self._to_float(cfg.get("tp_R1", defaults.get("tp_R1", 1.5)), 1.5))
        tp_r2 = max(tp_r1 + 0.1, self._to_float(cfg.get("tp_R2", defaults.get("tp_R2", 3.0)), 3.0))
        be_at_r = max(0.1, self._to_float(cfg.get("be_at_R", defaults.get("be_at_R", 1.0)), 1.0))
        stage_a_fraction = min(
            1.0,
            max(0.0, self._to_float(cfg.get("stage_a_fraction", defaults.get("stage_a_fraction", 0.5)), 0.5)),
        )

        trail_enabled = bool(cfg.get("trail_enabled", defaults.get("trail_enabled", False)))
        trail_start_r = max(0.1, self._to_float(cfg.get("trail_start_R", defaults.get("trail_start_R", 1.5)), 1.5))
        trail_atr_mult = max(0.0, self._to_float(cfg.get("trail_atr_mult", defaults.get("trail_atr_mult", 0.8)), 0.8))
        trail_update_min_seconds = max(
            0.2,
            self._to_float(cfg.get("trail_update_min_seconds", defaults.get("trail_update_min_seconds", 1.0)), 1.0),
        )
        trail_min_step_atr = max(0.0, self._to_float(cfg.get("trail_min_step_atr", defaults.get("trail_min_step_atr", 0.05)), 0.05))

        tick_buffer_seconds = max(
            30.0,
            self._to_float(
                cfg.get("tick_buffer_seconds", defaults.get("tick_buffer_seconds", max(displacement_lookback_sec + 5.0, 120.0))),
                120.0,
            ),
        )

        min_hold_bars = max(1, self._to_int(cfg.get("min_hold_bars", defaults.get("min_hold_bars", 1)), 1))
        min_cooldown_bars = max(1, self._to_int(cfg.get("min_cooldown_bars", defaults.get("min_cooldown_bars", 5)), 5))
        max_hold_seconds = max(1, self._to_int(cfg.get("max_hold_seconds", defaults.get("max_hold_seconds", 14400)), 14400))

        return {
            "atr_period": int(atr_period),
            "sweep_buffer_atr": float(sweep_buffer_atr),
            "reclaim_buffer_atr": float(reclaim_buffer_atr),
            "stop_buffer_atr": float(stop_buffer_atr),
            "reclaim_window_sec": int(reclaim_window_sec),
            "reclaim_extension_sec": int(reclaim_extension_sec),
            "displacement_mult": float(displacement_mult),
            "displacement_window_sec": float(displacement_window_sec),
            "displacement_lookback_sec": float(displacement_lookback_sec),
            "tp_R1": float(tp_r1),
            "tp_R2": float(tp_r2),
            "be_at_R": float(be_at_r),
            "stage_a_fraction": float(stage_a_fraction),
            "trail_enabled": bool(trail_enabled),
            "trail_start_R": float(trail_start_r),
            "trail_atr_mult": float(trail_atr_mult),
            "trail_update_min_seconds": float(trail_update_min_seconds),
            "trail_min_step_atr": float(trail_min_step_atr),
            "tick_buffer_seconds": float(tick_buffer_seconds),
            "min_hold_bars": int(min_hold_bars),
            "min_cooldown_bars": int(min_cooldown_bars),
            "max_hold_seconds": int(max_hold_seconds),
        }

    @staticmethod
    def _coerce_symbol_params(raw: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for symbol, params in raw.items():
            key = str(symbol or "").strip().upper()
            if not key or not isinstance(params, dict):
                continue
            out[key] = dict(params)
        return out

    def _activate_param_bundle(self, bundle: Dict[str, Any]) -> None:
        self.atr_period = int(bundle["atr_period"])
        self.sweep_buffer_atr = float(bundle["sweep_buffer_atr"])
        self.reclaim_buffer_atr = float(bundle["reclaim_buffer_atr"])
        self.stop_buffer_atr = float(bundle["stop_buffer_atr"])
        self.reclaim_window_sec = int(bundle["reclaim_window_sec"])
        self.reclaim_extension_sec = int(bundle["reclaim_extension_sec"])
        self.displacement_mult = float(bundle["displacement_mult"])
        self.displacement_window_sec = float(bundle["displacement_window_sec"])
        self.displacement_lookback_sec = float(bundle["displacement_lookback_sec"])
        self.tp_r1 = float(bundle["tp_R1"])
        self.tp_r2 = float(bundle["tp_R2"])
        self.be_at_r = float(bundle["be_at_R"])
        self.stage_a_fraction = float(bundle["stage_a_fraction"])
        self.trail_enabled = bool(bundle["trail_enabled"])
        self.trail_start_r = float(bundle["trail_start_R"])
        self.trail_atr_mult = float(bundle["trail_atr_mult"])
        self.trail_update_min_seconds = float(bundle["trail_update_min_seconds"])
        self.trail_min_step_atr = float(bundle["trail_min_step_atr"])
        self.tick_buffer_seconds = float(bundle["tick_buffer_seconds"])
        self.min_hold_bars = int(bundle["min_hold_bars"])
        self.min_cooldown_bars = int(bundle["min_cooldown_bars"])
        self.max_hold_seconds = int(bundle["max_hold_seconds"])

    def _activate_symbol_params(self, symbol: str) -> None:
        key = str(symbol or "").strip().upper()
        if not key:
            return
        if key == self._active_param_symbol:
            return
        merged = dict(self._base_params)
        overrides = self._symbol_params.get(key)
        if isinstance(overrides, dict) and overrides:
            merged.update(self._coerce_param_bundle(overrides, seed=merged))
        self._active_param_symbol = key
        self._activate_param_bundle(merged)

    def ingest_ticks(self, symbol: str, ticks: List[MarketTick]) -> None:
        key = str(symbol or "").strip().upper()
        if not key:
            return
        buf = self._tick_buffers.get(key)
        if buf is None:
            buf = deque()
            self._tick_buffers[key] = buf
        last_ms = int(self._last_ingested_ms.get(key, 0) or 0)

        for tick in ticks or []:
            ts_ms = int(tick.time_msc) if tick.time_msc is not None else int(tick.time_utc.timestamp() * 1000.0)
            if ts_ms <= last_ms:
                continue
            mid = tick.mid()
            if mid is None or not math.isfinite(float(mid)):
                continue
            bid = self._finite_float(getattr(tick, "bid", None))
            ask = self._finite_float(getattr(tick, "ask", None))
            spread = math.nan
            if bid is not None and ask is not None:
                raw_spread = float(ask) - float(bid)
                if math.isfinite(raw_spread) and raw_spread >= 0.0:
                    spread = float(raw_spread)
            buf.append((ts_ms, float(mid), float(spread)))
            last_ms = ts_ms

        if last_ms > 0:
            self._last_ingested_ms[key] = last_ms

        # Prune old ticks.
        now_ms = last_ms if last_ms > 0 else int(datetime.now(timezone.utc).timestamp() * 1000.0)
        cutoff = now_ms - int(max(5.0, float(self.tick_buffer_seconds)) * 1000.0)
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def apply_order_result(self, symbol: str, decision: StrategyDecision, result: Optional[Any]) -> None:
        st = self._symbol_state(symbol)
        full_exit_filled = bool(
            decision.action == DecisionAction.EXIT
            and result is not None
            and bool(getattr(result, "ok", False))
            and not self._is_partial_exit_decision(decision)
        )
        super().apply_order_result(symbol=symbol, decision=decision, result=result)

        if decision.action in {DecisionAction.BUY, DecisionAction.SELL} and result is not None and bool(getattr(result, "ok", False)):
            filled = self._finite_float(getattr(result, "filled_price", None))
            sl = self._finite_float(decision.sl)
            if filled is not None and sl is not None:
                st.metadata["initial_sl"] = float(sl)
                st.metadata["initial_risk_per_unit"] = abs(float(filled) - float(sl))
            st.metadata.pop("lsr_tick_setup", None)
            st.metadata.pop("stage_a_hit", None)
            st.metadata.pop("be_moved", None)
            st.metadata.pop("last_trail_update_ms", None)

        if full_exit_filled:
            for key in (
                "initial_sl",
                "initial_risk_per_unit",
                "stage_a_hit",
                "be_moved",
                "last_trail_update_ms",
                "lsr_tick_setup",
            ):
                st.metadata.pop(key, None)

    def _latest_tick(self, symbol: str) -> Optional[Tuple[int, float, float]]:
        key = str(symbol or "").strip().upper()
        buf = self._tick_buffers.get(key)
        if not buf:
            return None
        return buf[-1]

    def _latest_spread(self, symbol: str, now_ms: int, max_age_ms: int) -> Optional[Tuple[float, int]]:
        key = str(symbol or "").strip().upper()
        buf = self._tick_buffers.get(key)
        if not buf:
            return None
        age_limit = max(0, int(max_age_ms))
        for ts_ms, _, spread in reversed(buf):
            age_ms = int(now_ms) - int(ts_ms)
            if age_ms < 0:
                continue
            if age_ms > age_limit:
                break
            spread_f = self._finite_float(spread)
            if spread_f is None or spread_f < 0.0:
                continue
            return float(spread_f), int(age_ms)
        return None

    def _daily_levels(self, bars: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        attrs = getattr(bars, "attrs", {}) if bars is not None else {}
        if not isinstance(attrs, dict):
            return None, None
        mtf_info = attrs.get("mtf_info")
        if not isinstance(mtf_info, dict):
            return None, None
        daily_reference = mtf_info.get("daily_reference")
        if not isinstance(daily_reference, dict):
            return None, None
        pdh = self._finite_float(daily_reference.get("pdh"))
        pdl = self._finite_float(daily_reference.get("pdl"))
        return pdh, pdl

    def _displacement_ratio(self, symbol: str, now_ms: int) -> Optional[float]:
        key = str(symbol or "").strip().upper()
        buf = self._tick_buffers.get(key)
        if not buf or len(buf) < 5:
            return None
        data = list(buf)
        times = [t for t, _, _ in data]
        prices = [p for _, p, _ in data]
        if not times:
            return None

        window_ms = int(max(200.0, float(self.displacement_window_sec) * 1000.0))
        lookback_ms = int(max(1000.0, float(self.displacement_lookback_sec) * 1000.0))
        if now_ms - window_ms <= times[0]:
            return None

        def price_at(ts_target: int) -> Optional[float]:
            idx = bisect_right(times, ts_target) - 1
            if idx < 0:
                return None
            return prices[idx]

        p_now = price_at(now_ms)
        p_base = price_at(now_ms - window_ms)
        if p_now is None or p_base is None:
            return None
        current_disp = abs(float(p_now) - float(p_base))
        if current_disp <= 0:
            return 0.0

        disps: List[float] = []
        step_ms = 1000
        end_ms = now_ms - window_ms
        start_ms = max(times[0] + window_ms, now_ms - lookback_ms)
        t = end_ms
        while t >= start_ms:
            p1 = price_at(t)
            p0 = price_at(t - window_ms)
            if p1 is not None and p0 is not None:
                disp = abs(float(p1) - float(p0))
                if math.isfinite(disp):
                    disps.append(float(disp))
            t -= step_ms
        if len(disps) < 10:
            return None
        med = statistics.median(disps)
        if med <= 0 or not math.isfinite(med):
            return None
        return float(current_disp / med)

    def _adaptive_tp_r2(self, closed: pd.DataFrame, entry_atr: float) -> Dict[str, Any]:
        fallback = {
            "tp_r2": 3.5,
            "profile": "default",
            "entry_atr": float(entry_atr),
            "recent_atr_mean": None,
            "atr_vs_recent_ratio": None,
            "sample_count": 0,
            "low_vol_threshold_ratio": 0.9,
            "high_vol_threshold_ratio": 1.2,
        }
        if closed is None or closed.empty:
            return fallback
        atr_p = max(1, int(self.atr_period))
        high = pd.to_numeric(closed.get("high"), errors="coerce")
        low = pd.to_numeric(closed.get("low"), errors="coerce")
        close = pd.to_numeric(closed.get("close"), errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = tr.rolling(atr_p).mean().dropna()
        recent_window = 20
        if len(atr_series) < recent_window:
            return fallback
        recent = atr_series.tail(recent_window)
        recent_mean_raw = recent.mean()
        try:
            recent_mean = float(recent_mean_raw)
        except Exception:
            return fallback
        if not math.isfinite(recent_mean) or recent_mean <= 0:
            return fallback
        ratio = float(entry_atr) / recent_mean
        if not math.isfinite(ratio):
            return fallback
        if ratio < 0.9:
            tp_r2 = 2.2
            profile = "low_atr"
        elif ratio > 1.2:
            tp_r2 = 4.5
            profile = "high_atr"
        else:
            tp_r2 = 3.5
            profile = "normal_atr"
        return {
            "tp_r2": float(tp_r2),
            "profile": profile,
            "entry_atr": float(entry_atr),
            "recent_atr_mean": float(recent_mean),
            "atr_vs_recent_ratio": float(ratio),
            "sample_count": int(len(recent)),
            "low_vol_threshold_ratio": 0.9,
            "high_vol_threshold_ratio": 1.2,
        }

    def _segment_velocity(self, symbol: str, start_ms: int, end_ms: int) -> Optional[Dict[str, Any]]:
        key = str(symbol or "").strip().upper()
        buf = self._tick_buffers.get(key)
        if not buf:
            return None
        start_i = int(start_ms)
        end_i = int(end_ms)
        if end_i <= start_i:
            return None
        segment: List[Tuple[int, float]] = [
            (int(ts_ms), float(price))
            for ts_ms, price, _ in buf
            if int(ts_ms) >= start_i and int(ts_ms) <= end_i
        ]
        if len(segment) < 2:
            return None
        tick_count = int(len(segment) - 1)
        if tick_count <= 0:
            return None
        start_time_ms = int(segment[0][0])
        end_time_ms = int(segment[-1][0])
        start_price = float(segment[0][1])
        end_price = float(segment[-1][1])
        price_change = float(end_price - start_price)
        velocity = float(abs(price_change) / float(tick_count))
        if not (math.isfinite(price_change) and math.isfinite(velocity)):
            return None
        return {
            "tick_count": int(tick_count),
            "start_time_ms": int(start_time_ms),
            "end_time_ms": int(end_time_ms),
            "start_price": float(start_price),
            "end_price": float(end_price),
            "price_change": float(price_change),
            "price_change_abs": float(abs(price_change)),
            "velocity": float(velocity),
        }

    def _cooldown_step(self, st: StrategySymbolState) -> Optional[StrategyDecision]:
        if st.cooldown_bars_remaining <= 0:
            self._transition(st, StrategyState.IDLE, "COOLDOWN_COMPLETE")
            return None
        bar_time = st.last_closed_bar_time
        if bar_time is None:
            return self._hold("COOLDOWN_ACTIVE", {"remaining": int(st.cooldown_bars_remaining)})
        bar_key = bar_time.astimezone(timezone.utc).isoformat()
        last_key = str(st.metadata.get("cooldown_last_bar_time", "") or "")
        if bar_key != last_key:
            st.metadata["cooldown_last_bar_time"] = bar_key
            st.cooldown_bars_remaining = max(0, int(st.cooldown_bars_remaining) - 1)
        if st.cooldown_bars_remaining > 0:
            return self._hold("COOLDOWN_ACTIVE", {"remaining": int(st.cooldown_bars_remaining)})
        self._transition(st, StrategyState.IDLE, "COOLDOWN_COMPLETE")
        return None

    def _reset_setup(self, st: StrategySymbolState, reason: str) -> StrategyDecision:
        st.metadata.pop("lsr_tick_setup", None)
        self._transition(st, StrategyState.IDLE, reason)
        return self._hold(reason)

    def _reset_setup_with_metadata(
        self,
        st: StrategySymbolState,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StrategyDecision:
        st.metadata.pop("lsr_tick_setup", None)
        self._transition(st, StrategyState.IDLE, reason)
        return self._hold(reason, metadata=metadata or {})

    def _emit_stage_a_partial(self, position: Position, st: StrategySymbolState, move_rr: float) -> Optional[StrategyDecision]:
        if move_rr < self.tp_r1:
            return None
        if bool(st.metadata.get("stage_a_hit", False)):
            return None
        if self.stage_a_fraction <= 0:
            return None
        requested = float(position.volume) * float(self.stage_a_fraction)
        if requested <= 0:
            return None
        st.metadata["stage_a_hit"] = True
        st.metadata["stage_a_rr"] = float(move_rr)
        return StrategyDecision(
            action=DecisionAction.EXIT,
            reason="LSR_TICK_STAGE_A_PARTIAL_CLOSE",
            strategy=self.name,
            confidence=0.82,
            volume=requested,
            metadata={
                "is_partial": True,
                "partial_stage": "A",
                "position_volume_before": float(position.volume),
                "partial_volume_requested": float(requested),
                "stage_a_rr": float(move_rr),
            },
        )

    def _evaluate_impl(self, symbol: str, bars: pd.DataFrame, position: Optional[Position], st: StrategySymbolState) -> StrategyDecision:
        self._activate_symbol_params(symbol)
        if not self.enabled:
            return self._hold("DISABLED")
        if st.state == StrategyState.HALTED:
            return self._hold("HALTED_MANUAL_RECOVERY_REQUIRED")

        clean = sanitize_ohlc(bars)
        if clean is None or len(clean) < max(self.atr_period + 2, 10):
            return self._hold("INSUFFICIENT_BARS")
        closed = clean.iloc[:-1].copy() if len(clean) > 1 else clean.copy()
        atr = compute_atr(closed, period=self.atr_period)
        if atr is None or atr <= 0:
            return self._hold("INDICATORS_UNAVAILABLE")
        tp_profile = self._adaptive_tp_r2(closed=closed, entry_atr=float(atr))
        tp_r2_dynamic = float(tp_profile["tp_r2"])

        pdh, pdl = self._daily_levels(bars)
        if pdh is None or pdl is None:
            return self._hold("DAILY_REFERENCE_MISSING")

        latest = self._latest_tick(symbol)
        if latest is None:
            if st.state == StrategyState.COOLDOWN:
                stepped = self._cooldown_step(st)
                return stepped or self._hold("COOLDOWN_ACTIVE", {"remaining": int(st.cooldown_bars_remaining)})
            return self._hold("NO_TICKS")

        now_ms, mid, _ = latest
        now_dt = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc)

        spread_max_age_ms = int(max(2000.0, float(self.displacement_window_sec) * 5000.0))
        spread_snapshot = self._latest_spread(symbol=symbol, now_ms=int(now_ms), max_age_ms=spread_max_age_ms)
        if spread_snapshot is None:
            return self._hold("SPREAD_UNAVAILABLE_SKIP", {"spread_max_age_ms": int(spread_max_age_ms)})
        spread_used, spread_age_ms = spread_snapshot

        spread_limit = max(0.0, float(atr) * 0.3)
        if not math.isfinite(spread_limit):
            return self._hold("SPREAD_GUARD_UNAVAILABLE")
        if float(spread_used) > float(spread_limit):
            return self._hold(
                "HIGH_SPREAD_SKIP",
                {
                    "spread": float(spread_used),
                    "spread_limit": float(spread_limit),
                    "spread_age_ms": int(spread_age_ms),
                },
            )

        # Alpha: US Session Volatility Expansion (13:00-17:00 UTC)
        # Boost sweep buffer to avoid noise during high-impact windows.
        if 13 <= now_dt.hour < 17:
            sweep_buffer = float(atr) * float(self.sweep_buffer_atr) * 1.5
        else:
            sweep_buffer = float(atr) * float(self.sweep_buffer_atr)

        reclaim_buffer = float(atr) * float(self.reclaim_buffer_atr)
        stop_buffer = float(atr) * float(self.stop_buffer_atr)

        if st.state == StrategyState.COOLDOWN:
            stepped = self._cooldown_step(st)
            return stepped or self._hold("COOLDOWN_ACTIVE", {"remaining": int(st.cooldown_bars_remaining)})

        if st.state == StrategyState.ENTRY_PENDING and position is None:
            return self._hold("ENTRY_PENDING_WAIT_FILL")

        if st.state == StrategyState.IDLE and position is None:
            sweep_sell = mid >= (float(pdh) + sweep_buffer)
            sweep_buy = mid <= (float(pdl) - sweep_buffer)
            if not sweep_sell and not sweep_buy:
                return self._hold("IDLE_NO_SIGNAL")

            if sweep_sell:
                setup_side = Side.SELL
                level = float(pdh)
                level_name = "PDH"
                extreme = float(mid)
            else:
                setup_side = Side.BUY
                level = float(pdl)
                level_name = "PDL"
                extreme = float(mid)

            sweep_time_ms = int(now_ms)
            deadline_ms = sweep_time_ms + (int(self.reclaim_window_sec) * 1000)
            setup = _SweepSetup(
                side=setup_side,
                level_name=level_name,
                level=float(level),
                sweep_time_ms=sweep_time_ms,
                deadline_ms=deadline_ms,
                reclaim_window_sec=int(self.reclaim_window_sec),
                reclaim_extension_sec=int(self.reclaim_extension_sec),
                extended=False,
                sweep_buffer=float(sweep_buffer),
                reclaim_buffer=float(reclaim_buffer),
                stop_buffer=float(stop_buffer),
                extreme=float(extreme),
                extreme_time_ms=int(sweep_time_ms),
            )
            st.metadata["lsr_tick_setup"] = setup.to_dict()
            self._transition(st, StrategyState.SETUP, f"SWEEP_DETECTED_{level_name}")
            return self._hold(
                "LSR_TICK_SWEEP_DETECTED",
                {
                    "level_name": level_name,
                    "sweep_level": float(level),
                    "sweep_buffer": float(sweep_buffer),
                    "reclaim_buffer": float(reclaim_buffer),
                    "reclaim_window_sec": int(self.reclaim_window_sec),
                    "reclaim_extension_sec": int(self.reclaim_extension_sec),
                    "tick_time_utc": now_dt.isoformat(),
                },
            )

        if st.state == StrategyState.SETUP and position is None:
            setup = _SweepSetup.from_dict(st.metadata.get("lsr_tick_setup"))
            if setup is None:
                return self._reset_setup(st, "SETUP_MISSING")

            elapsed_ms = int(now_ms) - int(setup.sweep_time_ms)
            if elapsed_ms < 0:
                elapsed_ms = 0

            max_deadline_ms = int(setup.sweep_time_ms) + int(
                (setup.reclaim_window_sec + setup.reclaim_extension_sec) * 1000
            )
            if int(now_ms) > int(setup.deadline_ms):
                if (
                    (not setup.extended)
                    and setup.reclaim_extension_sec > 0
                    and int(setup.deadline_ms) < max_deadline_ms
                ):
                    setup.extended = True
                    setup.deadline_ms = max_deadline_ms
                    st.metadata["lsr_tick_setup"] = setup.to_dict()
                    return self._hold(
                        "LSR_TICK_RECLAIM_WINDOW_EXTENDED",
                        {
                            "level_name": setup.level_name,
                            "sweep_level": float(setup.level),
                            "elapsed_sec": float(elapsed_ms / 1000.0),
                            "new_deadline_sec": float((setup.deadline_ms - setup.sweep_time_ms) / 1000.0),
                        },
                    )
                return self._reset_setup(st, "RECLAIM_EXPIRED")

            if setup.side == Side.SELL:
                if float(mid) > float(setup.extreme):
                    setup.extreme = float(mid)
                    setup.extreme_time_ms = int(now_ms)
            else:
                if float(mid) < float(setup.extreme):
                    setup.extreme = float(mid)
                    setup.extreme_time_ms = int(now_ms)
            st.metadata["lsr_tick_setup"] = setup.to_dict()

            if setup.side == Side.SELL:
                reclaimed = mid <= (float(setup.level) - float(setup.reclaim_buffer))
            else:
                reclaimed = mid >= (float(setup.level) + float(setup.reclaim_buffer))
            if not reclaimed:
                return self._hold(
                    "LSR_TICK_WAIT_RECLAIM",
                    {
                        "level_name": setup.level_name,
                        "sweep_level": float(setup.level),
                        "elapsed_sec": float(elapsed_ms / 1000.0),
                        "deadline_sec": float((setup.deadline_ms - setup.sweep_time_ms) / 1000.0),
                        "extended": bool(setup.extended),
                        "extreme": float(setup.extreme),
                    },
                )

            sweep_velocity_ctx = self._segment_velocity(
                symbol=symbol,
                start_ms=int(setup.sweep_time_ms),
                end_ms=int(setup.extreme_time_ms),
            )
            # Use pre-sweep fallback window speed only when sweep->extreme speed is unavailable (e.g., extreme_time_ms == sweep_time_ms).
            if sweep_velocity_ctx is None:
                fallback_window_ms = int(max(float(self.displacement_window_sec) * 1000.0, 300.0))
                fallback_start_ms = int(setup.sweep_time_ms) - int(fallback_window_ms)
                sweep_velocity_ctx = self._segment_velocity(
                    symbol=symbol,
                    start_ms=int(fallback_start_ms),
                    end_ms=int(setup.sweep_time_ms),
                )
            reclaim_velocity_ctx = self._segment_velocity(
                symbol=symbol,
                start_ms=int(setup.extreme_time_ms),
                end_ms=int(now_ms),
            )
            if sweep_velocity_ctx is None or reclaim_velocity_ctx is None:
                return self._hold(
                    "RECLAIM_ACCEL_DATA_INSUFFICIENT_HOLD",
                    {
                        "sweep_velocity_ticks": int(sweep_velocity_ctx["tick_count"]) if sweep_velocity_ctx is not None else 0,
                        "reclaim_velocity_ticks": int(reclaim_velocity_ctx["tick_count"]) if reclaim_velocity_ctx is not None else 0,
                    },
                )

            sweep_velocity = float(sweep_velocity_ctx["velocity"])
            reclaim_velocity = float(reclaim_velocity_ctx["velocity"])
            if sweep_velocity <= 0.0 or reclaim_velocity <= 0.0:
                return self._hold(
                    "RECLAIM_ACCEL_DATA_INSUFFICIENT_HOLD",
                    {
                        "sweep_velocity": float(sweep_velocity),
                        "reclaim_velocity": float(reclaim_velocity),
                    },
                )
            reclaim_accel_ratio = float(reclaim_velocity / sweep_velocity)
            if reclaim_velocity < (1.3 * sweep_velocity):
                accel_reject_meta = {
                    "sweep_velocity": float(sweep_velocity),
                    "reclaim_velocity": float(reclaim_velocity),
                    "reclaim_accel_ratio": float(reclaim_accel_ratio),
                    "accel_ratio_min": 1.3,
                    "sweep_velocity_ticks": int(sweep_velocity_ctx["tick_count"]),
                    "reclaim_velocity_ticks": int(reclaim_velocity_ctx["tick_count"]),
                    "sweep_price_change_abs": float(sweep_velocity_ctx["price_change_abs"]),
                    "reclaim_price_change_abs": float(reclaim_velocity_ctx["price_change_abs"]),
                    "sweep_start_time_ms": int(sweep_velocity_ctx["start_time_ms"]),
                    "sweep_end_time_ms": int(sweep_velocity_ctx["end_time_ms"]),
                    "reclaim_start_time_ms": int(reclaim_velocity_ctx["start_time_ms"]),
                    "reclaim_end_time_ms": int(reclaim_velocity_ctx["end_time_ms"]),
                    "tick_time_ms": int(now_ms),
                }
                st.metadata["last_reclaim_accel_reject"] = dict(accel_reject_meta)
                LOGGER.info(
                    "%s reclaim acceleration reject %s ratio=%.4f min=%.2f sweep_v=%.6f reclaim_v=%.6f",
                    self.name,
                    symbol,
                    float(reclaim_accel_ratio),
                    1.3,
                    float(sweep_velocity),
                    float(reclaim_velocity),
                )
                return self._reset_setup_with_metadata(
                    st,
                    "RECLAIM_INSUFFICIENT_ACCELERATION",
                    metadata=accel_reject_meta,
                )

            disp_ratio = self._displacement_ratio(symbol, now_ms=int(now_ms))
            if disp_ratio is None or float(disp_ratio) < float(self.displacement_mult):
                st.metadata["last_displacement_ratio"] = disp_ratio
                return self._reset_setup(st, "RECLAIM_NO_DISPLACEMENT")

            entry_price = float(mid)
            if setup.side == Side.SELL:
                sl = float(setup.extreme) + float(setup.stop_buffer)
                risk_per_unit = abs(entry_price - float(sl))
                tp = entry_price - (risk_per_unit * float(tp_r2_dynamic))
            else:
                sl = float(setup.extreme) - float(setup.stop_buffer)
                risk_per_unit = abs(entry_price - float(sl))
                tp = entry_price + (risk_per_unit * float(tp_r2_dynamic))

            if risk_per_unit <= 0 or not math.isfinite(risk_per_unit):
                return self._reset_setup(st, "INVALID_RISK_PER_UNIT")

            st.pending_order = True
            self._transition(st, StrategyState.ENTRY_PENDING, "RECLAIM_DISPLACEMENT_ENTRY")

            signal_time = st.last_closed_bar_time or now_dt
            entry_metadata = {
                "sweep_level": float(setup.level),
                "sweep_level_name": setup.level_name,
                "sweep_extreme": float(setup.extreme),
                "sweep_time_utc": datetime.fromtimestamp(setup.sweep_time_ms / 1000.0, tz=timezone.utc).isoformat(),
                "reclaim_time_utc": now_dt.isoformat(),
                "reclaim_window_sec": int(setup.reclaim_window_sec),
                "reclaim_extension_sec": int(setup.reclaim_extension_sec),
                "reclaim_extended": bool(setup.extended),
                "displacement_ratio": float(disp_ratio),
                "displacement_mult": float(self.displacement_mult),
                "risk_per_unit": float(risk_per_unit),
                "expected_rr": float(tp_r2_dynamic),
                "entry_price": float(entry_price),
                "sl_price": float(sl),
                "tp_price": float(tp),
                "atr": float(atr),
                "entry_atr": tp_profile["entry_atr"],
                "atr_recent_mean": tp_profile["recent_atr_mean"],
                "atr_vs_recent_ratio": tp_profile["atr_vs_recent_ratio"],
                "atr_recent_sample_count": int(tp_profile["sample_count"]),
                "atr_profile": str(tp_profile["profile"]),
                "atr_low_vol_threshold_ratio": tp_profile["low_vol_threshold_ratio"],
                "atr_high_vol_threshold_ratio": tp_profile["high_vol_threshold_ratio"],
                "tp_r2_applied": float(tp_r2_dynamic),
                "tp_R2_dynamic": float(tp_r2_dynamic),
                "sweep_buffer": float(setup.sweep_buffer),
                "reclaim_buffer": float(setup.reclaim_buffer),
                "stop_buffer": float(setup.stop_buffer),
                "sweep_velocity": float(sweep_velocity),
                "reclaim_velocity": float(reclaim_velocity),
                "reclaim_accel_ratio": float(reclaim_accel_ratio),
                "reclaim_accel_ratio_min": 1.3,
                "sweep_velocity_ticks": int(sweep_velocity_ctx["tick_count"]),
                "reclaim_velocity_ticks": int(reclaim_velocity_ctx["tick_count"]),
                "tick_time_ms": int(now_ms),
            }
            st.metadata.pop("lsr_tick_setup", None)
            return self._emit_entry(
                side=setup.side,
                reason=f"LSR_TICK_{setup.side.value}_ENTRY",
                confidence=0.92,
                sl=float(sl),
                tp=float(tp),
                signal_bar_time=signal_time,
                min_hold_bars=int(self.min_hold_bars),
                metadata=entry_metadata,
            )

        if st.state == StrategyState.IN_POSITION and position is not None:
            entry_price = float(position.price_open)
            init_risk = self._finite_float(st.metadata.get("initial_risk_per_unit"))
            if init_risk is None:
                init_sl = self._finite_float(st.metadata.get("initial_sl"))
                if init_sl is None:
                    init_sl = self._finite_float(position.sl)
                if init_sl is not None:
                    init_risk = abs(entry_price - float(init_sl))
            if init_risk is None or init_risk <= 0:
                return self._hold("MANAGE_RISK_UNKNOWN")

            if position.side == Side.BUY:
                move = float(mid) - entry_price
            else:
                move = entry_price - float(mid)
            move_rr = float(move / init_risk) if init_risk > 0 else 0.0

            stage_a = self._emit_stage_a_partial(position=position, st=st, move_rr=move_rr)
            if stage_a is not None:
                return stage_a

            be_ready = bool(move_rr >= float(self.be_at_r) or st.metadata.get("stage_a_hit", False))
            if be_ready and not bool(st.metadata.get("be_moved", False)):
                current_sl = self._finite_float(position.sl)
                desired = float(entry_price)
                if position.side == Side.BUY:
                    improve = current_sl is None or desired > float(current_sl) + 1e-9
                else:
                    improve = current_sl is None or desired < float(current_sl) - 1e-9
                if improve:
                    st.metadata["be_moved"] = True
                    return self._hold("LSR_TICK_BREAK_EVEN", {"move_rr": float(move_rr)}, sl=float(desired))

            if self.trail_enabled and (move_rr >= float(self.trail_start_r) or bool(st.metadata.get("stage_a_hit", False))):
                now_s = float(now_ms) / 1000.0
                last_update_s = float(st.metadata.get("last_trail_update_ms", 0) or 0) / 1000.0
                if (now_s - last_update_s) >= float(self.trail_update_min_seconds):
                    current_sl = self._finite_float(position.sl)
                    if position.side == Side.BUY:
                        candidate = float(mid) - (float(atr) * float(self.trail_atr_mult))
                        candidate = max(candidate, entry_price)
                        if current_sl is None:
                            improve = True
                        else:
                            improve = candidate > float(current_sl) + (float(atr) * float(self.trail_min_step_atr))
                        if improve and candidate < float(mid):
                            st.metadata["last_trail_update_ms"] = int(now_ms)
                            return self._hold("LSR_TICK_TRAIL", {"move_rr": float(move_rr)}, sl=float(candidate))
                    else:
                        candidate = float(mid) + (float(atr) * float(self.trail_atr_mult))
                        candidate = min(candidate, entry_price)
                        if current_sl is None:
                            improve = True
                        else:
                            improve = candidate < float(current_sl) - (float(atr) * float(self.trail_min_step_atr))
                        if improve and candidate > float(mid):
                            st.metadata["last_trail_update_ms"] = int(now_ms)
                            return self._hold("LSR_TICK_TRAIL", {"move_rr": float(move_rr)}, sl=float(candidate))

            hold_seconds = 0.0
            open_time_utc: Optional[datetime] = None
            raw_open_time = position.time_open_utc
            if isinstance(raw_open_time, datetime):
                if raw_open_time.tzinfo is None:
                    open_time_utc = raw_open_time.replace(tzinfo=timezone.utc)
                else:
                    open_time_utc = raw_open_time.astimezone(timezone.utc)
            elif raw_open_time is not None:
                try:
                    open_ts = float(raw_open_time)
                    open_time_utc = datetime.fromtimestamp(open_ts, tz=timezone.utc)
                except Exception:
                    open_time_utc = None
            if open_time_utc is not None:
                hold_seconds = max(0.0, (now_dt - open_time_utc).total_seconds())
            if hold_seconds > self.max_hold_seconds and move_rr < 0.1:
                return self._emit_exit(
                    reason="LSR_TICK_ZOMBIE_EXIT",
                    confidence=0.7,
                    metadata={
                        "move_rr": float(move_rr),
                        "hold_seconds": float(hold_seconds),
                        "max_hold_seconds": int(self.max_hold_seconds),
                    },
                )

            return self._hold("IN_POSITION_HOLD", {"move_rr": float(move_rr)})

        return self._hold("NO_ACTION")
