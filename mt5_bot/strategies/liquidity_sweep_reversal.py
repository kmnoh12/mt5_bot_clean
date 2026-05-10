from __future__ import annotations

import importlib
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from core.models import DecisionAction, OrderResult, Position, Side, StrategyDecision, StrategyState, StrategySymbolState
from strategies.base import BaseStateMachineStrategy
from utils.indicators import compute_adx, compute_atr, compute_ema, parse_bar_time, sanitize_ohlc


class LiquiditySweepReversalStrategy(BaseStateMachineStrategy):
    """
    Liquidity Sweep Reversal (LSR)
    1) Detect sweep beyond local pivot.
    2) Confirm reclaim within time window + displacement candle.
    3) Enter reversal, stage out at R1, protect at break-even, finish at R2.
    """
    RUNTIME_OVERRIDE_KEYS = (
        "sweep_buffer_atr",
        "reclaim_window_sec",
        "displacement_mult",
    )

    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="liquidity_sweep_reversal", config=config, snapshot=snapshot)
        cfg = config or {}

        self.enabled = bool(cfg.get("enabled", True))
        self._base_params = self._coerce_param_bundle(cfg)
        self._symbol_params = self._coerce_symbol_params(cfg.get("symbol_params"))
        self._base_explicit_param_keys = self._extract_explicit_param_keys(cfg)
        self._symbol_explicit_param_keys = self._extract_symbol_explicit_param_keys(cfg.get("symbol_params"))
        self._recent_closed_candles: Dict[str, Any] = {}
        self._active_param_symbol = "__BASE__"
        self._activate_param_bundle(self._base_params)
        self._weekend_correction_cache_ttl_sec = max(
            30.0,
            self._to_float(cfg.get("weekend_correction_cache_ttl_sec", 300.0), 300.0),
        )
        self._weekend_correction_cache: Dict[str, Dict[str, Any]] = {}
        self._weekend_correction_fn = self._resolve_weekend_correction_fn()

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _extract_explicit_param_keys(raw_cfg: Any) -> set:
        if not isinstance(raw_cfg, dict):
            return set()
        keys = set()
        if "displacement_mult" in raw_cfg:
            keys.add("displacement_mult")
        if "sweep_buffer_atr" in raw_cfg or "sweep_buffer" in raw_cfg:
            keys.add("sweep_buffer_atr")
        return keys

    def _extract_symbol_explicit_param_keys(self, raw_symbol_cfg: Any) -> Dict[str, set]:
        if not isinstance(raw_symbol_cfg, dict):
            return {}
        out: Dict[str, set] = {}
        for symbol, payload in raw_symbol_cfg.items():
            symbol_key = str(symbol or "").strip().upper()
            if not symbol_key:
                continue
            out[symbol_key] = self._extract_explicit_param_keys(payload)
        return out

    def _coerce_param_bundle(
        self,
        raw_cfg: Dict[str, Any],
        seed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        defaults = dict(seed or {})

        atr_period = max(3, self._to_int(cfg.get("atr_period", defaults.get("atr_period", 14)), 14))
        trend_filter_enabled = bool(cfg.get("trend_filter_enabled", defaults.get("trend_filter_enabled", False)))
        trend_filter_ema_period = max(
            10,
            self._to_int(cfg.get("trend_filter_ema_period", defaults.get("trend_filter_ema_period", 200)), 200),
        )
        min_atr = max(
            0.0,
            self._to_float(cfg.get("min_atr", defaults.get("min_atr", 0.0)), 0.0),
        )
        pivot_lookback_sec = max(
            120,
            self._to_int(cfg.get("pivot_lookback_sec", defaults.get("pivot_lookback_sec", 1800)), 1800),
        )
        swing_window = max(5, self._to_int(cfg.get("swing_window", defaults.get("swing_window", 60)), 60))
        sweep_buffer_atr = max(
            0.0,
            self._to_float(
                cfg.get("sweep_buffer_atr", cfg.get("sweep_buffer", defaults.get("sweep_buffer_atr", 0.25))),
                0.25,
            ),
        )
        reclaim_buffer_atr = max(
            0.0,
            self._to_float(
                cfg.get("reclaim_buffer_atr", cfg.get("reclaim_buffer", defaults.get("reclaim_buffer_atr", 0.05))),
                0.05,
            ),
        )
        reclaim_window_sec = max(
            1,
            self._to_int(cfg.get("reclaim_window_sec", defaults.get("reclaim_window_sec", 15)), 15),
        )
        reclaim_extension_sec = max(
            0,
            self._to_int(
                cfg.get("reclaim_extension_sec", defaults.get("reclaim_extension_sec", 0)),
                0,
            ),
        )
        displacement_mult = max(
            1.0,
            self._to_float(cfg.get("displacement_mult", defaults.get("displacement_mult", 1.6)), 1.6),
        )
        displacement_lookback = max(
            5,
            self._to_int(cfg.get("displacement_lookback", defaults.get("displacement_lookback", 20)), 20),
        )
        sl_atr_mult = max(0.1, self._to_float(cfg.get("sl_atr_mult", defaults.get("sl_atr_mult", 0.8)), 0.8))
        stop_buffer_atr = max(
            0.0,
            self._to_float(cfg.get("stop_buffer_atr", defaults.get("stop_buffer_atr", 0.05)), 0.05),
        )
        tp_r1 = max(0.1, self._to_float(cfg.get("tp_R1", defaults.get("tp_r1", 1.2)), 1.2))
        tp_r2 = max(
            tp_r1 + 0.1,
            self._to_float(cfg.get("tp_R2", defaults.get("tp_r2", 2.5)), 2.5),
        )
        be_at_r = max(0.1, self._to_float(cfg.get("be_at_R", defaults.get("be_at_r", 1.0)), 1.0))
        max_hold_bars = max(2, self._to_int(cfg.get("max_hold_bars", defaults.get("max_hold_bars", 120)), 120))
        min_hold_bars = max(1, self._to_int(cfg.get("min_hold_bars", defaults.get("min_hold_bars", 1)), 1))
        zombie_bar_limit = max(
            5,
            self._to_int(cfg.get("zombie_bar_limit", defaults.get("zombie_bar_limit", 30)), 30),
        )
        zombie_rr_threshold = self._to_float(
            cfg.get("zombie_rr_threshold", defaults.get("zombie_rr_threshold", 0.2)),
            0.2,
        )
        hard_stop_rr = self._to_float(
            cfg.get("hard_stop_rr", defaults.get("hard_stop_rr", -2.0)),
            -2.0,
        )
        min_cooldown_bars = max(
            1,
            self._to_int(
                cfg.get("min_cooldown_bars", defaults.get("min_cooldown_bars", self.min_cooldown_bars)),
                self.min_cooldown_bars,
            ),
        )
        same_side_reentry_lock_bars = max(
            0,
            self._to_int(
                cfg.get(
                    "same_side_reentry_lock_bars",
                    defaults.get("same_side_reentry_lock_bars", 2),
                ),
                2,
            ),
        )
        weekend_monday_offset_hours = max(
            0,
            self._to_int(
                cfg.get("weekend_monday_offset_hours", defaults.get("weekend_monday_offset_hours", 6)),
                6,
            ),
        )

        atr_regime_window = max(
            10,
            self._to_int(cfg.get("atr_regime_window", defaults.get("atr_regime_window", 20)), 20),
        )
        atr_regime_min_ratio = max(
            0.05,
            self._to_float(cfg.get("atr_regime_min_ratio", defaults.get("atr_regime_min_ratio", 0.75)), 0.75),
        )
        chop_adaptive_risk_enabled = bool(
            cfg.get("chop_adaptive_risk_enabled", defaults.get("chop_adaptive_risk_enabled", False))
        )
        chop_risk_scale_min_raw = self._to_float(
            cfg.get("chop_risk_scale_min", defaults.get("chop_risk_scale_min", 0.70)),
            0.70,
        )
        chop_risk_scale_max_raw = self._to_float(
            cfg.get("chop_risk_scale_max", defaults.get("chop_risk_scale_max", 1.00)),
            1.00,
        )
        chop_risk_scale_min = min(1.0, max(0.0, chop_risk_scale_min_raw))
        chop_risk_scale_max = min(1.0, max(0.0, chop_risk_scale_max_raw))
        if chop_risk_scale_min > chop_risk_scale_max:
            chop_risk_scale_min, chop_risk_scale_max = chop_risk_scale_max, chop_risk_scale_min
        chop_risk_threshold = min(
            1.0,
            max(
                0.0,
                self._to_float(cfg.get("chop_risk_threshold", defaults.get("chop_risk_threshold", 0.55)), 0.55),
            ),
        )
        chop_risk_sensitivity = min(
            1.0,
            max(
                0.01,
                self._to_float(
                    cfg.get("chop_risk_sensitivity", defaults.get("chop_risk_sensitivity", 0.15)),
                    0.15,
                ),
            ),
        )
        adx_period = max(
            5,
            self._to_int(cfg.get("adx_period", defaults.get("adx_period", 14)), 14),
        )
        adx_trend_threshold = max(
            5.0,
            self._to_float(cfg.get("adx_trend_threshold", defaults.get("adx_trend_threshold", 25.0)), 25.0),
        )
        retest_enabled = bool(cfg.get("retest_enabled", defaults.get("retest_enabled", True)))
        retest_window_bars = max(
            1,
            self._to_int(cfg.get("retest_window_bars", defaults.get("retest_window_bars", 3)), 3),
        )
        retest_tolerance_atr = max(
            0.0,
            self._to_float(cfg.get("retest_tolerance_atr", defaults.get("retest_tolerance_atr", 0.10)), 0.10),
        )
        retest_reclaim_margin_atr = max(
            0.0,
            self._to_float(
                cfg.get("retest_reclaim_margin_atr", defaults.get("retest_reclaim_margin_atr", 0.05)),
                0.05,
            ),
        )
        fvg_enabled = bool(cfg.get("fvg_enabled", defaults.get("fvg_enabled", True)))
        adaptive_tp_enabled = bool(cfg.get("adaptive_tp_enabled", defaults.get("adaptive_tp_enabled", True)))
        tp_atr_low_ratio = max(
            0.05,
            self._to_float(cfg.get("tp_atr_low_ratio", defaults.get("tp_atr_low_ratio", 0.90)), 0.90),
        )
        tp_atr_high_ratio = max(
            tp_atr_low_ratio + 0.05,
            self._to_float(cfg.get("tp_atr_high_ratio", defaults.get("tp_atr_high_ratio", 1.20)), 1.20),
        )
        tp_r2_low_vol_mult = max(
            0.1,
            self._to_float(cfg.get("tp_r2_low_vol_mult", defaults.get("tp_r2_low_vol_mult", 0.85)), 0.85),
        )
        tp_r2_high_vol_mult = max(
            0.1,
            self._to_float(cfg.get("tp_r2_high_vol_mult", defaults.get("tp_r2_high_vol_mult", 1.10)), 1.10),
        )
        tp_r2_trend_penalty_mult = max(
            0.1,
            self._to_float(
                cfg.get("tp_r2_trend_penalty_mult", defaults.get("tp_r2_trend_penalty_mult", 0.80)),
                0.80,
            ),
        )
        tp_r2_floor = max(
            tp_r1 + 0.1,
            self._to_float(cfg.get("tp_r2_floor", defaults.get("tp_r2_floor", tp_r1 + 0.2)), tp_r1 + 0.2),
        )
        tp_r2_ceiling = max(
            tp_r2_floor + 0.1,
            self._to_float(
                cfg.get("tp_r2_ceiling", defaults.get("tp_r2_ceiling", max(tp_r2 * 1.35, tp_r2_floor + 0.1))),
                max(tp_r2 * 1.35, tp_r2_floor + 0.1),
            ),
        )

        # ── TP Trailing (추세 추적 익절) ──
        trail_tp_enabled = bool(cfg.get("trail_tp_enabled", defaults.get("trail_tp_enabled", True)))
        trail_tp_start_r = max(
            0.1,
            self._to_float(cfg.get("trail_tp_start_R", defaults.get("trail_tp_start_r", 1.0)), 1.0),
        )
        trail_tp_step_r = max(
            0.1,
            self._to_float(cfg.get("trail_tp_step_R", defaults.get("trail_tp_step_r", 0.5)), 0.5),
        )
        trail_tp_ratchet = bool(cfg.get("trail_tp_ratchet", defaults.get("trail_tp_ratchet", True)))

        return {
            "atr_period": atr_period,
            "trend_filter_enabled": trend_filter_enabled,
            "trend_filter_ema_period": trend_filter_ema_period,
            "min_atr": min_atr,
            "pivot_lookback_sec": pivot_lookback_sec,
            "swing_window": swing_window,
            "sweep_buffer_atr": sweep_buffer_atr,
            "reclaim_buffer_atr": reclaim_buffer_atr,
            "reclaim_window_sec": reclaim_window_sec,
            "reclaim_extension_sec": reclaim_extension_sec,
            "displacement_mult": displacement_mult,
            "displacement_lookback": displacement_lookback,
            "sl_atr_mult": sl_atr_mult,
            "stop_buffer_atr": stop_buffer_atr,
            "tp_r1": tp_r1,
            "tp_r2": tp_r2,
            "be_at_r": be_at_r,
            "max_hold_bars": max_hold_bars,
            "min_hold_bars": min_hold_bars,
            "zombie_bar_limit": zombie_bar_limit,
            "zombie_rr_threshold": zombie_rr_threshold,
            "hard_stop_rr": hard_stop_rr,
            "min_cooldown_bars": min_cooldown_bars,
            "same_side_reentry_lock_bars": same_side_reentry_lock_bars,
            "weekend_monday_offset_hours": weekend_monday_offset_hours,
            "atr_regime_window": atr_regime_window,
            "atr_regime_min_ratio": atr_regime_min_ratio,
            "chop_adaptive_risk_enabled": chop_adaptive_risk_enabled,
            "chop_risk_scale_min": chop_risk_scale_min,
            "chop_risk_scale_max": chop_risk_scale_max,
            "chop_risk_threshold": chop_risk_threshold,
            "chop_risk_sensitivity": chop_risk_sensitivity,
            "adx_period": adx_period,
            "adx_trend_threshold": adx_trend_threshold,
            "retest_enabled": retest_enabled,
            "retest_window_bars": retest_window_bars,
            "retest_tolerance_atr": retest_tolerance_atr,
            "retest_reclaim_margin_atr": retest_reclaim_margin_atr,
            "fvg_enabled": fvg_enabled,
            "adaptive_tp_enabled": adaptive_tp_enabled,
            "tp_atr_low_ratio": tp_atr_low_ratio,
            "tp_atr_high_ratio": tp_atr_high_ratio,
            "tp_r2_low_vol_mult": tp_r2_low_vol_mult,
            "tp_r2_high_vol_mult": tp_r2_high_vol_mult,
            "tp_r2_trend_penalty_mult": tp_r2_trend_penalty_mult,
            "tp_r2_floor": tp_r2_floor,
            "tp_r2_ceiling": tp_r2_ceiling,
            "trail_tp_enabled": trail_tp_enabled,
            "trail_tp_start_r": trail_tp_start_r,
            "trail_tp_step_r": trail_tp_step_r,
            "trail_tp_ratchet": trail_tp_ratchet,
        }

    def _coerce_symbol_params(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for symbol, payload in raw.items():
            key = str(symbol or "").strip().upper()
            if not key or not isinstance(payload, dict):
                continue
            out[key] = self._coerce_param_bundle(payload, seed=self._base_params)
        return out

    def _activate_param_bundle(self, bundle: Dict[str, Any]) -> None:
        self.atr_period = int(bundle["atr_period"])
        self.trend_filter_enabled = bool(bundle.get("trend_filter_enabled", False))
        self.trend_filter_ema_period = int(bundle.get("trend_filter_ema_period", 200))
        self.min_atr = float(bundle["min_atr"])
        self.pivot_lookback_sec = int(bundle["pivot_lookback_sec"])
        self.swing_window = int(bundle["swing_window"])
        self.sweep_buffer_atr = float(bundle["sweep_buffer_atr"])
        self.reclaim_buffer_atr = float(bundle["reclaim_buffer_atr"])
        self.reclaim_window_sec = int(bundle["reclaim_window_sec"])
        self.reclaim_extension_sec = int(bundle["reclaim_extension_sec"])
        self.displacement_mult = float(bundle["displacement_mult"])
        self.displacement_lookback = int(bundle["displacement_lookback"])
        self.sl_atr_mult = float(bundle["sl_atr_mult"])
        self.stop_buffer_atr = float(bundle["stop_buffer_atr"])
        self.tp_r1 = float(bundle["tp_r1"])
        self.tp_r2 = float(bundle["tp_r2"])
        self.be_at_r = float(bundle["be_at_r"])
        self.max_hold_bars = int(bundle["max_hold_bars"])
        self.min_hold_bars = int(bundle["min_hold_bars"])
        self.zombie_bar_limit = int(bundle["zombie_bar_limit"])
        self.zombie_rr_threshold = float(bundle["zombie_rr_threshold"])
        self.hard_stop_rr = float(bundle["hard_stop_rr"])
        self.min_cooldown_bars = int(bundle["min_cooldown_bars"])
        self.same_side_reentry_lock_bars = int(bundle["same_side_reentry_lock_bars"])
        self.weekend_monday_offset_hours = int(bundle["weekend_monday_offset_hours"])
        self.atr_regime_window = int(bundle["atr_regime_window"])
        self.atr_regime_min_ratio = float(bundle["atr_regime_min_ratio"])
        self.chop_adaptive_risk_enabled = bool(bundle["chop_adaptive_risk_enabled"])
        self.chop_risk_scale_min = float(bundle["chop_risk_scale_min"])
        self.chop_risk_scale_max = float(bundle["chop_risk_scale_max"])
        self.chop_risk_threshold = float(bundle["chop_risk_threshold"])
        self.chop_risk_sensitivity = float(bundle["chop_risk_sensitivity"])
        self.adx_period = int(bundle["adx_period"])
        self.adx_trend_threshold = float(bundle["adx_trend_threshold"])
        self.retest_enabled = bool(bundle["retest_enabled"])
        self.retest_window_bars = int(bundle["retest_window_bars"])
        self.retest_tolerance_atr = float(bundle["retest_tolerance_atr"])
        self.retest_reclaim_margin_atr = float(bundle["retest_reclaim_margin_atr"])
        self.fvg_enabled = bool(bundle["fvg_enabled"])
        self.adaptive_tp_enabled = bool(bundle["adaptive_tp_enabled"])
        self.tp_atr_low_ratio = float(bundle["tp_atr_low_ratio"])
        self.tp_atr_high_ratio = float(bundle["tp_atr_high_ratio"])
        self.tp_r2_low_vol_mult = float(bundle["tp_r2_low_vol_mult"])
        self.tp_r2_high_vol_mult = float(bundle["tp_r2_high_vol_mult"])
        self.tp_r2_trend_penalty_mult = float(bundle["tp_r2_trend_penalty_mult"])
        self.tp_r2_floor = float(bundle["tp_r2_floor"])
        self.tp_r2_ceiling = float(bundle["tp_r2_ceiling"])
        self.trail_tp_enabled = bool(bundle.get("trail_tp_enabled", True))
        self.trail_tp_start_r = float(bundle.get("trail_tp_start_r", 1.0))
        self.trail_tp_step_r = float(bundle.get("trail_tp_step_r", 0.5))
        self.trail_tp_ratchet = bool(bundle.get("trail_tp_ratchet", True))

    def apply_runtime_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(overrides, dict):
            return {}

        requested = {key: overrides[key] for key in self.RUNTIME_OVERRIDE_KEYS if key in overrides}
        if not requested:
            return {}

        merged_base = dict(self._base_params)
        merged_base.update(requested)
        normalized_base = self._coerce_param_bundle(merged_base, seed=self._base_params)
        applied = {key: normalized_base[key] for key in requested}

        self._base_params.update(applied)
        self._base_explicit_param_keys.update(applied.keys())

        for symbol, bundle in list(self._symbol_params.items()):
            merged_symbol = dict(bundle)
            merged_symbol.update(applied)
            self._symbol_params[symbol] = self._coerce_param_bundle(merged_symbol, seed=self._base_params)

        active_symbol = str(getattr(self, "_active_param_symbol", "__BASE__") or "__BASE__").upper()
        overrides_bundle = dict(self._symbol_params.get(active_symbol, self._base_params))
        self._activate_param_bundle(overrides_bundle)
        return applied

    def _activate_symbol_params(self, symbol: str) -> None:
        symbol_key = str(symbol or "").strip().upper()
        bundle = dict(self._symbol_params.get(symbol_key, self._base_params))
        if symbol_key == "BTCUSD":
            symbol_explicit = self._symbol_explicit_param_keys.get(symbol_key, set())
            if "displacement_mult" not in symbol_explicit and "displacement_mult" not in self._base_explicit_param_keys:
                bundle["displacement_mult"] = 1.35
            if "sweep_buffer_atr" not in symbol_explicit and "sweep_buffer_atr" not in self._base_explicit_param_keys:
                bundle["sweep_buffer_atr"] = 0.025
        if symbol_key == self._active_param_symbol:
            return
        self._activate_param_bundle(bundle)
        self._active_param_symbol = symbol_key

    @staticmethod
    def _resolve_weekend_correction_fn() -> Any:
        try:
            module = importlib.import_module("utils.weekend_calibrator")
        except Exception:
            return None
        fn = getattr(module, "compute_weekend_corrections", None)
        return fn if callable(fn) else None

    def _is_weekend(self, ts: Optional[datetime]) -> bool:
        ref = ts if isinstance(ts, datetime) else datetime.now(timezone.utc)
        weekday = ref.weekday()
        monday_cutoff_hour = min(23, max(0, int(getattr(self, "weekend_monday_offset_hours", 6))))
        # Extended Monday gap to cover chaotic Asian session (default up to 06:00 UTC)
        return weekday >= 5 or (weekday == 0 and ref.hour < monday_cutoff_hour)

    def _sanitize_weekend_corrections(self, raw: Any, *, default_reversion_sec: int) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {
            "k_penetration": 1.0,
            "t_reversion_sec": int(default_reversion_sec),
            "k_risk": 1.0,
            "k_displacement": 1.0,
            "k_sl_mult": 1.0,
            "source": "default",
        }
        if not isinstance(raw, dict):
            return sanitized

        k_penetration = self._finite_float(raw.get("k_penetration"))
        if k_penetration is not None and k_penetration > 0:
            sanitized["k_penetration"] = float(k_penetration)

        t_reversion_sec = self._to_int(raw.get("t_reversion_sec", default_reversion_sec), default_reversion_sec)
        sanitized["t_reversion_sec"] = max(1, int(t_reversion_sec))

        k_risk = self._finite_float(raw.get("k_risk"))
        if k_risk is not None and k_risk > 0:
            sanitized["k_risk"] = float(k_risk)

        k_disp = self._finite_float(raw.get("k_displacement"))
        if k_disp is not None and k_disp >= 1.0:
            sanitized["k_displacement"] = float(k_disp)

        k_sl_mult = self._finite_float(raw.get("k_sl_mult"))
        if k_sl_mult is not None and k_sl_mult >= 1.0:
            sanitized["k_sl_mult"] = float(k_sl_mult)

        sanitized["source"] = "calibrator"
        
        # Pass through raw metrics for telemetry if available
        if isinstance(raw, dict):
            for key in ["L_liquidity", "S_spread", "V_atr"]:
                val = self._finite_float(raw.get(key))
                if val is not None:
                    sanitized[key] = val
                    
        return sanitized

    def _get_weekend_corrections(self, symbol: str, *, default_reversion_sec: int) -> Dict[str, Any]:
        fallback = self._sanitize_weekend_corrections(None, default_reversion_sec=default_reversion_sec)
        if not callable(self._weekend_correction_fn):
            return fallback

        symbol_key = str(symbol or "").strip().upper()
        now_ts = time.time()
        cached = self._weekend_correction_cache.get(symbol_key)
        if isinstance(cached, dict):
            expires_ts = self._finite_float(cached.get("expires_ts")) or 0.0
            if expires_ts > now_ts:
                return self._sanitize_weekend_corrections(
                    cached.get("payload"),
                    default_reversion_sec=default_reversion_sec,
                )

        payload: Any = None
        try:
            payload = self._weekend_correction_fn(symbol_key)
        except Exception:
            payload = None

        self._weekend_correction_cache[symbol_key] = {
            "expires_ts": now_ts + self._weekend_correction_cache_ttl_sec,
            "payload": payload if isinstance(payload, dict) else None,
        }
        return self._sanitize_weekend_corrections(payload, default_reversion_sec=default_reversion_sec)

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

    @staticmethod
    def _safe_parse_time(value: Any) -> Optional[datetime]:
        parsed = parse_bar_time(value)
        if parsed is None:
            return None
        return parsed

    def _update_recent_closed_candles(self, symbol: str, closed: pd.DataFrame) -> None:
        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            return
        if closed is None or len(closed) < 3:
            self._recent_closed_candles.pop(symbol_key, None)
            return
        if "high" not in closed.columns or "low" not in closed.columns:
            self._recent_closed_candles.pop(symbol_key, None)
            return

        recent = closed.iloc[-3:]
        candles = []
        for _, row in recent.iterrows():
            high = self._finite_float(row.get("high"))
            low = self._finite_float(row.get("low"))
            if high is None or low is None or low > high:
                self._recent_closed_candles.pop(symbol_key, None)
                return
            candles.append({"high": float(high), "low": float(low)})
        self._recent_closed_candles[symbol_key] = candles

    def _check_fvg_confirmation(self, symbol: str, side: Side) -> bool:
        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            return False
        candles = self._recent_closed_candles.get(symbol_key)
        if not isinstance(candles, list) or len(candles) < 3:
            return False

        try:
            candle_n2 = candles[-3]
            candle_n = candles[-1]
            if not isinstance(candle_n2, dict) or not isinstance(candle_n, dict):
                return False

            n2_high = self._finite_float(candle_n2.get("high"))
            n2_low = self._finite_float(candle_n2.get("low"))
            n_high = self._finite_float(candle_n.get("high"))
            n_low = self._finite_float(candle_n.get("low"))
            if n2_high is None or n2_low is None or n_high is None or n_low is None:
                return False

            if side == Side.BUY:
                return bool(n2_high < n_low)
            if side == Side.SELL:
                return bool(n2_low > n_high)
            return False
        except Exception:
            return False

    def _bar_seconds(self, closed: pd.DataFrame) -> int:
        if "time" not in closed.columns or len(closed) < 2:
            return 60
        raw = pd.to_datetime(closed["time"], utc=True, errors="coerce")
        raw = raw.dropna()
        if len(raw) < 2:
            return 60
        diffs = raw.diff().dropna().dt.total_seconds()
        if diffs.empty:
            return 60
        med = self._finite_float(diffs.median())
        if med is None or med <= 0:
            return 60
        return max(1, int(round(med)))

    def _atr_regime_ratio(self, closed: pd.DataFrame) -> Optional[float]:
        period = max(2, int(self.atr_period))
        lookback = max(10, int(self.atr_regime_window))
        if len(closed) < (period + lookback + 2):
            return None

        high = pd.to_numeric(closed["high"], errors="coerce")
        low = pd.to_numeric(closed["low"], errors="coerce")
        close = pd.to_numeric(closed["close"], errors="coerce")
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = true_range.rolling(period).mean()
        if atr_series.empty:
            return None
        current_atr = self._finite_float(atr_series.iloc[-1])
        history = atr_series.iloc[-(lookback + 1) : -1]
        if history is None or len(history) < 3:
            return None
        history_median = self._finite_float(history.median())
        if current_atr is None or history_median is None or history_median <= 0:
            return None
        return float(current_atr / history_median)

    def _pivot_levels(self, closed: pd.DataFrame, bar_sec: int) -> Optional[Dict[str, float]]:
        lookback_bars = max(self.swing_window + 2, int(math.ceil(self.pivot_lookback_sec / max(1, bar_sec))))
        if len(closed) < lookback_bars + 2:
            return None

        reference = closed.iloc[-(lookback_bars + 1) : -1]
        if reference.empty:
            return None
        return {
            "pivot_high": float(reference["high"].max()),
            "pivot_low": float(reference["low"].min()),
        }

    def _displacement_ratio(self, closed: pd.DataFrame) -> Optional[float]:
        if len(closed) < self.displacement_lookback + 1:
            return None
        latest = closed.iloc[-1]
        latest_range = float(latest["high"]) - float(latest["low"])
        if latest_range <= 0:
            return None

        reference = closed.iloc[-(self.displacement_lookback + 1) : -1]
        ranges = (reference["high"] - reference["low"]).astype(float)
        med = self._finite_float(ranges.median())
        if med is None or med <= 0:
            return None
        return float(latest_range / med)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _compute_chop_score(
        self,
        *,
        displacement_ratio: Optional[float],
        atr_regime_ratio: Optional[float],
    ) -> float:
        disp = displacement_ratio if displacement_ratio is not None else self.displacement_mult
        regime = atr_regime_ratio if atr_regime_ratio is not None else 1.0

        disp_floor = 1.0
        disp_ceiling = max(disp_floor + 0.25, self.displacement_mult * 1.25)
        disp_norm = self._clamp((disp - disp_floor) / max(1e-9, (disp_ceiling - disp_floor)), 0.0, 1.0)

        regime_floor = max(0.05, self.atr_regime_min_ratio * 0.5)
        regime_ceiling = max(regime_floor + 0.25, 1.0)
        regime_norm = self._clamp((regime - regime_floor) / max(1e-9, (regime_ceiling - regime_floor)), 0.0, 1.0)

        trend_score = (disp_norm * 0.6) + (regime_norm * 0.4)
        # Higher chop score means less trend quality and a choppier market regime.
        return float(self._clamp(1.0 - trend_score, 0.0, 1.0))

    def _compute_chop_risk_scale(self, chop_score: float) -> float:
        scale_min = self._clamp(float(self.chop_risk_scale_min), 0.0, 1.0)
        scale_max = self._clamp(float(self.chop_risk_scale_max), 0.0, 1.0)
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min

        threshold = self._clamp(float(self.chop_risk_threshold), 0.0, 1.0)
        sensitivity = self._clamp(float(self.chop_risk_sensitivity), 0.01, 1.0)
        score = self._clamp(float(chop_score), 0.0, 1.0)

        if score <= threshold:
            return float(scale_max)
        if scale_max <= scale_min + 1e-12:
            return float(scale_min)

        # Above the threshold, risk is reduced in discrete sensitivity-based steps.
        max_steps = max(1, int(math.ceil((1.0 - threshold) / sensitivity)))
        step_idx = int(math.floor((score - threshold) / sensitivity)) + 1
        step_idx = max(0, min(max_steps, step_idx))
        step_fraction = step_idx / max_steps
        scaled = scale_max - ((scale_max - scale_min) * step_fraction)
        return float(self._clamp(scaled, scale_min, scale_max))

    def _adaptive_tp_profile(
        self,
        *,
        atr_regime_ratio: Optional[float],
        adx: Optional[float],
        weekend_profile_applied: bool,
    ) -> Dict[str, Any]:
        base_tp_r2 = float(self.tp_r2)
        if not bool(self.adaptive_tp_enabled):
            return {
                "tp_r2_base": float(base_tp_r2),
                "tp_r2_applied": float(base_tp_r2),
                "profile": "DISABLED",
                "atr_regime_ratio": float(atr_regime_ratio) if atr_regime_ratio is not None else None,
                "adx": float(adx) if adx is not None else None,
                "trend_threshold": float(self.adx_trend_threshold),
            }

        ratio = float(atr_regime_ratio) if atr_regime_ratio is not None else 1.0
        applied = float(base_tp_r2)
        profile = "NORMAL_VOL"

        if ratio <= float(self.tp_atr_low_ratio):
            applied *= float(self.tp_r2_low_vol_mult)
            profile = "LOW_VOL"
        elif ratio >= float(self.tp_atr_high_ratio):
            applied *= float(self.tp_r2_high_vol_mult)
            profile = "HIGH_VOL"

        trend_threshold = float(self.adx_trend_threshold + (2.0 if weekend_profile_applied else 0.0))
        if adx is not None and float(adx) >= trend_threshold:
            applied *= float(self.tp_r2_trend_penalty_mult)
            profile = f"{profile}_TREND_PENALTY"

        applied = float(self._clamp(applied, float(self.tp_r2_floor), float(self.tp_r2_ceiling)))
        return {
            "tp_r2_base": float(base_tp_r2),
            "tp_r2_applied": float(applied),
            "profile": str(profile),
            "atr_regime_ratio": float(ratio),
            "adx": float(adx) if adx is not None else None,
            "trend_threshold": float(trend_threshold),
        }

    def _build_pending_sweep(
        self,
        *,
        side: Side,
        level: float,
        extreme: float,
        sweep_time: Optional[datetime],
        reclaim_window_sec: Optional[int] = None,
        reclaim_extension_sec: Optional[int] = None,
        reclaim_extended: bool = False,
    ) -> Dict[str, Any]:
        active_reclaim_window_sec = self._to_int(reclaim_window_sec, self.reclaim_window_sec)
        extension_sec = self._to_int(reclaim_extension_sec, self.reclaim_extension_sec)
        return {
            "side": side.value,
            "level": float(level),
            "extreme": float(extreme),
            "sweep_time_utc": sweep_time.isoformat() if sweep_time is not None else "",
            "reclaim_window_sec": int(max(1, active_reclaim_window_sec)),
            "reclaim_extension_sec": int(max(0, extension_sec)),
            "reclaim_extended": bool(reclaim_extended),
        }

    def _build_pending_retest(
        self,
        *,
        side: Side,
        level: float,
        extreme: float,
        sweep_time: Optional[datetime],
        reclaim_time: Optional[datetime],
        window_bars: Optional[int] = None,
    ) -> Dict[str, Any]:
        active_window_bars = self._to_int(window_bars, self.retest_window_bars)
        return {
            "side": side.value,
            "level": float(level),
            "extreme": float(extreme),
            "sweep_time_utc": sweep_time.isoformat() if sweep_time is not None else "",
            "reclaim_time_utc": reclaim_time.isoformat() if reclaim_time is not None else "",
            "window_bars": int(max(1, active_window_bars)),
        }

    def _pending_sweep_from_state(self, st: StrategySymbolState) -> Optional[Dict[str, Any]]:
        raw = st.metadata.get("pending_sweep")
        if not isinstance(raw, dict):
            return None
        side_text = str(raw.get("side", "")).upper()
        if side_text not in Side.__members__:
            return None
        level = self._finite_float(raw.get("level"))
        extreme = self._finite_float(raw.get("extreme"))
        if level is None or extreme is None:
            return None
        sweep_time = self._safe_parse_time(raw.get("sweep_time_utc"))
        reclaim_window_sec = max(
            1,
            self._to_int(raw.get("reclaim_window_sec", self.reclaim_window_sec), self.reclaim_window_sec),
        )
        reclaim_extension_sec = max(
            0,
            self._to_int(raw.get("reclaim_extension_sec", self.reclaim_extension_sec), self.reclaim_extension_sec),
        )
        return {
            "side": Side[side_text],
            "level": float(level),
            "extreme": float(extreme),
            "sweep_time": sweep_time,
            "reclaim_window_sec": int(reclaim_window_sec),
            "reclaim_extension_sec": int(reclaim_extension_sec),
            "reclaim_extended": bool(raw.get("reclaim_extended", False)),
        }

    def _pending_retest_from_state(self, st: StrategySymbolState) -> Optional[Dict[str, Any]]:
        raw = st.metadata.get("pending_retest")
        if not isinstance(raw, dict):
            return None
        side_text = str(raw.get("side", "")).upper()
        if side_text not in Side.__members__:
            return None
        level = self._finite_float(raw.get("level"))
        extreme = self._finite_float(raw.get("extreme"))
        if level is None or extreme is None:
            return None
        reclaim_time = self._safe_parse_time(raw.get("reclaim_time_utc"))
        sweep_time = self._safe_parse_time(raw.get("sweep_time_utc"))
        window_bars = max(1, self._to_int(raw.get("window_bars", self.retest_window_bars), self.retest_window_bars))
        return {
            "side": Side[side_text],
            "level": float(level),
            "extreme": float(extreme),
            "reclaim_time": reclaim_time,
            "sweep_time": sweep_time,
            "window_bars": int(window_bars),
        }

    def _clear_pending_retest(self, st: StrategySymbolState) -> None:
        st.metadata.pop("pending_retest", None)
        st.metadata.pop("pending_retest_age_bars", None)

    def _clear_pending_sweep(self, st: StrategySymbolState) -> None:
        st.metadata.pop("pending_sweep", None)
        st.metadata.pop("pending_reclaim_age_sec", None)
        self._clear_pending_retest(st)

    def _manage_setup_state(
        self,
        *,
        symbol: str,
        st: StrategySymbolState,
        signal_bar_time: Optional[datetime],
        open_price: float,
        close_price: float,
        high_price: float,
        low_price: float,
        atr: float,
        bar_sec: int,
        sweep_buy: bool,
        sweep_sell: bool,
        reclaim_buy: bool,
        reclaim_sell: bool,
        is_displacement: bool,
        displacement_ratio: Optional[float],
        effective_disp_mult: float,
        atr_regime: str,
        atr_regime_ratio: Optional[float],
        is_low_vol_regime: bool,
        pivot_high: float,
        pivot_low: float,
        effective_reclaim_window_sec: int,
        effective_reclaim_extension_sec: int,
        risk_context: Dict[str, Any],
        weekend_profile: Dict[str, Any],
    ) -> StrategyDecision:
        pending = self._pending_sweep_from_state(st)
        if pending is None:
            self._transition(st, StrategyState.IDLE, "SETUP_MISSING_PENDING_SWEEP")
            return self._hold("SETUP_RESET")

        elapsed_sec = 0.0
        sweep_time = pending.get("sweep_time")
        if signal_bar_time is not None and isinstance(sweep_time, datetime):
            elapsed_sec = max(0.0, (signal_bar_time - sweep_time).total_seconds())
        st.metadata["pending_reclaim_age_sec"] = float(elapsed_sec)

        side = pending["side"]
        level = float(pending["level"])
        extreme = float(pending["extreme"])

        active_reclaim_window_sec = self._to_int(pending.get("reclaim_window_sec"), effective_reclaim_window_sec)
        pending_reclaim_extension_sec = self._to_int(
            pending.get("reclaim_extension_sec"),
            self.reclaim_extension_sec,
        )
        pending_reclaim_extended = bool(pending.get("reclaim_extended", False))
        pending_retest = self._pending_retest_from_state(st)
        if pending_retest is not None and pending_retest["side"] != side:
            self._clear_pending_retest(st)
            pending_retest = None

        if elapsed_sec > float(active_reclaim_window_sec):
            if (not pending_reclaim_extended) and pending_reclaim_extension_sec > 0:
                active_reclaim_window_sec = int(active_reclaim_window_sec) + int(pending_reclaim_extension_sec)
                st.metadata["pending_sweep"] = self._build_pending_sweep(
                    side=side,
                    level=level,
                    extreme=extreme,
                    sweep_time=sweep_time if isinstance(sweep_time, datetime) else None,
                    reclaim_window_sec=active_reclaim_window_sec,
                    reclaim_extension_sec=pending_reclaim_extension_sec,
                    reclaim_extended=True,
                )
                return self._hold(
                    "SETUP_RECLAIM_EXTENDED",
                    {
                        "pending_side": side.value,
                        "pending_level": level,
                        "pending_reclaim_age_sec": float(elapsed_sec),
                        "active_reclaim_window_sec": int(active_reclaim_window_sec),
                        "pending_reclaim_extended": True,
                        "pending_reclaim_extension_sec": int(pending_reclaim_extension_sec),
                    },
                )

            self._clear_pending_sweep(st)
            self._transition(st, StrategyState.IDLE, "RECLAIM_WINDOW_EXPIRED")
            return self._hold("SETUP_EXPIRED")

        if pending_retest is not None:
            retest_level = float(pending_retest["level"])
            retest_extreme = float(pending_retest["extreme"])
            retest_side = pending_retest["side"]
            retest_sweep_time = pending_retest.get("sweep_time")
            retest_window_bars = max(1, int(pending_retest.get("window_bars", self.retest_window_bars)))
            retest_reclaim_time = pending_retest.get("reclaim_time")
            retest_age_bars: Optional[int] = None
            if isinstance(signal_bar_time, datetime) and isinstance(retest_reclaim_time, datetime):
                elapsed_from_reclaim = max(0.0, (signal_bar_time - retest_reclaim_time).total_seconds())
                retest_age_bars = int(elapsed_from_reclaim // max(1, int(bar_sec)))
            st.metadata["pending_retest_age_bars"] = int(retest_age_bars) if retest_age_bars is not None else 0

            if retest_age_bars is not None and retest_age_bars > retest_window_bars:
                self._clear_pending_sweep(st)
                self._transition(st, StrategyState.IDLE, "RETEST_WINDOW_EXPIRED")
                return self._hold(
                    "SETUP_RETEST_EXPIRED",
                    {
                        "pending_side": retest_side.value,
                        "pending_level": float(retest_level),
                        "pending_retest_age_bars": int(retest_age_bars),
                        "retest_window_bars": int(retest_window_bars),
                    },
                )

            if retest_age_bars is None or retest_age_bars < 1:
                return self._hold(
                    "SETUP_WAIT_RETEST",
                    {
                        "pending_side": retest_side.value,
                        "pending_level": float(retest_level),
                        "pending_retest_age_bars": int(retest_age_bars or 0),
                        "retest_window_bars": int(retest_window_bars),
                        "retest_touched": False,
                    },
                )

            if is_low_vol_regime:
                return self._hold(
                    "LSR_LOW_VOL_BLOCK",
                    {
                        "atr_regime": atr_regime,
                        "atr_regime_ratio": float(atr_regime_ratio) if atr_regime_ratio is not None else None,
                        "atr_regime_min_ratio": float(self.atr_regime_min_ratio),
                        "pending_retest": True,
                    },
                )

            retest_tolerance = float(atr) * float(self.retest_tolerance_atr)
            retest_margin = float(atr) * float(self.retest_reclaim_margin_atr)
            if retest_side == Side.BUY:
                retest_touched = bool(low_price <= (retest_level + retest_tolerance))
                retest_accepted = bool(close_price >= (retest_level + retest_margin) and close_price > open_price)
                retest_ok = bool(retest_touched and retest_accepted)
                updated_extreme = min(retest_extreme, float(low_price))
            else:
                retest_touched = bool(high_price >= (retest_level - retest_tolerance))
                retest_accepted = bool(close_price <= (retest_level - retest_margin) and close_price < open_price)
                retest_ok = bool(retest_touched and retest_accepted)
                updated_extreme = max(retest_extreme, float(high_price))

            if retest_ok:
                return self._emit_lsr_entry(
                    st=st,
                    side=retest_side,
                    close_price=close_price,
                    atr=float(atr),
                    bar_sec=bar_sec,
                    signal_bar_time=signal_bar_time,
                    pending={
                        "level": float(retest_level),
                        "extreme": float(updated_extreme),
                        "sweep_time": retest_sweep_time if isinstance(retest_sweep_time, datetime) else sweep_time,
                    },
                    displacement_ratio=displacement_ratio,
                    atr_regime_ratio=atr_regime_ratio,
                    risk_context=risk_context,
                    weekend_profile=weekend_profile,
                )

            return self._hold(
                "SETUP_WAIT_RETEST",
                {
                    "pending_side": retest_side.value,
                    "pending_level": float(retest_level),
                    "pending_retest_age_bars": int(retest_age_bars),
                    "retest_window_bars": int(retest_window_bars),
                    "retest_touched": bool(retest_touched),
                    "retest_accepted": bool(retest_accepted),
                    "retest_tolerance": float(retest_tolerance),
                    "retest_margin": float(retest_margin),
                },
            )

        if side == Side.BUY and reclaim_buy and is_displacement:
            if is_low_vol_regime:
                return self._hold(
                    "LSR_LOW_VOL_BLOCK",
                    {
                        "atr_regime": atr_regime,
                        "atr_regime_ratio": float(atr_regime_ratio) if atr_regime_ratio is not None else None,
                        "atr_regime_min_ratio": float(self.atr_regime_min_ratio),
                    },
                )
            if self.fvg_enabled and not self._check_fvg_confirmation(symbol, Side.BUY):
                return self._hold(
                    "SETUP_WAIT_FVG_CONFIRMATION",
                    {
                        "pending_side": side.value,
                        "pending_level": level,
                        "fvg_confirmed": False,
                        "displacement_pass": bool(is_displacement),
                        "displacement_ratio": float(displacement_ratio) if displacement_ratio is not None else None,
                        "displacement_mult": float(effective_disp_mult),
                    },
                )
            if bool(self.retest_enabled):
                st.metadata["pending_retest"] = self._build_pending_retest(
                    side=Side.BUY,
                    level=level,
                    extreme=min(extreme, low_price),
                    sweep_time=sweep_time if isinstance(sweep_time, datetime) else signal_bar_time,
                    reclaim_time=signal_bar_time,
                    window_bars=self.retest_window_bars,
                )
                return self._hold(
                    "SETUP_WAIT_RETEST",
                    {
                        "pending_side": side.value,
                        "pending_level": level,
                        "pending_retest_age_bars": 0,
                        "retest_window_bars": int(self.retest_window_bars),
                        "retest_touched": False,
                    },
                )
            return self._emit_lsr_entry(
                st=st,
                side=Side.BUY,
                close_price=close_price,
                atr=float(atr),
                bar_sec=bar_sec,
                signal_bar_time=signal_bar_time,
                pending={
                    "level": level,
                    "extreme": min(extreme, low_price),
                    "sweep_time": sweep_time,
                },
                displacement_ratio=displacement_ratio,
                atr_regime_ratio=atr_regime_ratio,
                risk_context=risk_context,
                weekend_profile=weekend_profile,
            )
        if side == Side.SELL and reclaim_sell and is_displacement:
            if is_low_vol_regime:
                return self._hold(
                    "LSR_LOW_VOL_BLOCK",
                    {
                        "atr_regime": atr_regime,
                        "atr_regime_ratio": float(atr_regime_ratio) if atr_regime_ratio is not None else None,
                        "atr_regime_min_ratio": float(self.atr_regime_min_ratio),
                    },
                )
            if self.fvg_enabled and not self._check_fvg_confirmation(symbol, Side.SELL):
                return self._hold(
                    "SETUP_WAIT_FVG_CONFIRMATION",
                    {
                        "pending_side": side.value,
                        "pending_level": level,
                        "fvg_confirmed": False,
                        "displacement_pass": bool(is_displacement),
                        "displacement_ratio": float(displacement_ratio) if displacement_ratio is not None else None,
                        "displacement_mult": float(effective_disp_mult),
                    },
                )
            if bool(self.retest_enabled):
                st.metadata["pending_retest"] = self._build_pending_retest(
                    side=Side.SELL,
                    level=level,
                    extreme=max(extreme, high_price),
                    sweep_time=sweep_time if isinstance(sweep_time, datetime) else signal_bar_time,
                    reclaim_time=signal_bar_time,
                    window_bars=self.retest_window_bars,
                )
                return self._hold(
                    "SETUP_WAIT_RETEST",
                    {
                        "pending_side": side.value,
                        "pending_level": level,
                        "pending_retest_age_bars": 0,
                        "retest_window_bars": int(self.retest_window_bars),
                        "retest_touched": False,
                    },
                )
            return self._emit_lsr_entry(
                st=st,
                side=Side.SELL,
                close_price=close_price,
                atr=float(atr),
                bar_sec=bar_sec,
                signal_bar_time=signal_bar_time,
                pending={
                    "level": level,
                    "extreme": max(extreme, high_price),
                    "sweep_time": sweep_time,
                },
                displacement_ratio=displacement_ratio,
                atr_regime_ratio=atr_regime_ratio,
                risk_context=risk_context,
                weekend_profile=weekend_profile,
            )

        # Replace stale setup if opposite fresh sweep appears.
        if side == Side.BUY and sweep_sell:
            self._clear_pending_retest(st)
            st.metadata["pending_sweep"] = self._build_pending_sweep(
                side=Side.SELL,
                level=pivot_high,
                extreme=high_price,
                sweep_time=signal_bar_time,
                reclaim_window_sec=effective_reclaim_window_sec,
                reclaim_extension_sec=effective_reclaim_extension_sec,
                reclaim_extended=False,
            )
            self._transition(st, StrategyState.SETUP, "SETUP_REPLACED_BY_SELL_SWEEP")
            return self._hold("LSR_WAIT_SELL_RECLAIM")
        if side == Side.SELL and sweep_buy:
            self._clear_pending_retest(st)
            st.metadata["pending_sweep"] = self._build_pending_sweep(
                side=Side.BUY,
                level=pivot_low,
                extreme=low_price,
                sweep_time=signal_bar_time,
                reclaim_window_sec=effective_reclaim_window_sec,
                reclaim_extension_sec=effective_reclaim_extension_sec,
                reclaim_extended=False,
            )
            self._transition(st, StrategyState.SETUP, "SETUP_REPLACED_BY_BUY_SWEEP")
            return self._hold("LSR_WAIT_BUY_RECLAIM")

        return self._hold(
            "SETUP_WAIT_RECLAIM",
            {
                "pending_side": side.value,
                "pending_level": level,
                "pending_reclaim_age_sec": elapsed_sec,
                "active_reclaim_window_sec": int(max(1, active_reclaim_window_sec)),
                "pending_reclaim_extended": bool(pending_reclaim_extended),
                "displacement_ratio": displacement_ratio,
            },
        )

    @staticmethod
    def _normalize_utc_time(value: Optional[datetime]) -> Optional[datetime]:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _build_sweep_event_key(self, side: Side, sweep_time: Optional[datetime]) -> str:
        sweep_utc = self._normalize_utc_time(sweep_time)
        if sweep_utc is None:
            return ""
        return f"{side.value}|{sweep_utc.isoformat()}"

    def _record_position_close(
        self,
        st: StrategySymbolState,
        *,
        close_side: Optional[Side],
        close_time: Optional[datetime],
        source: str,
        marker: Optional[str] = None,
    ) -> None:
        if isinstance(close_side, Side):
            st.metadata["last_close_side"] = close_side.value
        else:
            st.metadata.pop("last_close_side", None)

        close_utc = self._normalize_utc_time(close_time)
        if close_utc is not None:
            st.metadata["last_close_bar_time_utc"] = close_utc.isoformat()
        else:
            st.metadata.pop("last_close_bar_time_utc", None)

        st.metadata["last_close_source"] = str(source or "")
        if marker:
            st.metadata["last_close_marker"] = marker

    def _bars_since_close(
        self,
        st: StrategySymbolState,
        *,
        signal_bar_time: Optional[datetime],
        bar_sec: int,
    ) -> Optional[int]:
        signal_utc = self._normalize_utc_time(signal_bar_time)
        if signal_utc is None:
            return None
        close_time = self._safe_parse_time(st.metadata.get("last_close_bar_time_utc"))
        close_utc = self._normalize_utc_time(close_time)
        if close_utc is None:
            return None
        elapsed_sec = (signal_utc - close_utc).total_seconds()
        if elapsed_sec <= 0:
            return 0
        unit_sec = max(1, int(bar_sec))
        return int(elapsed_sec // unit_sec)

    def _maybe_record_reconciliation_close(
        self,
        st: StrategySymbolState,
        *,
        signal_bar_time: Optional[datetime],
    ) -> None:
        reason = str(st.last_reason or "")
        if reason not in {"POSITION_MISSING", "RECONCILIATION_COOLDOWN"}:
            return
        marker = f"{reason}|{st.updated_at_utc.isoformat()}"
        if str(st.metadata.get("last_close_marker", "") or "") == marker:
            return
        close_time = signal_bar_time or st.last_closed_bar_time
        close_side = st.bias if isinstance(st.bias, Side) else None
        self._record_position_close(
            st,
            close_side=close_side,
            close_time=close_time,
            source=reason,
            marker=marker,
        )

    def _risk_hints(
        self,
        *,
        displacement_ratio: Optional[float],
        atr_regime_ratio: Optional[float],
        expected_rr: float,
        risk_context: Dict[str, Any],
    ) -> Dict[str, float]:
        disp = displacement_ratio if displacement_ratio is not None else 1.0
        disp_bonus = max(0.0, min(0.10, (disp - self.displacement_mult) * 0.08))
        regime_ratio = atr_regime_ratio if atr_regime_ratio is not None else 1.0

        loss_streak = max(0, int(self._finite_float(risk_context.get("loss_streak")) or 0))
        equity = self._finite_float(risk_context.get("equity"))
        equity_peak = self._finite_float(risk_context.get("equity_peak"))
        drawdown = 0.0
        if equity is not None and equity_peak is not None and equity_peak > 0:
            drawdown = max(0.0, (equity_peak - equity) / equity_peak)

        penalty = min(0.20, (loss_streak * 0.02) + (drawdown * 0.35))
        regime_tighten = max(0.0, min(0.25, (1.0 - regime_ratio) * 0.50))
        regime_volume_tighten = max(0.0, min(0.35, (1.0 - regime_ratio) * 0.70))
        win_probability = min(0.85, max(0.35, 0.54 + disp_bonus - penalty - regime_tighten))
        
        # Anti-Martingale: Slash size on streaks > 2 to protect equity
        streak_dampener = 0.5 if loss_streak > 2 else 1.0
        
        base_volume_scale = min(1.4, max(0.6, 0.9 + (disp - 1.0) * 0.2 - regime_volume_tighten))
        base_volume_scale *= streak_dampener
        
        chop_score = self._compute_chop_score(
            displacement_ratio=displacement_ratio,
            atr_regime_ratio=atr_regime_ratio,
        )
        chop_risk_scale = (
            self._compute_chop_risk_scale(chop_score)
            if bool(self.chop_adaptive_risk_enabled)
            else 1.0
        )
        volume_scale = max(0.0, base_volume_scale * chop_risk_scale)

        return {
            "win_probability": float(win_probability),
            "payoff_ratio": float(max(0.2, expected_rr)),
            "expected_rr": float(expected_rr),
            "volume_scale": float(volume_scale),
            "atr_regime_ratio": float(regime_ratio),
            "chop_score": float(chop_score),
            "chop_risk_scale": float(chop_risk_scale),
        }

    def _emit_lsr_entry(
        self,
        *,
        st: StrategySymbolState,
        side: Side,
        close_price: float,
        atr: float,
        bar_sec: int,
        signal_bar_time: Optional[datetime],
        pending: Dict[str, Any],
        displacement_ratio: Optional[float],
        atr_regime_ratio: Optional[float],
        risk_context: Dict[str, Any],
        weekend_profile: Dict[str, Any],
    ) -> StrategyDecision:
        level = float(pending["level"])
        extreme = float(pending["extreme"])
        sweep_time = self._safe_parse_time(pending.get("sweep_time"))
        sweep_event_key = self._build_sweep_event_key(side=side, sweep_time=sweep_time)
        last_handled_sweep_event_key = str(st.metadata.get("last_handled_sweep_event_key", "") or "")
        if sweep_event_key and sweep_event_key == last_handled_sweep_event_key:
            return self._hold(
                "LSR_DUPLICATE_GHOST_RECONCILIATION_SWEEP_EVENT",
                {"sweep_event_key": sweep_event_key},
            )

        reentry_lock_bars = max(0, int(self.same_side_reentry_lock_bars))
        if reentry_lock_bars > 0:
            last_close_side_text = str(st.metadata.get("last_close_side", "")).upper()
            if last_close_side_text in Side.__members__ and Side[last_close_side_text] == side:
                bars_since_close = self._bars_since_close(st, signal_bar_time=signal_bar_time, bar_sec=bar_sec)
                if bars_since_close is not None and bars_since_close < reentry_lock_bars:
                    return self._hold(
                        "LSR_SAME_SIDE_REENTRY_LOCK_ACTIVE",
                        {
                            "reentry_lock_bars": int(reentry_lock_bars),
                            "bars_since_close": int(bars_since_close),
                            "last_close_side": side.value,
                        },
                    )

        weekend_profile_applied = weekend_profile.get("applied") is True
        adx_threshold = 25.0 if weekend_profile_applied else 20.0
        adx = self._finite_float(risk_context.get("adx"))
        if adx is not None and adx < adx_threshold:
            return self._hold(
                "LSR_ADX_MOMENTUM_BLOCK",
                {"adx": float(adx), "adx_threshold": adx_threshold},
            )

        tp_profile = self._adaptive_tp_profile(
            atr_regime_ratio=atr_regime_ratio,
            adx=adx,
            weekend_profile_applied=weekend_profile_applied,
        )
        tp_r2_applied = float(tp_profile["tp_r2_applied"])
        expected_rr = (self.tp_r1 * 0.5) + (tp_r2_applied * 0.5)
        risk_hints = self._risk_hints(
            displacement_ratio=displacement_ratio,
            atr_regime_ratio=atr_regime_ratio,
            expected_rr=expected_rr,
            risk_context=risk_context,
        )
        weekend_k_risk = self._finite_float(weekend_profile.get("k_risk"))
        if weekend_k_risk is None or weekend_k_risk <= 0:
            weekend_k_risk = 1.0
        risk_hints["volume_scale"] = float(max(0.0, risk_hints["volume_scale"] * weekend_k_risk))

        weekend_k_sl_mult = float(weekend_profile.get("k_sl_mult", 1.0))
        effective_sl_mult = self.sl_atr_mult * max(1.0, weekend_k_sl_mult)

        if side == Side.BUY:
            structural_sl = min(extreme, close_price - (atr * effective_sl_mult)) - (atr * self.stop_buffer_atr)
            sl = float(structural_sl)
            risk_per_unit = close_price - sl
            tp = close_price + (risk_per_unit * tp_r2_applied)
            stage_a_target = close_price + (risk_per_unit * self.tp_r1)
        else:
            structural_sl = max(extreme, close_price + (atr * effective_sl_mult)) + (atr * self.stop_buffer_atr)
            sl = float(structural_sl)
            risk_per_unit = sl - close_price
            tp = close_price - (risk_per_unit * tp_r2_applied)
            stage_a_target = close_price - (risk_per_unit * self.tp_r1)

        risk_per_unit = max(1e-9, float(risk_per_unit))
        confidence = min(0.95, max(0.55, 0.6 + max(0.0, (risk_hints["volume_scale"] - 0.9) * 0.5)))

        st.bias = side
        st.pending_order = True
        st.entry_price = float(close_price)
        st.entry_bar_time = signal_bar_time
        st.peak_price = float(close_price)
        st.trough_price = float(close_price)
        st.metadata["initial_risk"] = float(risk_per_unit)
        st.metadata["risk_per_unit"] = float(risk_per_unit)
        st.metadata["stage_a_hit"] = False
        st.metadata["stage_a_target"] = float(stage_a_target)
        st.metadata["entry_level"] = float(level)
        st.metadata["entry_extreme"] = float(extreme)
        st.metadata["tp_r2_applied"] = float(tp_r2_applied)
        st.metadata["bars_in_trade"] = 0
        st.metadata["last_manage_bar_time"] = ""
        if sweep_event_key:
            st.metadata["last_handled_sweep_event_key"] = sweep_event_key
        self._clear_pending_sweep(st)
        self._transition(st, StrategyState.ENTRY_PENDING, "LSR_ENTRY_EMITTED")

        effective_reclaim_window_sec = self._to_int(
            weekend_profile.get("effective_reclaim_window_sec", self.reclaim_window_sec),
            self.reclaim_window_sec,
        )
        effective_sweep_buffer_atr = self._finite_float(weekend_profile.get("effective_sweep_buffer_atr"))
        if effective_sweep_buffer_atr is None or effective_sweep_buffer_atr <= 0:
            effective_sweep_buffer_atr = float(self.sweep_buffer_atr)
        weekend_k_penetration = self._finite_float(weekend_profile.get("k_penetration"))
        if weekend_k_penetration is None or weekend_k_penetration <= 0:
            weekend_k_penetration = 1.0
        weekend_t_reversion_sec = self._to_int(
            weekend_profile.get("t_reversion_sec", effective_reclaim_window_sec),
            effective_reclaim_window_sec,
        )

        entry_metadata: Dict[str, Any] = {
            "entry_style": "liquidity_sweep_reversal",
            "signal_close": float(close_price),
            "risk_per_unit": float(risk_per_unit),
            "sweep_level": float(level),
            "sweep_extreme": float(extreme),
            "sweep_event_key": sweep_event_key,
            "stage_a_target": float(stage_a_target),
            "tp_r1": float(self.tp_r1),
            "tp_r2": float(tp_r2_applied),
            "tp_r2_base": float(self.tp_r2),
            "tp_r2_applied": float(tp_r2_applied),
            "tp_profile": str(tp_profile.get("profile")),
            "adx_threshold": float(adx_threshold),
            "adx_entry": float(adx) if adx is not None else None,
            "adx_trend_threshold": float(tp_profile.get("trend_threshold", self.adx_trend_threshold)),
            "displacement_ratio": float(displacement_ratio) if displacement_ratio is not None else None,
            "reclaim_window_sec": int(max(1, effective_reclaim_window_sec)),
            "weekend_corrections_applied": bool(weekend_profile.get("applied", False)),
            "weekend_correction_source": str(weekend_profile.get("source", "default")),
            "weekend_k_penetration": float(weekend_k_penetration),
            "weekend_t_reversion_sec": int(max(1, weekend_t_reversion_sec)),
            "weekend_k_risk": float(weekend_k_risk),
            "effective_sweep_buffer_atr": float(effective_sweep_buffer_atr),
            "effective_reclaim_window_sec": int(max(1, effective_reclaim_window_sec)),
            **risk_hints,
        }
        return self._emit_entry(
            side=side,
            reason=f"LSR_{side.value}_ENTRY",
            confidence=confidence,
            sl=float(sl),
            tp=float(tp),
            signal_bar_time=signal_bar_time,
            min_hold_bars=self.min_hold_bars,
            metadata=entry_metadata,
        )

    def _maybe_stage_a_partial(
        self,
        *,
        position: Position,
        st: StrategySymbolState,
        move_rr: float,
    ) -> Optional[StrategyDecision]:
        if move_rr < self.tp_r1:
            return None
        if bool(st.metadata.get("stage_a_hit", False)):
            return None

        partial_volume = self._resolve_partial_close_volume(position=position, st=st)
        if partial_volume is None:
            return None

        position_volume = float(position.volume)
        st.metadata["stage_a_hit"] = True
        st.metadata["stage_a_rr"] = float(move_rr)
        st.metadata["stage_a_volume"] = float(partial_volume)

        return StrategyDecision(
            action=DecisionAction.EXIT,
            reason="LSR_STAGE_A_PARTIAL_CLOSE",
            strategy=self.name,
            confidence=0.82,
            volume=partial_volume,
            metadata={
                "is_partial": True,
                "partial_stage": "A",
                "position_volume_before": position_volume,
                "partial_volume_requested": partial_volume,
                "stage_a_rr": float(move_rr),
            },
        )

    def apply_order_result(
        self,
        symbol: str,
        decision: StrategyDecision,
        result: Optional[OrderResult],
    ) -> None:
        st = self._symbol_state(symbol)
        close_side = st.bias if isinstance(st.bias, Side) else None
        close_time = st.last_closed_bar_time
        full_exit_filled = bool(
            decision.action == DecisionAction.EXIT
            and result is not None
            and result.ok
            and not self._is_partial_exit_decision(decision)
        )
        super().apply_order_result(symbol=symbol, decision=decision, result=result)
        if full_exit_filled:
            self._record_position_close(
                st,
                close_side=close_side,
                close_time=close_time,
                source="EXIT_FILLED",
            )

    def _evaluate_impl(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: Optional[Position],
        st: StrategySymbolState,
    ) -> StrategyDecision:
        self._activate_symbol_params(symbol)
        clean = sanitize_ohlc(bars)
        if clean is None or len(clean) < max(self.atr_period + 6, self.displacement_lookback + 6):
            return self._hold("INSUFFICIENT_BARS")

        # All signals use closed bars only.
        closed = clean.iloc[:-1].copy() if len(clean) > 1 else clean.copy()
        if len(closed) < max(self.atr_period + 5, self.displacement_lookback + 5):
            return self._hold("INSUFFICIENT_CLOSED_BARS")
        self._update_recent_closed_candles(symbol, closed)

        atr = compute_atr(closed, period=self.atr_period)
        if atr is None or atr <= 0:
            return self._hold("INDICATORS_UNAVAILABLE")

        ema = None
        if self.trend_filter_enabled:
            ema = compute_ema(closed, period=self.trend_filter_ema_period)
            if ema is None:
                return self._hold("TREND_FILTER_UNAVAILABLE")

        atr_regime_ratio = self._atr_regime_ratio(closed)
        adx = compute_adx(closed, period=self.adx_period)

        bar_sec = self._bar_seconds(closed)
        levels = self._pivot_levels(closed, bar_sec=bar_sec)
        if levels is None:
            return self._hold("PIVOT_NOT_READY")

        latest = closed.iloc[-1]
        close_price = float(latest["close"])
        open_price = float(latest["open"])
        high_price = float(latest["high"])
        low_price = float(latest["low"])
        signal_bar_time = parse_bar_time(latest.get("time")) if "time" in closed.columns else None

        # Note: stale-data/market-closed checks are handled at the runtime/broker layer.

        displacement_ratio = self._displacement_ratio(closed)

        weekend_applied = self._is_weekend(signal_bar_time)
        weekend_raw = (
            self._get_weekend_corrections(symbol=symbol, default_reversion_sec=self.reclaim_window_sec)
            if weekend_applied
            else None
        )
        weekend_corrections = self._sanitize_weekend_corrections(
            weekend_raw,
            default_reversion_sec=self.reclaim_window_sec,
        )
        
        weekend_k_disp = float(weekend_corrections.get("k_displacement", 1.0)) if weekend_applied else 1.0
        effective_disp_mult = self.displacement_mult * weekend_k_disp
        is_displacement = displacement_ratio is not None and displacement_ratio >= effective_disp_mult
        is_low_vol_regime = bool(
            atr_regime_ratio is not None and atr_regime_ratio < self.atr_regime_min_ratio
        )
        atr_regime = "NO_ENTRY_LOW_VOL" if is_low_vol_regime else "NORMAL"

        weekend_k_penetration = (
            float(weekend_corrections["k_penetration"]) if weekend_applied else 1.0
        )
        weekend_k_risk = float(weekend_corrections["k_risk"]) if weekend_applied else 1.0
        weekend_k_sl_mult = float(weekend_corrections.get("k_sl_mult", 1.0)) if weekend_applied else 1.0

        effective_reclaim_window_sec = (
            int(weekend_corrections["t_reversion_sec"]) if weekend_applied else int(self.reclaim_window_sec)
        )
        effective_reclaim_extension_sec = 0 if weekend_applied else self.reclaim_extension_sec
        effective_sweep_buffer_atr = float(self.sweep_buffer_atr * weekend_k_penetration)
        weekend_profile: Dict[str, Any] = {
            "applied": bool(weekend_applied),
            "source": str(weekend_corrections.get("source", "default")),
            "k_penetration": float(weekend_k_penetration),
            "k_displacement": float(weekend_k_disp),
            "k_sl_mult": float(weekend_k_sl_mult),
            "effective_disp_mult": float(effective_disp_mult),
            "t_reversion_sec": int(max(1, effective_reclaim_window_sec)),
            "k_risk": float(weekend_k_risk),
            "effective_sweep_buffer_atr": float(effective_sweep_buffer_atr),
            "effective_reclaim_window_sec": int(max(1, effective_reclaim_window_sec)),
        }
        for key in ["L_liquidity", "S_spread", "V_atr"]:
            val = weekend_corrections.get(key)
            if val is not None:
                weekend_profile[key] = float(val)

        sweep_buffer = atr * effective_sweep_buffer_atr
        reclaim_buffer = atr * self.reclaim_buffer_atr * max(1.0, (atr_regime_ratio if atr_regime_ratio is not None else 1.0))
        pivot_high = float(levels["pivot_high"])
        pivot_low = float(levels["pivot_low"])

        sweep_buy = low_price <= (pivot_low - sweep_buffer)
        if sweep_buy and self.trend_filter_enabled and ema is not None and close_price < ema:
            sweep_buy = False

        sweep_sell = high_price >= (pivot_high + sweep_buffer)
        if sweep_sell and self.trend_filter_enabled and ema is not None and close_price > ema:
            sweep_sell = False
        reclaim_buy = close_price >= (pivot_low + reclaim_buffer) and close_price > open_price
        reclaim_sell = close_price <= (pivot_high - reclaim_buffer) and close_price < open_price

        risk_context = {
            "atr_regime": atr_regime,
            "atr_regime_ratio": float(atr_regime_ratio) if atr_regime_ratio is not None else None,
            "atr_regime_min_ratio": float(self.atr_regime_min_ratio),
            "adx": float(adx) if adx is not None else None,
        }
        attrs = getattr(bars, "attrs", {}) if bars is not None else {}
        if isinstance(attrs, dict):
            raw_risk_ctx = attrs.get("risk_context")
            if isinstance(raw_risk_ctx, dict):
                merged_ctx = dict(raw_risk_ctx)
                merged_ctx.update(risk_context)
                risk_context = merged_ctx

        if st.state == StrategyState.HALTED:
            return self._hold("HALTED_MANUAL_RECOVERY_REQUIRED")

        if st.state == StrategyState.COOLDOWN:
            self._maybe_record_reconciliation_close(st, signal_bar_time=signal_bar_time)
            if st.cooldown_bars_remaining > 0:
                st.cooldown_bars_remaining -= 1
                return self._hold("COOLDOWN_ACTIVE", {"remaining": st.cooldown_bars_remaining})
            self._transition(st, StrategyState.IDLE, "COOLDOWN_COMPLETE")

        if st.state == StrategyState.ENTRY_PENDING:
            if position is not None:
                self._transition(st, StrategyState.IN_POSITION, "POSITION_DETECTED_WHILE_PENDING")
            else:
                return self._hold("ENTRY_PENDING_WAIT_FILL")

        if atr < self.min_atr and st.state in (StrategyState.IDLE, StrategyState.SETUP):
            return self._hold("LSR_LOW_ATR_BLOCK")

        if st.state == StrategyState.IDLE:
            if sweep_buy and reclaim_buy and is_displacement:
                if is_low_vol_regime:
                    return self._hold(
                        "LSR_LOW_VOL_BLOCK",
                        {
                            "atr_regime": atr_regime,
                            "atr_regime_ratio": float(atr_regime_ratio) if atr_regime_ratio is not None else None,
                            "atr_regime_min_ratio": float(self.atr_regime_min_ratio),
                        },
                    )
                if self._check_fvg_confirmation(symbol, Side.BUY):
                    pending = self._build_pending_sweep(
                        side=Side.BUY,
                        level=pivot_low,
                        extreme=low_price,
                        sweep_time=signal_bar_time,
                        reclaim_window_sec=effective_reclaim_window_sec,
                        reclaim_extension_sec=effective_reclaim_extension_sec,
                        reclaim_extended=False,
                    )
                    if bool(self.retest_enabled):
                        st.metadata["pending_sweep"] = dict(pending)
                        st.metadata["pending_retest"] = self._build_pending_retest(
                            side=Side.BUY,
                            level=float(pending["level"]),
                            extreme=float(pending["extreme"]),
                            sweep_time=signal_bar_time,
                            reclaim_time=signal_bar_time,
                            window_bars=self.retest_window_bars,
                        )
                        self._transition(st, StrategyState.SETUP, "LSR_BUY_RECLAIM_WAIT_RETEST")
                        return self._hold(
                            "SETUP_WAIT_RETEST",
                            {
                                "pending_side": Side.BUY.value,
                                "pending_level": float(pending["level"]),
                                "pending_retest_age_bars": 0,
                                "retest_window_bars": int(self.retest_window_bars),
                                "retest_touched": False,
                            },
                        )
                    return self._emit_lsr_entry(
                        st=st,
                        side=Side.BUY,
                        close_price=close_price,
                        atr=float(atr),
                        bar_sec=bar_sec,
                        signal_bar_time=signal_bar_time,
                        pending={
                            "level": pending["level"],
                            "extreme": pending["extreme"],
                            "sweep_time": signal_bar_time,
                        },
                        displacement_ratio=displacement_ratio,
                        atr_regime_ratio=atr_regime_ratio,
                        risk_context=risk_context,
                        weekend_profile=weekend_profile,
                    )
            if sweep_sell and reclaim_sell and is_displacement:
                if is_low_vol_regime:
                    return self._hold(
                        "LSR_LOW_VOL_BLOCK",
                        {
                            "atr_regime": atr_regime,
                            "atr_regime_ratio": float(atr_regime_ratio) if atr_regime_ratio is not None else None,
                            "atr_regime_min_ratio": float(self.atr_regime_min_ratio),
                        },
                    )
                if self._check_fvg_confirmation(symbol, Side.SELL):
                    pending = self._build_pending_sweep(
                        side=Side.SELL,
                        level=pivot_high,
                        extreme=high_price,
                        sweep_time=signal_bar_time,
                        reclaim_window_sec=effective_reclaim_window_sec,
                        reclaim_extension_sec=effective_reclaim_extension_sec,
                        reclaim_extended=False,
                    )
                    if bool(self.retest_enabled):
                        st.metadata["pending_sweep"] = dict(pending)
                        st.metadata["pending_retest"] = self._build_pending_retest(
                            side=Side.SELL,
                            level=float(pending["level"]),
                            extreme=float(pending["extreme"]),
                            sweep_time=signal_bar_time,
                            reclaim_time=signal_bar_time,
                            window_bars=self.retest_window_bars,
                        )
                        self._transition(st, StrategyState.SETUP, "LSR_SELL_RECLAIM_WAIT_RETEST")
                        return self._hold(
                            "SETUP_WAIT_RETEST",
                            {
                                "pending_side": Side.SELL.value,
                                "pending_level": float(pending["level"]),
                                "pending_retest_age_bars": 0,
                                "retest_window_bars": int(self.retest_window_bars),
                                "retest_touched": False,
                            },
                        )
                    return self._emit_lsr_entry(
                        st=st,
                        side=Side.SELL,
                        close_price=close_price,
                        atr=float(atr),
                        bar_sec=bar_sec,
                        signal_bar_time=signal_bar_time,
                        pending={
                            "level": pending["level"],
                            "extreme": pending["extreme"],
                            "sweep_time": signal_bar_time,
                        },
                        displacement_ratio=displacement_ratio,
                        atr_regime_ratio=atr_regime_ratio,
                        risk_context=risk_context,
                        weekend_profile=weekend_profile,
                    )

            if sweep_buy:
                self._clear_pending_retest(st)
                st.metadata["pending_sweep"] = self._build_pending_sweep(
                    side=Side.BUY,
                    level=pivot_low,
                    extreme=low_price,
                    sweep_time=signal_bar_time,
                    reclaim_window_sec=effective_reclaim_window_sec,
                    reclaim_extension_sec=effective_reclaim_extension_sec,
                    reclaim_extended=False,
                )
                self._transition(st, StrategyState.SETUP, "LSR_BUY_SWEEP_DETECTED")
                return self._hold("LSR_WAIT_BUY_RECLAIM")

            if sweep_sell:
                self._clear_pending_retest(st)
                st.metadata["pending_sweep"] = self._build_pending_sweep(
                    side=Side.SELL,
                    level=pivot_high,
                    extreme=high_price,
                    sweep_time=signal_bar_time,
                    reclaim_window_sec=effective_reclaim_window_sec,
                    reclaim_extension_sec=effective_reclaim_extension_sec,
                    reclaim_extended=False,
                )
                self._transition(st, StrategyState.SETUP, "LSR_SELL_SWEEP_DETECTED")
                return self._hold("LSR_WAIT_SELL_RECLAIM")

            sweep_buy_miss = float(low_price - (pivot_low - sweep_buffer))
            sweep_sell_miss = float((pivot_high + sweep_buffer) - high_price)
            reclaim_buy_miss = float((pivot_low + reclaim_buffer) - close_price)
            reclaim_sell_miss = float(close_price - (pivot_high - reclaim_buffer))

            return self._hold(
                "NO_SWEEP_SETUP",
                {
                    "weekend_profile": weekend_profile,
                    "sweep_buy": bool(sweep_buy),
                    "sweep_sell": bool(sweep_sell),
                    "sweep_pass": bool(sweep_buy or sweep_sell),
                    "reclaim_buy": bool(reclaim_buy),
                    "reclaim_sell": bool(reclaim_sell),
                    "reclaim_pass": bool(reclaim_buy or reclaim_sell),
                    "displacement_pass": bool(is_displacement),
                    "displacement_ratio": float(displacement_ratio) if displacement_ratio is not None else None,
                    "displacement_mult": float(effective_disp_mult),
                    "sweep_buffer": float(sweep_buffer),
                    "reclaim_buffer": float(reclaim_buffer),
                    "miss_dist": {
                        "sweep_buy": sweep_buy_miss,
                        "sweep_sell": sweep_sell_miss,
                        "reclaim_buy": reclaim_buy_miss,
                        "reclaim_sell": reclaim_sell_miss,
                    },
                },
            )

        if st.state == StrategyState.SETUP:
            return self._manage_setup_state(
                symbol=symbol,
                st=st,
                signal_bar_time=signal_bar_time,
                open_price=open_price,
                close_price=close_price,
                high_price=high_price,
                low_price=low_price,
                atr=float(atr),
                bar_sec=bar_sec,
                sweep_buy=sweep_buy,
                sweep_sell=sweep_sell,
                reclaim_buy=reclaim_buy,
                reclaim_sell=reclaim_sell,
                is_displacement=is_displacement,
                displacement_ratio=displacement_ratio,
                effective_disp_mult=effective_disp_mult,
                atr_regime=atr_regime,
                atr_regime_ratio=atr_regime_ratio,
                is_low_vol_regime=is_low_vol_regime,
                pivot_high=pivot_high,
                pivot_low=pivot_low,
                effective_reclaim_window_sec=effective_reclaim_window_sec,
                effective_reclaim_extension_sec=effective_reclaim_extension_sec,
                risk_context=risk_context,
                weekend_profile=weekend_profile,
            )

        if st.state == StrategyState.IN_POSITION:
            if position is None:
                self._record_position_close(
                    st,
                    close_side=st.bias if isinstance(st.bias, Side) else None,
                    close_time=signal_bar_time or st.last_closed_bar_time,
                    source="POSITION_MISSING",
                )
                self._transition(st, StrategyState.COOLDOWN, "POSITION_NOT_FOUND")
                st.cooldown_bars_remaining = self.min_cooldown_bars
                return self._hold("WAITING_RECONCILIATION")

            current_bar_key = signal_bar_time.isoformat() if signal_bar_time is not None else ""
            last_manage_bar_key = str(st.metadata.get("last_manage_bar_time", "") or "")
            bar_advanced = bool(current_bar_key) and current_bar_key != last_manage_bar_key
            bars_in_trade = int(st.metadata.get("bars_in_trade", 0))
            if bar_advanced:
                bars_in_trade += 1
                st.metadata["bars_in_trade"] = bars_in_trade
                st.metadata["last_manage_bar_time"] = current_bar_key

            entry_price = st.entry_price if st.entry_price is not None else float(position.price_open)
            risk_per_unit = float(st.metadata.get("initial_risk", 0.0) or 0.0)
            if risk_per_unit <= 0 and position.sl is not None:
                risk_per_unit = abs(entry_price - float(position.sl))
            if risk_per_unit <= 0:
                risk_per_unit = max(float(atr) * self.sl_atr_mult, 1e-6)
            st.metadata["risk_per_unit"] = float(risk_per_unit)

            desired_sl = position.sl
            desired_tp = position.tp

            if position.side == Side.BUY:
                st.peak_price = max(st.peak_price or close_price, close_price)
            else:
                st.trough_price = min(st.trough_price or close_price, close_price)

            if position.side == Side.BUY:
                move_rr = (close_price - entry_price) / max(risk_per_unit, 1e-9)
            else:
                move_rr = (entry_price - close_price) / max(risk_per_unit, 1e-9)
            tp_r2_target = float(st.metadata.get("tp_r2_applied", self.tp_r2) or self.tp_r2)

            # ── TP Trailing: 추세 방향으로 TP 상향 추적 ──
            if self.trail_tp_enabled and move_rr >= self.trail_tp_start_r:
                # 현재 move_rr이 trail_tp_start_r를 넘은 초과분을 계산
                excess_rr = move_rr - self.trail_tp_start_r
                # 초과분 1R당 trail_tp_step_r만큼 TP를 추가 이동
                tp_boost_r = excess_rr * self.trail_tp_step_r
                # 새 TP = 초기 TP + boost
                initial_tp_r = tp_r2_target
                new_tp_r = initial_tp_r + tp_boost_r
                if position.side == Side.BUY:
                    new_tp = entry_price + (new_tp_r * risk_per_unit)
                    if self.trail_tp_ratchet and desired_tp is not None:
                        new_tp = max(float(desired_tp), new_tp)
                    desired_tp = new_tp
                else:
                    new_tp = entry_price - (new_tp_r * risk_per_unit)
                    if self.trail_tp_ratchet and desired_tp is not None:
                        new_tp = min(float(desired_tp), new_tp)
                    desired_tp = new_tp
                st.metadata["trail_tp_active"] = True
                st.metadata["trail_tp_boost_r"] = float(tp_boost_r)
                st.metadata["trail_tp_new_r"] = float(new_tp_r)
                st.metadata["trail_tp_price"] = float(desired_tp)
                # TP가 추적되고 있으면 tp_r2_target도 동적으로 올림
                tp_r2_target = new_tp_r

            if bars_in_trade > 20 and move_rr < float(self.hard_stop_rr):
                self._transition(st, StrategyState.EXIT_READY, "LSR_DRAWDOWN_CAP_EXIT")
            elif bars_in_trade > self.zombie_bar_limit and move_rr < self.zombie_rr_threshold:
                self._transition(st, StrategyState.EXIT_READY, "LSR_STAGNANT_EXIT")
            elif position.side == Side.BUY:
                partial = self._maybe_stage_a_partial(position=position, st=st, move_rr=move_rr)
                if partial is not None:
                    return partial

                if bool(st.metadata.get("stage_a_hit", False)) or move_rr >= self.be_at_r:
                    be_sl = float(entry_price)
                    desired_sl = max(float(desired_sl) if desired_sl is not None else be_sl, be_sl)

                if move_rr >= tp_r2_target and bars_in_trade >= self.min_hold_bars:
                    self._transition(st, StrategyState.EXIT_READY, "LSR_BUY_TP_R2")
                elif bar_advanced and bars_in_trade >= self.max_hold_bars:
                    self._transition(st, StrategyState.EXIT_READY, "LSR_BUY_TIME_STOP")
                else:
                    return self._hold(
                        "LSR_BUY_MANAGE",
                        {
                            "bars_in_trade": bars_in_trade,
                            "move_rr": float(move_rr),
                            "stage_a_hit": bool(st.metadata.get("stage_a_hit", False)),
                            "risk_per_unit": float(risk_per_unit),
                            "tp_r2_target": float(tp_r2_target),
                            "trail_tp_active": bool(st.metadata.get("trail_tp_active", False)),
                            "trail_tp_boost_r": float(st.metadata.get("trail_tp_boost_r", 0.0)),
                        },
                        sl=desired_sl,
                        tp=desired_tp,
                    )
            else:
                partial = self._maybe_stage_a_partial(position=position, st=st, move_rr=move_rr)
                if partial is not None:
                    return partial

                if bool(st.metadata.get("stage_a_hit", False)) or move_rr >= self.be_at_r:
                    be_sl = float(entry_price)
                    desired_sl = min(float(desired_sl) if desired_sl is not None else be_sl, be_sl)

                if move_rr >= tp_r2_target and bars_in_trade >= self.min_hold_bars:
                    self._transition(st, StrategyState.EXIT_READY, "LSR_SELL_TP_R2")
                elif bar_advanced and bars_in_trade >= self.max_hold_bars:
                    self._transition(st, StrategyState.EXIT_READY, "LSR_SELL_TIME_STOP")
                else:
                    return self._hold(
                        "LSR_SELL_MANAGE",
                        {
                            "bars_in_trade": bars_in_trade,
                            "move_rr": float(move_rr),
                            "stage_a_hit": bool(st.metadata.get("stage_a_hit", False)),
                            "risk_per_unit": float(risk_per_unit),
                            "tp_r2_target": float(tp_r2_target),
                            "trail_tp_active": bool(st.metadata.get("trail_tp_active", False)),
                            "trail_tp_boost_r": float(st.metadata.get("trail_tp_boost_r", 0.0)),
                        },
                        sl=desired_sl,
                        tp=desired_tp,
                    )

        if st.state == StrategyState.EXIT_READY:
            st.pending_order = False
            return self._emit_exit(
                reason=f"LSR_EXIT:{st.last_reason}",
                confidence=0.78,
                metadata={
                    "tp_r1": float(self.tp_r1),
                    "tp_r2": float(st.metadata.get("tp_r2_applied", self.tp_r2) or self.tp_r2),
                },
            )

        return self._hold("NO_ACTION")
