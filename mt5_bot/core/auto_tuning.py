from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

from utils.indicators import compute_adx, compute_atr, sanitize_ohlc


class ParameterAutoTuningLoop:
    TUNABLE_PARAMETERS: Tuple[str, ...] = (
        "trend_score_threshold",
        "trend_strength_threshold",
        "meanrev_max_strength",
        "breakout_lookback",
        "trend_sl_atr_mult",
        "trend_tp_r_multiple",
        "trailing_atr_mult",
        "trailing_start_rr",
        "regime_flip_exit_threshold",
        "sweep_buffer_atr",
        "reclaim_window_sec",
        "displacement_mult",
    )

    _DEFAULT_BASELINE: Dict[str, float] = {
        "trend_score_threshold": 0.32,
        "trend_strength_threshold": 0.42,
        "meanrev_max_strength": 0.40,
        "breakout_lookback": 5.0,
        "trend_sl_atr_mult": 1.2,
        "trend_tp_r_multiple": 2.1,
        "trailing_atr_mult": 1.0,
        "trailing_start_rr": 0.8,
        "regime_flip_exit_threshold": 0.18,
        "sweep_buffer_atr": 0.25,
        "reclaim_window_sec": 15.0,
        "displacement_mult": 1.6,
    }

    _DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
        "trend_score_threshold": (0.08, 0.85),
        "trend_strength_threshold": (0.12, 0.90),
        "meanrev_max_strength": (0.12, 0.80),
        "breakout_lookback": (3.0, 30.0),
        "trend_sl_atr_mult": (0.6, 3.2),
        "trend_tp_r_multiple": (0.8, 5.0),
        "trailing_atr_mult": (0.4, 3.0),
        "trailing_start_rr": (0.1, 3.0),
        "regime_flip_exit_threshold": (0.05, 0.55),
        "sweep_buffer_atr": (0.0, 1.5),
        "reclaim_window_sec": (5.0, 900.0),
        "displacement_mult": (1.0, 4.0),
    }

    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        root_cfg = config or {}
        auto_cfg = dict(root_cfg.get("auto_tuning", {})) if isinstance(root_cfg.get("auto_tuning"), dict) else {}
        strategies_cfg = dict(root_cfg.get("strategies", {})) if isinstance(root_cfg.get("strategies"), dict) else {}
        trend_cfg = (
            dict(strategies_cfg.get("trend_regime_sm", {}))
            if isinstance(strategies_cfg.get("trend_regime_sm"), dict)
            else {}
        )
        lsr_cfg = (
            dict(strategies_cfg.get("liquidity_sweep_reversal", {}))
            if isinstance(strategies_cfg.get("liquidity_sweep_reversal"), dict)
            else {}
        )

        self.enabled = bool(auto_cfg.get("enabled", False))
        self.target_symbols = self._normalize_symbols(auto_cfg.get("target_symbols", ["BTCUSD", "ETHUSD"]))
        self.tune_interval_seconds = max(5.0, float(auto_cfg.get("tune_interval_seconds", 300)))
        self.lookback_bars = max(20, int(auto_cfg.get("lookback_bars", 120)))
        self.min_bars = max(20, int(auto_cfg.get("min_bars", 80)))
        self.smoothing_alpha = self._clamp(float(auto_cfg.get("smoothing_alpha", 0.25)), 0.01, 1.0)

        self.atr_period = max(5, int(trend_cfg.get("atr_period", 14)))
        self.adx_period = max(5, int(trend_cfg.get("adx_period", 14)))

        self.bounds = self._load_bounds(auto_cfg.get("parameter_bounds", {}))
        self.base_parameters = self._load_base_parameters(trend_cfg=trend_cfg, lsr_cfg=lsr_cfg)
        self._bars_by_symbol: Dict[str, pd.DataFrame] = {}

        snap = snapshot if isinstance(snapshot, dict) else {}
        self._last_tuned_at = max(0.0, float(snap.get("last_tuned_at", 0.0) or 0.0))
        self._last_skip_reason = str(snap.get("last_skip_reason", "") or "")
        self._update_count = max(0, int(snap.get("update_count", 0) or 0))
        self._last_metrics = dict(snap.get("metrics", {})) if isinstance(snap.get("metrics"), dict) else {}
        self._last_symbol_metrics = (
            dict(snap.get("symbol_metrics", {})) if isinstance(snap.get("symbol_metrics"), dict) else {}
        )
        snap_overrides = dict(snap.get("overrides", {})) if isinstance(snap.get("overrides"), dict) else {}
        self._overrides = self._normalize_overrides(snap_overrides) if snap_overrides else {}

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _normalize_symbols(raw: Any) -> Tuple[str, ...]:
        if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
            return tuple()
        symbols = []
        for item in raw:
            text = str(item or "").strip().upper()
            if text and text not in symbols:
                symbols.append(text)
        return tuple(symbols)

    def _load_bounds(self, raw_bounds: Any) -> Dict[str, Tuple[float, float]]:
        bounds_cfg = raw_bounds if isinstance(raw_bounds, dict) else {}
        loaded: Dict[str, Tuple[float, float]] = {}
        for name in self.TUNABLE_PARAMETERS:
            default_lo, default_hi = self._DEFAULT_BOUNDS[name]
            section = bounds_cfg.get(name, {})
            if not isinstance(section, dict):
                loaded[name] = (float(default_lo), float(default_hi))
                continue
            try:
                lo = float(section.get("min", default_lo))
            except (TypeError, ValueError):
                lo = float(default_lo)
            try:
                hi = float(section.get("max", default_hi))
            except (TypeError, ValueError):
                hi = float(default_hi)
            if hi < lo:
                hi = lo
            loaded[name] = (float(lo), float(hi))
        return loaded

    def _load_base_parameters(self, trend_cfg: Dict[str, Any], lsr_cfg: Dict[str, Any]) -> Dict[str, float]:
        base: Dict[str, float] = {}
        lsr_parameter_keys = {"sweep_buffer_atr", "reclaim_window_sec", "displacement_mult"}
        for name in self.TUNABLE_PARAMETERS:
            strategy_cfg = lsr_cfg if name in lsr_parameter_keys else trend_cfg
            raw_value = strategy_cfg.get(name, self._DEFAULT_BASELINE[name])
            try:
                coerced = self._coerce_parameter(name, raw_value)
            except (TypeError, ValueError):
                coerced = self._coerce_parameter(name, self._DEFAULT_BASELINE[name])
            base[name] = self._apply_bound(name, coerced)
        return base

    def _coerce_parameter(self, name: str, raw_value: Any) -> float:
        value = float(raw_value)
        if name == "breakout_lookback":
            return float(max(3, int(round(value))))
        if name == "reclaim_window_sec":
            return float(max(1, int(round(value))))
        return float(value)

    def _apply_bound(self, name: str, raw_value: float) -> float:
        lo, hi = self.bounds[name]
        bounded = self._clamp(float(raw_value), lo, hi)
        if name == "breakout_lookback":
            return float(max(int(round(lo)), min(int(round(hi)), int(round(bounded)))))
        if name == "reclaim_window_sec":
            return float(max(int(round(lo)), min(int(round(hi)), int(round(bounded)))))
        return float(bounded)

    def _normalize_overrides(self, overrides: Dict[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for name in self.TUNABLE_PARAMETERS:
            if name not in overrides:
                continue
            try:
                normalized[name] = self._apply_bound(name, self._coerce_parameter(name, overrides[name]))
            except (TypeError, ValueError):
                continue
        return normalized

    def ingest_symbol_bars(self, symbol: str, bars: Any) -> None:
        text = str(symbol or "").strip().upper()
        if not text:
            return
        if self.target_symbols and text not in self.target_symbols:
            return
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            return
        self._bars_by_symbol[text] = bars.tail(max(self.lookback_bars + 8, self.min_bars + 8)).copy()

    def ingest_bars(self, bars_by_symbol: Dict[str, Any]) -> None:
        if not isinstance(bars_by_symbol, dict):
            return
        for symbol, bars in bars_by_symbol.items():
            self.ingest_symbol_bars(symbol, bars)

    def _compute_whipsaw_ratio(self, returns: pd.Series) -> float:
        prev_sign = 0
        flips = 0
        transitions = 0
        for value in returns:
            if value > 0:
                sign = 1
            elif value < 0:
                sign = -1
            else:
                sign = 0
            if sign == 0:
                continue
            if prev_sign != 0:
                transitions += 1
                if sign != prev_sign:
                    flips += 1
            prev_sign = sign
        if transitions <= 0:
            return 0.0
        return float(flips) / float(transitions)

    def _compute_symbol_metrics(self, bars: pd.DataFrame) -> Optional[Dict[str, float]]:
        clean = sanitize_ohlc(bars)
        if clean is None or len(clean) < (self.min_bars + 1):
            return None

        closed = clean.iloc[:-1].copy()
        if len(closed) < self.min_bars:
            return None

        window = closed.tail(self.lookback_bars).reset_index(drop=True)
        if len(window) < self.min_bars:
            return None

        atr = compute_atr(window, period=self.atr_period)
        adx = compute_adx(window, period=self.adx_period)
        if atr is None or adx is None:
            return None

        close_last = float(window["close"].iloc[-1])
        if close_last <= 0:
            return None

        returns = window["close"].pct_change().dropna()
        if returns.empty:
            return None

        return_volatility = float(returns.std(ddof=0))
        returns_abs_sum = float(returns.abs().sum())
        trend_persistence = abs(float(returns.sum())) / max(returns_abs_sum, 1e-12)
        whipsaw_proxy = self._compute_whipsaw_ratio(returns)
        chop_proxy = self._clamp(((1.0 - trend_persistence) * 0.55) + (whipsaw_proxy * 0.45), 0.0, 1.0)
        atr_percent = float(atr) / max(close_last, 1e-12)

        return {
            "atr_percent": float(max(0.0, atr_percent)),
            "return_volatility": float(max(0.0, return_volatility)),
            "trend_persistence": float(self._clamp(trend_persistence, 0.0, 1.0)),
            "adx": float(max(0.0, adx)),
            "whipsaw_proxy": float(self._clamp(whipsaw_proxy, 0.0, 1.0)),
            "chop_proxy": float(chop_proxy),
        }

    def _aggregate_metrics(self, symbol_metrics: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not symbol_metrics:
            return None
        count = float(len(symbol_metrics))
        keys = (
            "atr_percent",
            "return_volatility",
            "trend_persistence",
            "adx",
            "whipsaw_proxy",
            "chop_proxy",
        )
        aggregate: Dict[str, float] = {}
        for key in keys:
            aggregate[key] = float(sum(float(payload.get(key, 0.0)) for payload in symbol_metrics.values()) / count)
        aggregate["symbol_count"] = float(len(symbol_metrics))
        return aggregate

    def _derive_targets(self, metrics: Dict[str, float]) -> Dict[str, float]:
        atr_percent = float(metrics.get("atr_percent", 0.0))
        return_vol = float(metrics.get("return_volatility", 0.0))
        adx = float(metrics.get("adx", 0.0))
        trend_persistence = self._clamp(float(metrics.get("trend_persistence", 0.0)), 0.0, 1.0)
        whipsaw_proxy = self._clamp(float(metrics.get("whipsaw_proxy", 0.0)), 0.0, 1.0)
        chop_proxy = self._clamp(float(metrics.get("chop_proxy", 0.0)), 0.0, 1.0)

        atr_norm = self._clamp((atr_percent - 0.0008) / 0.018, 0.0, 1.0)
        adx_norm = self._clamp((adx - 12.0) / 30.0, 0.0, 1.0)
        vol_ratio = return_vol / max(atr_percent * 0.75, 1e-8)
        vol_norm = self._clamp(vol_ratio / 1.6, 0.0, 1.0)

        trend_quality = self._clamp((adx_norm * 0.48) + (trend_persistence * 0.36) + ((1.0 - whipsaw_proxy) * 0.16), 0.0, 1.0)
        chop_pressure = self._clamp((chop_proxy * 0.64) + ((1.0 - trend_persistence) * 0.36), 0.0, 1.0)

        base = self.base_parameters

        # Alpha Reinforcement (2026-02-16): Cap TP in high chop to secure profits
        trend_tp_target = base["trend_tp_r_multiple"] + (trend_quality * 0.95) - (chop_pressure * 0.55)
        if chop_proxy > 0.75:
            trend_tp_target = min(trend_tp_target, 1.5)

        targets: Dict[str, float] = {
            "trend_score_threshold": base["trend_score_threshold"] + (chop_pressure * 0.14) - (trend_quality * 0.10) + ((vol_norm - 0.5) * 0.05),
            "trend_strength_threshold": base["trend_strength_threshold"] + (chop_pressure * 0.16) - (trend_quality * 0.08) + (atr_norm * 0.04),
            "meanrev_max_strength": base["meanrev_max_strength"] + (chop_pressure * 0.20) - (trend_quality * 0.12),
            "breakout_lookback": base["breakout_lookback"] + (chop_pressure * 2.5) + (vol_norm * 1.5) - (trend_quality * 1.5),
            "trend_sl_atr_mult": base["trend_sl_atr_mult"] + (atr_norm * 0.55) + (chop_pressure * 0.25),
            "trend_tp_r_multiple": trend_tp_target,
            "trailing_atr_mult": base["trailing_atr_mult"] + (trend_quality * 0.35) + (atr_norm * 0.20) - (chop_pressure * 0.32),
            "trailing_start_rr": base["trailing_start_rr"] + (trend_quality * 0.45) - (chop_pressure * 0.35),
            "regime_flip_exit_threshold": base["regime_flip_exit_threshold"] + (trend_quality * 0.08) - (chop_pressure * 0.09),
            "sweep_buffer_atr": base["sweep_buffer_atr"] + (chop_proxy * 0.5 * base["sweep_buffer_atr"]),
            "reclaim_window_sec": base["reclaim_window_sec"] * (1.0 - (chop_proxy * 0.5)),
            "displacement_mult": base["displacement_mult"] + (chop_proxy * 0.3),
        }
        bounded: Dict[str, float] = {}
        for name, value in targets.items():
            bounded[name] = self._apply_bound(name, value)
        return bounded

    def _smooth_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        smoothed: Dict[str, float] = {}
        alpha = self.smoothing_alpha
        for name in self.TUNABLE_PARAMETERS:
            target = float(targets[name])
            previous = float(self._overrides.get(name, self.base_parameters[name]))
            value = previous + ((target - previous) * alpha)
            smoothed[name] = self._apply_bound(name, value)
        return smoothed

    def _skip(self, reason: str, now_ts: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._last_skip_reason = reason
        payload: Dict[str, Any] = {
            "updated": False,
            "reason": reason,
            "timestamp": float(now_ts),
        }
        if extra:
            payload.update(extra)
        return payload

    def step(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        now_value = float(now_ts if now_ts is not None else time.time())
        if not self.enabled:
            return self._skip("disabled", now_value)

        if self._last_tuned_at > 0.0:
            elapsed = now_value - self._last_tuned_at
            if elapsed < self.tune_interval_seconds:
                return self._skip(
                    "interval_not_elapsed",
                    now_value,
                    {
                        "seconds_until_next": float(self.tune_interval_seconds - elapsed),
                    },
                )

        required_symbols = self.target_symbols or tuple(self._bars_by_symbol.keys())
        symbol_metrics: Dict[str, Dict[str, float]] = {}
        missing_symbols = []
        for symbol in required_symbols:
            bars = self._bars_by_symbol.get(symbol)
            if bars is None:
                missing_symbols.append(symbol)
                continue
            metrics = self._compute_symbol_metrics(bars)
            if metrics is None:
                missing_symbols.append(symbol)
                continue
            symbol_metrics[symbol] = metrics

        if missing_symbols:
            return self._skip(
                "insufficient_bars",
                now_value,
                {
                    "missing_symbols": list(missing_symbols),
                    "symbols_with_metrics": list(symbol_metrics.keys()),
                },
            )

        aggregate = self._aggregate_metrics(symbol_metrics)
        if aggregate is None:
            return self._skip("metrics_unavailable", now_value)

        targets = self._derive_targets(aggregate)
        smoothed = self._smooth_targets(targets)

        self._overrides = smoothed
        self._last_metrics = aggregate
        self._last_symbol_metrics = symbol_metrics
        self._last_tuned_at = now_value
        self._last_skip_reason = ""
        self._update_count += 1

        return {
            "updated": True,
            "reason": "updated",
            "timestamp": float(now_value),
            "overrides": dict(smoothed),
            "metrics": dict(aggregate),
            "symbol_metrics": dict(symbol_metrics),
        }

    @property
    def overrides(self) -> Dict[str, float]:
        return dict(self._overrides)

    def snapshot(self) -> Dict[str, Any]:
        last_tuned_iso = (
            datetime.fromtimestamp(self._last_tuned_at, tz=timezone.utc).isoformat()
            if self._last_tuned_at > 0.0
            else None
        )
        return {
            "enabled": self.enabled,
            "target_symbols": list(self.target_symbols),
            "tune_interval_seconds": float(self.tune_interval_seconds),
            "lookback_bars": int(self.lookback_bars),
            "min_bars": int(self.min_bars),
            "smoothing_alpha": float(self.smoothing_alpha),
            "parameter_bounds": {
                name: {"min": float(bounds[0]), "max": float(bounds[1])}
                for name, bounds in self.bounds.items()
            },
            "base_parameters": dict(self.base_parameters),
            "overrides": dict(self._overrides),
            "metrics": dict(self._last_metrics),
            "symbol_metrics": dict(self._last_symbol_metrics),
            "last_tuned_at": float(self._last_tuned_at),
            "last_tuned_at_utc": last_tuned_iso,
            "last_skip_reason": self._last_skip_reason,
            "update_count": int(self._update_count),
        }
