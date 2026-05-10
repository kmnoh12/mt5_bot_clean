from __future__ import annotations

import math
from typing import Any, Dict, Optional

from core.models import SymbolConstraints


class CostEdgeGuard:
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = True
        self.min_edge_to_cost_ratio_default = 3.0
        self.min_edge_to_cost_ratio_by_symbol: Dict[str, float] = {"GOLD": 2.5}
        self.spread_sample_bars = 120
        self.use_recent_deal_cost_stats = True
        self._recent_cost_stats_by_symbol: Dict[str, Dict[str, float]] = {}

        self.update_config(config)
        self._restore_snapshot(snapshot or {})

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(out)

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.min_edge_to_cost_ratio_default = max(
            0.1, self._to_float(cfg.get("min_edge_to_cost_ratio_default", 3.0), 3.0)
        )
        raw = cfg.get("min_edge_to_cost_ratio_by_symbol", {"GOLD": 2.5})
        parsed: Dict[str, float] = {}
        if isinstance(raw, dict):
            for symbol, value in raw.items():
                key = str(symbol or "").strip().upper()
                if not key:
                    continue
                parsed[key] = max(0.1, self._to_float(value, 3.0))
        self.min_edge_to_cost_ratio_by_symbol = parsed or {"GOLD": 2.5}
        self.spread_sample_bars = max(10, int(self._to_float(cfg.get("spread_sample_bars", 120), 120)))
        self.use_recent_deal_cost_stats = bool(cfg.get("use_recent_deal_cost_stats", True))

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        raw = snapshot.get("recent_cost_stats_by_symbol")
        if not isinstance(raw, dict):
            return
        parsed: Dict[str, Dict[str, float]] = {}
        for symbol, values in raw.items():
            if not isinstance(values, dict):
                continue
            key = str(symbol or "").strip().upper()
            if not key:
                continue
            parsed[key] = {
                "avg_cost_usd": max(0.0, self._to_float(values.get("avg_cost_usd"), 0.0)),
                "samples": max(0.0, self._to_float(values.get("samples"), 0.0)),
            }
        self._recent_cost_stats_by_symbol = parsed

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "min_edge_to_cost_ratio_default": float(self.min_edge_to_cost_ratio_default),
            "min_edge_to_cost_ratio_by_symbol": dict(self.min_edge_to_cost_ratio_by_symbol),
            "spread_sample_bars": int(self.spread_sample_bars),
            "use_recent_deal_cost_stats": bool(self.use_recent_deal_cost_stats),
            "recent_cost_stats_by_symbol": dict(self._recent_cost_stats_by_symbol),
        }

    def record_cost(self, symbol: str, realized_cost_usd: Optional[float]) -> None:
        if not self.use_recent_deal_cost_stats:
            return
        if realized_cost_usd is None:
            return
        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            return
        value = abs(self._to_float(realized_cost_usd, 0.0))
        current = self._recent_cost_stats_by_symbol.get(symbol_key, {"avg_cost_usd": 0.0, "samples": 0.0})
        old_avg = self._to_float(current.get("avg_cost_usd"), 0.0)
        old_n = max(0.0, self._to_float(current.get("samples"), 0.0))
        n = min(500.0, old_n + 1.0)
        alpha = 1.0 / min(50.0, n)
        avg = ((1.0 - alpha) * old_avg) + (alpha * value)
        self._recent_cost_stats_by_symbol[symbol_key] = {"avg_cost_usd": float(avg), "samples": float(n)}

    def evaluate_entry(
        self,
        *,
        symbol: str,
        decision_metadata: Dict[str, Any],
        requested_volume: float,
        constraints: Optional[SymbolConstraints],
    ) -> Dict[str, Any]:
        symbol_key = str(symbol or "").strip().upper()
        if not self.enabled:
            return {"allow": True, "reason": "COST_EDGE_DISABLED", "ratio": float("inf")}

        metadata = dict(decision_metadata or {})
        stop_distance = abs(self._to_float(metadata.get("risk_per_unit"), 0.0))
        expected_rr = self._to_float(metadata.get("expected_rr"), 1.5)
        if expected_rr <= 0:
            expected_rr = 1.5
        contract_size = float(constraints.contract_size) if constraints is not None else 1.0
        volume = max(0.001, self._to_float(requested_volume, 0.01))
        expected_edge = stop_distance * expected_rr * volume * max(1e-9, float(contract_size))

        stats = self._recent_cost_stats_by_symbol.get(symbol_key, {})
        learned_cost = self._to_float(stats.get("avg_cost_usd"), 0.0)
        fallback_cost = max(0.25, stop_distance * volume * max(1e-9, float(contract_size)) * 0.05)
        expected_cost = max(fallback_cost, learned_cost)
        ratio = float(expected_edge / max(expected_cost, 1e-9))
        threshold = float(self.min_edge_to_cost_ratio_by_symbol.get(symbol_key, self.min_edge_to_cost_ratio_default))
        allow = bool(ratio >= threshold)
        return {
            "allow": allow,
            "reason": "OK" if allow else "EDGE_TOO_LOW",
            "ratio": float(ratio),
            "threshold": float(threshold),
            "expected_edge_usd": float(expected_edge),
            "expected_cost_usd": float(expected_cost),
        }
