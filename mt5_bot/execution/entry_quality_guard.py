from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from core.models import DecisionAction


class EntryQualityGuard:
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = True
        self.trend_only_symbols = {"BTCUSD", "ETHUSD"}
        self.min_score = 0.62
        self.min_score_risk_off = 0.68
        self.min_score_risk_on = 0.58
        self.lookback_closed_trades = 200
        self.min_winner_pnl_usd = 5.0
        self.max_churn_abs_pnl_usd = 2.0
        self.max_churn_hold_seconds = 300.0

        self._closed_trades: List[Dict[str, Any]] = []
        self._open_context_by_ticket: Dict[str, Dict[str, Any]] = {}
        self._winner_profile: Dict[str, Dict[str, float]] = {}
        self._last_score_by_symbol: Dict[str, float] = {}

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

    @staticmethod
    def _ticket_key(ticket: Any) -> str:
        try:
            return str(int(ticket))
        except Exception:
            return str(ticket)

    @staticmethod
    def _is_trade_action(action: DecisionAction) -> bool:
        return action in {DecisionAction.BUY, DecisionAction.SELL}

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        raw_symbols = cfg.get("trend_only_symbols", ["BTCUSD", "ETHUSD"])
        parsed = set()
        if isinstance(raw_symbols, (list, tuple, set)):
            for item in raw_symbols:
                text = str(item or "").strip().upper()
                if text:
                    parsed.add(text)
        self.trend_only_symbols = parsed or {"BTCUSD", "ETHUSD"}
        self.min_score = self._clamp_01(self._to_float(cfg.get("min_score", 0.62), 0.62))
        self.min_score_risk_off = self._clamp_01(self._to_float(cfg.get("min_score_risk_off", 0.68), 0.68))
        self.min_score_risk_on = self._clamp_01(self._to_float(cfg.get("min_score_risk_on", 0.58), 0.58))
        self.lookback_closed_trades = max(20, int(self._to_float(cfg.get("lookback_closed_trades", 200), 200)))
        self.min_winner_pnl_usd = max(0.0, self._to_float(cfg.get("min_winner_pnl_usd", 5.0), 5.0))
        self.max_churn_abs_pnl_usd = max(0.0, self._to_float(cfg.get("max_churn_abs_pnl_usd", 2.0), 2.0))
        self.max_churn_hold_seconds = max(1.0, self._to_float(cfg.get("max_churn_hold_seconds", 300.0), 300.0))

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        raw_closed = snapshot.get("closed_trades")
        if isinstance(raw_closed, list):
            cleaned = []
            for item in raw_closed:
                if not isinstance(item, dict):
                    continue
                cleaned.append(dict(item))
            self._closed_trades = cleaned[-self.lookback_closed_trades :]

        raw_open = snapshot.get("open_context_by_ticket")
        if isinstance(raw_open, dict):
            cleaned_open = {}
            for key, value in raw_open.items():
                if isinstance(value, dict):
                    cleaned_open[str(key)] = dict(value)
            self._open_context_by_ticket = cleaned_open

        raw_profile = snapshot.get("winner_profile")
        if isinstance(raw_profile, dict):
            profile: Dict[str, Dict[str, float]] = {}
            for symbol, metrics in raw_profile.items():
                if not isinstance(metrics, dict):
                    continue
                profile[str(symbol).upper()] = {
                    "trend_strength": self._to_float(metrics.get("trend_strength"), 0.0),
                    "adx_norm": self._to_float(metrics.get("adx_norm"), 0.0),
                    "ema_align": self._to_float(metrics.get("ema_align"), 0.0),
                    "m5_align": self._to_float(metrics.get("m5_align"), 0.0),
                    "samples": self._to_float(metrics.get("samples"), 0.0),
                }
            self._winner_profile = profile

        raw_scores = snapshot.get("last_score_by_symbol")
        if isinstance(raw_scores, dict):
            self._last_score_by_symbol = {
                str(symbol).upper(): self._to_float(score, 0.0) for symbol, score in raw_scores.items()
            }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "trend_only_symbols": sorted(self.trend_only_symbols),
            "min_score": float(self.min_score),
            "min_score_risk_off": float(self.min_score_risk_off),
            "min_score_risk_on": float(self.min_score_risk_on),
            "lookback_closed_trades": int(self.lookback_closed_trades),
            "min_winner_pnl_usd": float(self.min_winner_pnl_usd),
            "max_churn_abs_pnl_usd": float(self.max_churn_abs_pnl_usd),
            "max_churn_hold_seconds": float(self.max_churn_hold_seconds),
            "closed_trades": list(self._closed_trades[-self.lookback_closed_trades :]),
            "open_context_by_ticket": dict(self._open_context_by_ticket),
            "winner_profile": dict(self._winner_profile),
            "last_score_by_symbol": dict(self._last_score_by_symbol),
        }

    def record_entry_context(self, ticket: Any, symbol: str, metadata: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        key = self._ticket_key(ticket)
        snapshot = dict(metadata or {})
        indicator = snapshot.get("indicator_snapshot")
        if isinstance(indicator, dict):
            snapshot["indicator_snapshot"] = dict(indicator)
        snapshot["symbol"] = str(symbol or "").strip().upper()
        self._open_context_by_ticket[key] = snapshot

    def _risk_mode_threshold(self) -> Tuple[float, str]:
        sample = self._closed_trades[-20:]
        if not sample:
            return float(self.min_score), "neutral"
        pnl_values = [self._to_float(item.get("pnl"), 0.0) for item in sample]
        total = float(sum(pnl_values))
        churn_count = 0
        for item in sample:
            pnl = abs(self._to_float(item.get("pnl"), 0.0))
            hold = self._to_float(item.get("hold_seconds"), 999999.0)
            if pnl <= self.max_churn_abs_pnl_usd and hold <= self.max_churn_hold_seconds:
                churn_count += 1
        if total < 0:
            return float(self.min_score_risk_off), "risk_off"
        if total > 0 and churn_count == 0:
            return float(self.min_score_risk_on), "risk_on"
        return float(self.min_score), "neutral"

    def evaluate_entry(
        self,
        symbol: str,
        decision_action: DecisionAction,
        decision_metadata: Dict[str, Any],
        m5_aligned: bool,
    ) -> Dict[str, Any]:
        symbol_key = str(symbol or "").strip().upper()
        metadata = dict(decision_metadata or {})
        style = str(metadata.get("entry_style", "") or "").strip().lower()

        if not self.enabled or not self._is_trade_action(decision_action):
            return {"allow": True, "score": 1.0, "threshold": 0.0, "reason": "ENTRY_QUALITY_DISABLED"}

        if symbol_key in self.trend_only_symbols and style == "mean_reversion":
            return {
                "allow": False,
                "score": 0.0,
                "threshold": 1.0,
                "reason": "ENTRY_QUALITY_BLOCK",
                "detail": "TREND_ONLY_SYMBOL_BLOCKED_MEANREV",
            }

        indicator = metadata.get("indicator_snapshot")
        indicator = dict(indicator) if isinstance(indicator, dict) else {}
        trend_strength = self._clamp_01(self._to_float(indicator.get("trend_strength"), 0.0))
        adx_norm = self._clamp_01(self._to_float(indicator.get("adx_norm"), 0.0))
        ema_gap_atr = self._to_float(indicator.get("ema_gap_atr"), 0.0)
        ema_align = self._clamp_01(abs(ema_gap_atr))
        direction_ok = (decision_action == DecisionAction.BUY and ema_gap_atr >= 0.0) or (
            decision_action == DecisionAction.SELL and ema_gap_atr <= 0.0
        )
        direction_bonus = 1.0 if direction_ok else 0.0
        m5_score = 1.0 if m5_aligned else 0.0

        score = (
            (trend_strength * 0.35)
            + (adx_norm * 0.25)
            + ((ema_align * direction_bonus) * 0.20)
            + (m5_score * 0.20)
        )
        threshold, mode = self._risk_mode_threshold()
        self._last_score_by_symbol[symbol_key] = float(score)
        allow = bool(score >= threshold)
        return {
            "allow": allow,
            "score": float(score),
            "threshold": float(threshold),
            "reason": "OK" if allow else "ENTRY_QUALITY_BLOCK",
            "risk_mode": mode,
            "features": {
                "trend_strength": trend_strength,
                "adx_norm": adx_norm,
                "ema_align": ema_align,
                "direction_ok": direction_ok,
                "m5_align": m5_score,
            },
        }

    def record_closed_trade(
        self,
        *,
        ticket: Any,
        symbol: str,
        pnl: Optional[float],
        hold_seconds: Optional[float],
    ) -> Dict[str, Any]:
        symbol_key = str(symbol or "").strip().upper()
        ticket_key = self._ticket_key(ticket)
        context = self._open_context_by_ticket.pop(ticket_key, {})
        indicator = context.get("indicator_snapshot")
        indicator = dict(indicator) if isinstance(indicator, dict) else {}
        entry_style = str(context.get("entry_style", "") or "")

        record = {
            "ticket": ticket_key,
            "symbol": symbol_key,
            "pnl": self._to_float(pnl, 0.0) if pnl is not None else None,
            "hold_seconds": self._to_float(hold_seconds, 0.0) if hold_seconds is not None else None,
            "entry_style": entry_style,
            "trend_strength": self._to_float(indicator.get("trend_strength"), 0.0),
            "adx_norm": self._to_float(indicator.get("adx_norm"), 0.0),
            "ema_align": abs(self._to_float(indicator.get("ema_gap_atr"), 0.0)),
            "m5_align": 1.0 if bool(context.get("m5_align", False)) else 0.0,
        }
        self._closed_trades.append(record)
        self._closed_trades = self._closed_trades[-self.lookback_closed_trades :]

        winner_samples: Dict[str, List[Dict[str, Any]]] = {}
        for item in self._closed_trades:
            pnl_value = item.get("pnl")
            if pnl_value is None:
                continue
            if self._to_float(pnl_value, 0.0) < self.min_winner_pnl_usd:
                continue
            key = str(item.get("symbol", "")).upper()
            winner_samples.setdefault(key, []).append(item)

        updated = False
        profile: Dict[str, Dict[str, float]] = {}
        for key, samples in winner_samples.items():
            if not samples:
                continue
            count = float(len(samples))
            trend_strength = sum(self._to_float(x.get("trend_strength"), 0.0) for x in samples) / count
            adx_norm = sum(self._to_float(x.get("adx_norm"), 0.0) for x in samples) / count
            ema_align = sum(self._to_float(x.get("ema_align"), 0.0) for x in samples) / count
            m5_align = sum(self._to_float(x.get("m5_align"), 0.0) for x in samples) / count
            profile[key] = {
                "trend_strength": float(trend_strength),
                "adx_norm": float(adx_norm),
                "ema_align": float(ema_align),
                "m5_align": float(m5_align),
                "samples": float(count),
            }
        if profile != self._winner_profile:
            updated = True
            self._winner_profile = profile

        return {
            "updated": updated,
            "winner_profile": dict(self._winner_profile),
            "closed_count": len(self._closed_trades),
            "last_symbol": symbol_key,
        }
