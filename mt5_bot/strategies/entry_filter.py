from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


SPREAD_TOO_WIDE = "spread_too_wide"
FEE_ADJUSTED_RR_TOO_LOW = "fee_adjusted_rr_too_low"
SIGNAL_SCORE_TOO_LOW = "signal_score_too_low"
LATE_ENTRY = "late_entry"
MIN_LOT_RISK_EXCEEDS_HARD_MAX = "min_lot_risk_exceeds_hard_max"
DAILY_BLEED_GUARD_ACTIVE = "daily_bleed_guard_active"
CONSECUTIVE_LOSS_COOLDOWN = "consecutive_loss_cooldown"
MAX_OPEN_POSITIONS_REACHED = "max_open_positions_reached"
BROKER_STOP_LEVEL_VIOLATION = "broker_stop_level_violation"
DATA_GAP = "data_gap"
LIVE_GATE_CLOSED = "live_gate_closed"
PAPER_ONLY_MODE = "paper_only_mode"
OOS_NOT_READY = "oos_not_ready"
LLM_VETO_FORBIDDEN_ATTEMPT = "llm_veto_forbidden_attempt"


BLOCK_REASONS = {
    SPREAD_TOO_WIDE,
    FEE_ADJUSTED_RR_TOO_LOW,
    SIGNAL_SCORE_TOO_LOW,
    LATE_ENTRY,
    MIN_LOT_RISK_EXCEEDS_HARD_MAX,
    DAILY_BLEED_GUARD_ACTIVE,
    CONSECUTIVE_LOSS_COOLDOWN,
    MAX_OPEN_POSITIONS_REACHED,
    BROKER_STOP_LEVEL_VIOLATION,
    DATA_GAP,
    LIVE_GATE_CLOSED,
    PAPER_ONLY_MODE,
    OOS_NOT_READY,
    LLM_VETO_FORBIDDEN_ATTEMPT,
}


@dataclass(frozen=True)
class EntryFilterDecision:
    allow: bool
    reason: str
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    opportunity: Any = None

    @property
    def passed(self) -> bool:
        return bool(self.allow)

    @property
    def blocked(self) -> bool:
        return not self.allow

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow": bool(self.allow),
            "passed": bool(self.allow),
            "reason": self.reason,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


class FeeAwareEntryFilter:
    """Pure-Python pre-trade gate for dict or TradeOpportunity-like objects."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        nested = cfg.get("fee_aware_entry_filter")
        if isinstance(nested, dict):
            cfg = dict(nested)

        self.enabled = _bool_cfg(cfg.get("enabled"), True)
        self.min_signal_score = _float_cfg(cfg.get("min_signal_score"), 0.0)
        self.min_reward_to_net_risk_ratio = _float_cfg(
            cfg.get("min_reward_to_net_risk_ratio"),
            _float_cfg(cfg.get("min_fee_adjusted_rr"), 1.0),
        )
        self.hard_max_net_loss_usd = _float_cfg(cfg.get("hard_max_net_loss_usd"), math.inf)
        self.max_spread = _float_cfg(cfg.get("max_spread"), math.inf)
        self.max_spread_points = _float_cfg(cfg.get("max_spread_points"), self.max_spread)
        self.max_signal_age_seconds = _float_cfg(cfg.get("max_signal_age_seconds"), math.inf)
        self.max_bars_late = _float_cfg(cfg.get("max_bars_late"), math.inf)
        self.max_open_positions = _int_cfg(cfg.get("max_open_positions"), 999999)
        self.broker_stop_level_points = _float_cfg(cfg.get("broker_stop_level_points"), 0.0)
        self.max_data_gap_seconds = _float_cfg(cfg.get("max_data_gap_seconds"), math.inf)
        self.paper_only_mode = _bool_cfg(cfg.get("paper_only_mode"), False)
        self.require_oos_ready = _bool_cfg(cfg.get("require_oos_ready"), False)
        self.spread_limit_by_symbol = _float_map(cfg.get("spread_limit_by_symbol"))

    def evaluate(self, opportunity: Any, context: Optional[Dict[str, Any]] = None) -> EntryFilterDecision:
        if not self.enabled:
            return EntryFilterDecision(True, "ok", [], {"enabled": False}, opportunity)

        ctx = dict(context or {})
        reasons: List[str] = []
        metrics: Dict[str, Any] = {}

        symbol = str(_first(opportunity, ("symbol",), "") or "").strip().upper()
        signal_score = _float_cfg(_first(opportunity, ("signal_score", "score", "confidence")), 0.0)
        min_signal_score = _float_cfg(ctx.get("min_signal_score"), self.min_signal_score)
        metrics["signal_score"] = signal_score
        metrics["min_signal_score"] = min_signal_score
        if signal_score < min_signal_score:
            reasons.append(SIGNAL_SCORE_TOO_LOW)

        fee_adjusted_rr = _float_cfg(
            _first(opportunity, ("fee_adjusted_rr", "reward_to_net_risk_ratio", "net_rr", "expected_rr")),
            0.0,
        )
        min_rr = _float_cfg(ctx.get("min_reward_to_net_risk_ratio"), self.min_reward_to_net_risk_ratio)
        metrics["fee_adjusted_rr"] = fee_adjusted_rr
        metrics["min_reward_to_net_risk_ratio"] = min_rr
        if fee_adjusted_rr < min_rr:
            reasons.append(FEE_ADJUSTED_RR_TOO_LOW)

        spread = _float_cfg(_first(opportunity, ("spread", "spread_points")), 0.0)
        spread_limit = self.spread_limit_by_symbol.get(symbol, self.max_spread_points)
        spread_limit = _float_cfg(ctx.get("max_spread_points"), spread_limit)
        metrics["spread"] = spread
        metrics["max_spread_points"] = spread_limit
        if spread > spread_limit:
            reasons.append(SPREAD_TOO_WIDE)

        estimated_loss = _optional_float(
            _first(opportunity, ("estimated_sl_net_loss", "estimated_net_loss_usd", "net_loss_at_sl"))
        )
        min_lot_loss = _optional_float(
            _first(opportunity, ("min_lot_estimated_sl_net_loss", "min_lot_net_loss", "estimated_min_lot_net_loss"))
        )
        hard_max = _float_cfg(ctx.get("hard_max_net_loss_usd"), self.hard_max_net_loss_usd)
        metrics["hard_max_net_loss_usd"] = hard_max
        if estimated_loss is not None:
            metrics["estimated_sl_net_loss"] = estimated_loss
        if min_lot_loss is not None:
            metrics["min_lot_estimated_sl_net_loss"] = min_lot_loss
        if (estimated_loss is not None and estimated_loss > hard_max) or (
            min_lot_loss is not None and min_lot_loss > hard_max
        ):
            reasons.append(MIN_LOT_RISK_EXCEEDS_HARD_MAX)

        if not self._position_size_feasible(opportunity):
            reasons.append(MIN_LOT_RISK_EXCEEDS_HARD_MAX)

        if self._is_late(opportunity, ctx):
            reasons.append(LATE_ENTRY)

        daily_bleed_reason = self._daily_bleed_reason(opportunity, ctx)
        if daily_bleed_reason:
            reasons.append(daily_bleed_reason)

        open_positions = _int_cfg(ctx.get("open_positions_count", _first(opportunity, ("open_positions_count",))), 0)
        max_open = _int_cfg(ctx.get("max_open_positions"), self.max_open_positions)
        metrics["open_positions_count"] = open_positions
        metrics["max_open_positions"] = max_open
        if open_positions >= max_open:
            reasons.append(MAX_OPEN_POSITIONS_REACHED)

        stop_distance_points = _optional_float(_first(opportunity, ("stop_distance_points", "sl_distance_points")))
        stop_level = _float_cfg(ctx.get("broker_stop_level_points"), self.broker_stop_level_points)
        if stop_distance_points is not None:
            metrics["stop_distance_points"] = stop_distance_points
            metrics["broker_stop_level_points"] = stop_level
            if stop_distance_points < stop_level:
                reasons.append(BROKER_STOP_LEVEL_VIOLATION)

        if _bool_cfg(_first(opportunity, ("data_gap", "has_data_gap"), ctx.get("data_gap")), False):
            reasons.append(DATA_GAP)
        gap_age = _optional_float(_first(opportunity, ("data_gap_seconds",), ctx.get("data_gap_seconds")))
        max_gap = _float_cfg(ctx.get("max_data_gap_seconds"), self.max_data_gap_seconds)
        if gap_age is not None:
            metrics["data_gap_seconds"] = gap_age
            metrics["max_data_gap_seconds"] = max_gap
            if gap_age > max_gap:
                reasons.append(DATA_GAP)

        if not _bool_cfg(ctx.get("live_gate_open", _first(opportunity, ("live_gate_open",), True)), True):
            reasons.append(LIVE_GATE_CLOSED)
        if _bool_cfg(ctx.get("paper_only_mode", _first(opportunity, ("paper_only_mode",), self.paper_only_mode)), False):
            reasons.append(PAPER_ONLY_MODE)
        if self.require_oos_ready and not _bool_cfg(ctx.get("oos_ready", _first(opportunity, ("oos_ready",), False)), False):
            reasons.append(OOS_NOT_READY)
        if _bool_cfg(ctx.get("oos_not_ready", _first(opportunity, ("oos_not_ready",), False)), False):
            reasons.append(OOS_NOT_READY)
        if _bool_cfg(
            ctx.get("llm_veto_forbidden_attempt", _first(opportunity, ("llm_veto_forbidden_attempt",), False)),
            False,
        ):
            reasons.append(LLM_VETO_FORBIDDEN_ATTEMPT)

        unique_reasons = _dedupe(reasons)
        allow = not unique_reasons
        return EntryFilterDecision(
            allow=allow,
            reason="ok" if allow else unique_reasons[0],
            reasons=unique_reasons,
            metrics=metrics,
            opportunity=opportunity,
        )

    def filter(self, opportunity: Any, context: Optional[Dict[str, Any]] = None) -> EntryFilterDecision:
        return self.evaluate(opportunity, context)

    def __call__(self, opportunity: Any, context: Optional[Dict[str, Any]] = None) -> EntryFilterDecision:
        return self.evaluate(opportunity, context)

    def _position_size_feasible(self, opportunity: Any) -> bool:
        explicit = _first(opportunity, ("position_size_feasible", "size_feasible"))
        if explicit is not None:
            return _bool_cfg(explicit, True)
        requested = _optional_float(_first(opportunity, ("lot", "volume", "requested_lot", "recommended_lot")))
        if requested is None:
            return True
        min_lot = _optional_float(_first(opportunity, ("min_lot", "volume_min")))
        max_lot = _optional_float(_first(opportunity, ("max_lot", "volume_max")))
        if min_lot is not None and requested < min_lot:
            return False
        if max_lot is not None and requested > max_lot:
            return False
        return requested > 0.0

    def _is_late(self, opportunity: Any, context: Dict[str, Any]) -> bool:
        if _bool_cfg(_first(opportunity, ("late_entry", "is_late"), context.get("late_entry")), False):
            return True
        age_seconds = _optional_float(_first(opportunity, ("signal_age_seconds", "age_seconds"), context.get("signal_age_seconds")))
        if age_seconds is not None and age_seconds > _float_cfg(context.get("max_signal_age_seconds"), self.max_signal_age_seconds):
            return True
        bars_late = _optional_float(_first(opportunity, ("bars_late",), context.get("bars_late")))
        return bool(bars_late is not None and bars_late > _float_cfg(context.get("max_bars_late"), self.max_bars_late))

    def _daily_bleed_reason(self, opportunity: Any, context: Dict[str, Any]) -> Optional[str]:
        if _bool_cfg(context.get("consecutive_loss_cooldown", _first(opportunity, ("consecutive_loss_cooldown",), False)), False):
            return CONSECUTIVE_LOSS_COOLDOWN
        if _bool_cfg(context.get("daily_bleed_guard_active", _first(opportunity, ("daily_bleed_guard_active",), False)), False):
            return DAILY_BLEED_GUARD_ACTIVE

        guard = context.get("daily_bleed_guard")
        if guard is None:
            return None
        symbol = str(_first(opportunity, ("symbol",), context.get("symbol", "")) or "")
        direction = _first(opportunity, ("direction", "side", "action"), context.get("direction"))
        setup_key = _first(opportunity, ("setup_key", "strategy"), context.get("setup_key"))
        now_ts = context.get("now_ts")
        block = None
        if hasattr(guard, "entry_block"):
            block = guard.entry_block(symbol=symbol, now_ts=now_ts, direction=direction, setup_key=setup_key)
            raw_reason = getattr(block, "reason", block)
        elif hasattr(guard, "should_block_entry"):
            raw_reason = guard.should_block_entry(symbol=symbol, now_ts=now_ts, direction=direction, setup_key=setup_key)
        else:
            raw_reason = None
        if not raw_reason:
            return None
        text = str(raw_reason).upper()
        if "CONSECUTIVE" in text or "COOLDOWN" in text:
            return CONSECUTIVE_LOSS_COOLDOWN
        return DAILY_BLEED_GUARD_ACTIVE


def _first(source: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = _get_value(source, name)
        if value is not None:
            return value
    return default


def _get_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        if name in source:
            return source.get(name)
        metadata = source.get("metadata")
        if isinstance(metadata, dict) and name in metadata:
            return metadata.get(name)
        return None
    if hasattr(source, name):
        return getattr(source, name)
    metadata = getattr(source, "metadata", None)
    if isinstance(metadata, dict) and name in metadata:
        return metadata.get(name)
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    parsed = _float_cfg(value, math.nan)
    return parsed if math.isfinite(parsed) else None


def _float_cfg(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return parsed if math.isfinite(parsed) or math.isinf(parsed) else float(default)


def _int_cfg(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool_cfg(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _float_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for key, raw in value.items():
        symbol = str(key or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = _float_cfg(raw, math.inf)
    return out


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
