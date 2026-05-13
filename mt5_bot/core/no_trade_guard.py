from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


WARNING_NO_TRADE_24H = "no_trade_24h_warning"
FAILURE_NO_TRADE_2D = "no_trade_2d_failure"
ZERO_TRADE_NOT_SUCCESS = "zero_trade_not_success"


@dataclass(frozen=True)
class OpportunityRecord:
    opportunity_id: str
    symbol: str = ""
    score: float = 0.0
    fee_adjusted_rr: float = 0.0
    reason: str = ""
    timestamp: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "symbol": self.symbol,
            "score": float(self.score),
            "fee_adjusted_rr": float(self.fee_adjusted_rr),
            "reason": self.reason,
            "timestamp": self.timestamp,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class NoTradeBiasSnapshot:
    raw_signal_count: int
    scored_signal_count: int
    eligible_signal_count: int
    executed_trade_count: int
    no_trade_hours: float
    no_trade_days_count: int
    block_rate_by_reason: Dict[str, float]
    block_count_by_reason: Dict[str, int]
    top_rejected_opportunities: List[Dict[str, Any]]
    best_missed_opportunity: Optional[Dict[str, Any]]
    status: str
    warnings: List[str]
    failures: List[str]
    zero_trade_success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_signal_count": int(self.raw_signal_count),
            "scored_signal_count": int(self.scored_signal_count),
            "eligible_signal_count": int(self.eligible_signal_count),
            "executed_trade_count": int(self.executed_trade_count),
            "no_trade_hours": float(self.no_trade_hours),
            "no_trade_days_count": int(self.no_trade_days_count),
            "block_rate_by_reason": dict(self.block_rate_by_reason),
            "block_count_by_reason": dict(self.block_count_by_reason),
            "top_rejected_opportunities": list(self.top_rejected_opportunities),
            "best_missed_opportunity": dict(self.best_missed_opportunity) if self.best_missed_opportunity else None,
            "status": self.status,
            "warnings": list(self.warnings),
            "failures": list(self.failures),
            "zero_trade_success": bool(self.zero_trade_success),
        }


class NoTradeBiasGuard:
    def __init__(self, config: Optional[Dict[str, Any]] = None, snapshot: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        nested = cfg.get("no_trade_bias_guard")
        if isinstance(nested, dict):
            cfg = dict(nested)
        self.warning_no_trade_hours = _float_cfg(cfg.get("warning_no_trade_hours"), 24.0)
        self.failure_no_trade_hours = _float_cfg(cfg.get("failure_no_trade_hours"), 48.0)
        self.top_rejected_limit = _int_cfg(cfg.get("top_rejected_limit"), 5)

        self.raw_signal_count = 0
        self.scored_signal_count = 0
        self.eligible_signal_count = 0
        self.executed_trade_count = 0
        self._started_ts: Optional[float] = None
        self._last_trade_ts: Optional[float] = None
        self._block_count_by_reason: Dict[str, int] = {}
        self._rejected: List[OpportunityRecord] = []
        self._missed: List[OpportunityRecord] = []
        if snapshot:
            self._restore(snapshot)

    def record_raw_signal(self, opportunity: Any = None, now_ts: Optional[float] = None) -> None:
        self._mark_started(now_ts)
        self.raw_signal_count += 1

    def record_scored_signal(self, opportunity: Any = None, now_ts: Optional[float] = None) -> None:
        self._mark_started(now_ts)
        self.scored_signal_count += 1

    def record_eligible_signal(self, opportunity: Any = None, now_ts: Optional[float] = None) -> None:
        self._mark_started(now_ts)
        self.eligible_signal_count += 1
        if opportunity is not None:
            self._missed.append(_opportunity_record(opportunity, "", now_ts))

    def record_executed_trade(self, opportunity: Any = None, now_ts: Optional[float] = None) -> None:
        ts = _now_ts(now_ts)
        self._mark_started(ts)
        self.executed_trade_count += 1
        self._last_trade_ts = ts

    def record_rejection(self, opportunity: Any, reason: Any = None, now_ts: Optional[float] = None) -> None:
        self._mark_started(now_ts)
        reasons = _reason_list(reason)
        if not reasons:
            reasons = _reason_list(_get_value(opportunity, "reason") or _get_value(opportunity, "reasons"))
        for item in reasons:
            self._block_count_by_reason[item] = self._block_count_by_reason.get(item, 0) + 1
        first_reason = reasons[0] if reasons else "unknown"
        self._rejected.append(_opportunity_record(opportunity, first_reason, now_ts))

    def record_filter_decision(self, opportunity: Any, decision: Any, now_ts: Optional[float] = None) -> None:
        allow = bool(_get_value(decision, "allow"))
        if allow:
            self.record_eligible_signal(opportunity, now_ts)
            return
        reasons = _get_value(decision, "reasons") or _get_value(decision, "reason")
        self.record_rejection(opportunity, reasons, now_ts)

    def snapshot(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        ts = _now_ts(now_ts)
        no_trade_hours = self.no_trade_hours(ts)
        warnings: List[str] = []
        failures: List[str] = []
        if self.executed_trade_count == 0:
            failures.append(ZERO_TRADE_NOT_SUCCESS)
        if no_trade_hours >= self.warning_no_trade_hours:
            warnings.append(WARNING_NO_TRADE_24H)
        if no_trade_hours >= self.failure_no_trade_hours:
            failures.append(FAILURE_NO_TRADE_2D)
        block_total = sum(self._block_count_by_reason.values())
        block_rate = {
            reason: count / block_total if block_total else 0.0
            for reason, count in sorted(self._block_count_by_reason.items())
        }
        snapshot = NoTradeBiasSnapshot(
            raw_signal_count=self.raw_signal_count,
            scored_signal_count=self.scored_signal_count,
            eligible_signal_count=self.eligible_signal_count,
            executed_trade_count=self.executed_trade_count,
            no_trade_hours=no_trade_hours,
            no_trade_days_count=int(no_trade_hours // 24.0),
            block_rate_by_reason=block_rate,
            block_count_by_reason=dict(sorted(self._block_count_by_reason.items())),
            top_rejected_opportunities=[item.to_dict() for item in self.top_rejected_opportunities()],
            best_missed_opportunity=self.best_missed_opportunity(),
            status="failure" if failures else ("warning" if warnings else "ok"),
            warnings=warnings,
            failures=failures,
            zero_trade_success=False if self.executed_trade_count == 0 else True,
        )
        return snapshot.to_dict()

    def no_trade_hours(self, now_ts: Optional[float] = None) -> float:
        ts = _now_ts(now_ts)
        anchor = self._last_trade_ts if self._last_trade_ts is not None else self._started_ts
        if anchor is None:
            return 0.0
        return max(0.0, (ts - anchor) / 3600.0)

    def top_rejected_opportunities(self) -> List[OpportunityRecord]:
        return sorted(self._rejected, key=lambda item: (item.score, item.fee_adjusted_rr), reverse=True)[
            : max(0, self.top_rejected_limit)
        ]

    def best_missed_opportunity(self) -> Optional[Dict[str, Any]]:
        candidates = self._missed + self._rejected
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.score, item.fee_adjusted_rr)).to_dict()

    def _mark_started(self, now_ts: Optional[float]) -> None:
        if self._started_ts is None:
            self._started_ts = _now_ts(now_ts)

    def _restore(self, snapshot: Dict[str, Any]) -> None:
        self.raw_signal_count = _int_cfg(snapshot.get("raw_signal_count"), 0)
        self.scored_signal_count = _int_cfg(snapshot.get("scored_signal_count"), 0)
        self.eligible_signal_count = _int_cfg(snapshot.get("eligible_signal_count"), 0)
        self.executed_trade_count = _int_cfg(snapshot.get("executed_trade_count"), 0)
        self._started_ts = _optional_float(snapshot.get("started_ts"))
        self._last_trade_ts = _optional_float(snapshot.get("last_trade_ts"))
        counts = snapshot.get("block_count_by_reason")
        if isinstance(counts, dict):
            self._block_count_by_reason = {str(key): _int_cfg(value, 0) for key, value in counts.items()}


def _opportunity_record(opportunity: Any, reason: str, now_ts: Optional[float]) -> OpportunityRecord:
    raw = dict(opportunity) if isinstance(opportunity, dict) else {}
    return OpportunityRecord(
        opportunity_id=str(_get_value(opportunity, "id") or _get_value(opportunity, "opportunity_id") or ""),
        symbol=str(_get_value(opportunity, "symbol") or ""),
        score=_float_cfg(_get_value(opportunity, "signal_score") or _get_value(opportunity, "score"), 0.0),
        fee_adjusted_rr=_float_cfg(_get_value(opportunity, "fee_adjusted_rr"), 0.0),
        reason=str(reason or ""),
        timestamp=_now_ts(now_ts) if now_ts is not None else None,
        raw=raw,
    )


def _reason_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _get_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    if hasattr(source, name):
        return getattr(source, name)
    return None


def _now_ts(now_ts: Optional[float]) -> float:
    if now_ts is None:
        return datetime.now(timezone.utc).timestamp()
    return float(now_ts)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_cfg(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_cfg(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
