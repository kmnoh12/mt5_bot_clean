from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_PROFIT_LOCK_STAGES = (
    {"trigger_net_profit": 30.0, "lock_net_profit": 15.0, "target_net_profit": 45.0},
    {"trigger_net_profit": 20.0, "lock_net_profit": 10.0, "target_net_profit": 30.0},
    {"trigger_net_profit": 10.0, "lock_net_profit": 5.0, "target_net_profit": 20.0},
    {"trigger_net_profit": 5.0, "lock_net_profit": 2.0, "target_net_profit": None},
    {"trigger_net_profit": 3.0, "lock_net_profit": 1.0, "target_net_profit": None},
    {"trigger_net_profit": 2.0, "lock_net_profit": 0.0, "target_net_profit": None},
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_active_opportunity_report(
    source: Any,
    *,
    generated_at_utc: Optional[str] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Build a pure-Python active opportunity report from offline data.

    ``source`` may be a dict, dataclass, or object with attributes. Candidates
    may also be dicts or dataclasses. The builder never calls MT5 or broker APIs.
    """

    data = _as_mapping(source)
    candidate_items = _sequence(source) if not data and isinstance(source, (list, tuple, set)) else _candidate_items(data)
    candidates = [_normalize_candidate(item, default_timeframe=_current_timeframe(data)) for item in candidate_items]
    explicit_rejected = [
        _normalize_candidate(item, default_timeframe=_current_timeframe(data), force_rejected=True)
        for item in _sequence_value(data, ("rejected_candidates", "rejected", "rejects"))
    ]

    eligible = [item for item in candidates if item["eligible"]]
    rejected = [item for item in candidates if not item["eligible"]] + explicit_rejected
    long_candidates = _top_by_direction(eligible, "long", top_n=top_n)
    short_candidates = _top_by_direction(eligible, "short", top_n=top_n)
    best = max(eligible, key=_candidate_sort_key) if eligible else None

    block_reasons = _block_reason_counts(rejected)
    for reason in _sequence_value(data, ("block_reasons", "blocks")):
        reason_text = _reason_text(reason)
        if reason_text:
            block_reasons[reason_text] = block_reasons.get(reason_text, 0) + 1

    symbols = _current_symbols(data, candidates + explicit_rejected)
    timeframe = _current_timeframe(data)
    return {
        "generated_at_utc": generated_at_utc or str(_value(data, ("generated_at_utc", "generated_at"), utc_now_iso())),
        "current_symbols": symbols,
        "current_timeframe": timeframe,
        "top_long_candidates": long_candidates,
        "top_short_candidates": short_candidates,
        "rejected_candidates": rejected,
        "block_reasons": dict(sorted(block_reasons.items())),
        "best_eligible_candidate": best,
        "candidate_count": len(candidates) + len(explicit_rejected),
        "eligible_count": len(eligible),
        "rejected_count": len(rejected),
    }


def build_active_opportunity_markdown(report_or_source: Any, *, top_n: int = 5) -> str:
    report = (
        report_or_source
        if isinstance(report_or_source, Mapping) and "top_long_candidates" in report_or_source
        else build_active_opportunity_report(report_or_source, top_n=top_n)
    )

    lines = [
        "# Active Opportunity Report",
        "",
        f"- generated_at_utc: {report.get('generated_at_utc')}",
        f"- current_symbols: {', '.join(report.get('current_symbols') or []) or '-'}",
        f"- current_timeframe: {report.get('current_timeframe') or '-'}",
        f"- eligible/rejected: {report.get('eligible_count', 0)}/{report.get('rejected_count', 0)}",
        "",
    ]

    best = report.get("best_eligible_candidate")
    lines.extend(["## Best Eligible Candidate", ""])
    lines.extend(_candidate_detail_lines(best))
    lines.append("")

    lines.extend(["## Top Long Candidates", ""])
    lines.extend(_candidate_table(report.get("top_long_candidates") or []))
    lines.extend(["", "## Top Short Candidates", ""])
    lines.extend(_candidate_table(report.get("top_short_candidates") or []))
    lines.extend(["", "## Rejected Candidates", ""])
    lines.extend(_candidate_table(report.get("rejected_candidates") or [], include_reasons=True))
    lines.extend(["", "## Block Reasons", ""])
    reasons = report.get("block_reasons") or {}
    if reasons:
        for reason, count in sorted(reasons.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_active_opportunity_reports(
    source: Any,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    generated_at_utc: Optional[str] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    report = build_active_opportunity_report(source, generated_at_utc=generated_at_utc, top_n=top_n)
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_out.write_text(build_active_opportunity_markdown(report, top_n=top_n), encoding="utf-8")
    return report


def _normalize_candidate(item: Any, *, default_timeframe: Optional[str], force_rejected: bool = False) -> Dict[str, Any]:
    data = _candidate_mapping(item)
    direction = _normalize_direction(_value(data, ("direction", "side", "action")))
    score = _finite_float(_value(data, ("score", "rank_score", "edge_score", "edge", "confidence"), 0.0)) or 0.0
    reasons = _candidate_reasons(data)
    explicit_eligible = _bool_or_none(_value(data, ("eligible", "allow", "passed", "is_eligible")))
    eligible = bool(explicit_eligible) if explicit_eligible is not None else not reasons
    if force_rejected:
        eligible = False

    current_spread = _finite_float(_value(data, ("current_spread", "spread", "spread_points")))
    current_cost = _finite_float(
        _value(
            data,
            (
                "current_cost",
                "current_spread_cost",
                "estimated_cost_usd",
                "estimated_round_trip_cost",
                "round_trip_cost",
                "cost",
            ),
        )
    )
    estimated_lot = _finite_float(_value(data, ("estimated_lot", "lot", "volume", "position_size", "recommended_lot")))
    estimated_sl_net_loss = _finite_float(
        _value(
            data,
            (
                "estimated_sl_net_loss",
                "estimated_net_loss_usd",
                "expected_net_loss_at_sl",
                "net_loss",
            ),
        )
    )
    estimated_tp_net_profit = _finite_float(
        _value(
            data,
            (
                "estimated_tp_net_profit",
                "estimated_net_profit_at_tp_usd",
                "expected_net_profit_at_tp",
                "net_profit",
            ),
        )
    )

    computed = _compute_net_risk(data, direction=direction, lot=estimated_lot, current_cost=current_cost)
    if estimated_sl_net_loss is None:
        estimated_sl_net_loss = computed.get("estimated_sl_net_loss")
    if estimated_tp_net_profit is None:
        estimated_tp_net_profit = computed.get("estimated_tp_net_profit")
    if current_cost is None:
        current_cost = computed.get("current_cost")

    candidate = {
        "symbol": _text(_value(data, ("symbol", "instrument")), default=""),
        "timeframe": _text(_value(data, ("timeframe", "current_timeframe"), default_timeframe), default=""),
        "direction": direction,
        "score": score,
        "eligible": eligible,
        "setup": _text(_value(data, ("setup", "strategy", "name")), default=""),
        "block_reasons": reasons,
        "current_spread": current_spread,
        "current_cost": current_cost,
        "estimated_lot": estimated_lot,
        "estimated_sl_net_loss": estimated_sl_net_loss,
        "estimated_tp_net_profit": estimated_tp_net_profit,
        "profit_lock_plan": _profit_lock_plan(data, estimated_tp_net_profit),
    }
    return candidate


def _compute_net_risk(
    data: Mapping[str, Any],
    *,
    direction: str,
    lot: Optional[float],
    current_cost: Optional[float],
) -> Dict[str, Optional[float]]:
    entry = _finite_float(_value(data, ("entry_price", "entry")))
    stop = _finite_float(_value(data, ("stop_price", "sl_price", "invalidation_price", "stop_loss")))
    tp = _finite_float(_value(data, ("take_profit_price", "tp_price", "target_reference_price", "target_price")))
    tick_size = _finite_float(_value(data, ("tick_size", "point"), 1.0))
    tick_value = _finite_float(_value(data, ("tick_value",), 1.0))
    lot_value = lot if lot is not None else _finite_float(_value(data, ("lot", "volume", "position_size")))

    if entry is None or lot_value is None or tick_size is None or tick_value is None or tick_size <= 0:
        return {}

    cost = current_cost
    if cost is None:
        spread = _finite_float(_value(data, ("current_spread", "spread", "spread_points"), 0.0)) or 0.0
        slippage_points = _finite_float(_value(data, ("expected_slippage_points", "slippage_points"), 0.0)) or 0.0
        commission = _finite_float(_value(data, ("commission_per_lot", "commission"), 0.0)) or 0.0
        spread_cost = (spread / tick_size) * tick_value * lot_value
        slippage_cost = slippage_points * tick_value * lot_value
        cost = spread_cost + slippage_cost + (commission * lot_value * 2.0)

    out: Dict[str, Optional[float]] = {"current_cost": _round_float(cost)}
    if stop is not None:
        adverse = entry - stop if direction == "long" else stop - entry
        if adverse > 0:
            out["estimated_sl_net_loss"] = _round_float((adverse / tick_size) * tick_value * lot_value + cost)
    if tp is not None:
        favorable = tp - entry if direction == "long" else entry - tp
        if favorable > 0:
            out["estimated_tp_net_profit"] = _round_float((favorable / tick_size) * tick_value * lot_value - cost)
    return out


def _profit_lock_plan(data: Mapping[str, Any], estimated_tp_net_profit: Optional[float]) -> Dict[str, Any]:
    explicit = _value(data, ("profit_lock_plan", "profit_lock"))
    if isinstance(explicit, Mapping):
        return dict(explicit)
    stages = [dict(stage) for stage in DEFAULT_PROFIT_LOCK_STAGES]
    reachable = []
    if estimated_tp_net_profit is not None:
        reachable = [stage for stage in stages if estimated_tp_net_profit + 1e-12 >= float(stage["trigger_net_profit"])]
    return {
        "enabled": True,
        "basis": "net_unrealized_pnl_after_estimated_exit_costs",
        "stages": stages,
        "reachable_stages_at_estimated_tp": reachable,
    }


def _candidate_items(data: Mapping[str, Any]) -> Sequence[Any]:
    items = _sequence_value(data, ("candidates", "opportunities", "results", "items"))
    if items:
        return items
    singular = _value(data, ("candidate", "opportunity"))
    if singular is not None:
        return _sequence(singular)
    if _value(data, ("symbol", "instrument")) is not None or _value(data, ("direction", "side", "action")) is not None:
        return [data]
    return []


def _candidate_mapping(item: Any) -> Mapping[str, Any]:
    data = dict(_as_mapping(item))
    nested = _as_mapping(_value(data, ("opportunity", "candidate")))
    if nested:
        merged = dict(nested)
        for key, value in data.items():
            if key not in {"opportunity", "candidate"} and value is not None:
                merged[key] = value
        data = merged

    filter_result = _as_mapping(_value(data, ("filter_result", "filter", "quality_filter")))
    if filter_result:
        for key in ("eligible", "allow", "passed", "is_eligible", "block_reasons", "reject_reasons", "reason", "reject_reason"):
            if key in filter_result and key not in data:
                data[key] = filter_result[key]
    return data


def _top_by_direction(candidates: Sequence[Dict[str, Any]], direction: str, *, top_n: int) -> List[Dict[str, Any]]:
    return sorted((item for item in candidates if item["direction"] == direction), key=_candidate_sort_key, reverse=True)[
        : max(0, int(top_n))
    ]


def _candidate_sort_key(item: Mapping[str, Any]) -> tuple:
    net_profit = item.get("estimated_tp_net_profit")
    return (float(item.get("score") or 0.0), float(net_profit) if net_profit is not None else -math.inf)


def _block_reason_counts(rejected: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in rejected:
        reasons = item.get("block_reasons") or ["UNSPECIFIED_REJECTION"]
        for reason in reasons:
            reason_text = _reason_text(reason)
            if reason_text:
                counts[reason_text] = counts.get(reason_text, 0) + 1
    return counts


def _candidate_reasons(data: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("block_reasons", "reject_reasons", "reasons"):
        for item in _sequence_value(data, (key,)):
            text = _reason_text(item)
            if text and text not in out:
                out.append(text)
    for key in ("block_reason", "reject_reason", "reason"):
        text = _reason_text(_value(data, (key,)))
        if text and text not in out and not _looks_like_positive_reason(text):
            out.append(text)
    return out


def _looks_like_positive_reason(reason: str) -> bool:
    return reason.strip().upper() in {"OK", "ALLOW", "ALLOWED", "PASS", "PASSED"}


def _candidate_detail_lines(candidate: Optional[Mapping[str, Any]]) -> List[str]:
    if not candidate:
        return ["No eligible candidate."]
    return [
        f"- symbol: {candidate.get('symbol') or '-'}",
        f"- direction: {candidate.get('direction') or '-'}",
        f"- score: {_fmt(candidate.get('score'))}",
        f"- current spread/cost: {_fmt(candidate.get('current_spread'))} / {_fmt(candidate.get('current_cost'))}",
        f"- estimated lot: {_fmt(candidate.get('estimated_lot'))}",
        f"- estimated SL net loss: {_fmt(candidate.get('estimated_sl_net_loss'))}",
        f"- estimated TP net profit: {_fmt(candidate.get('estimated_tp_net_profit'))}",
        f"- profit-lock stages: {len((candidate.get('profit_lock_plan') or {}).get('stages') or [])}",
    ]


def _candidate_table(candidates: Sequence[Mapping[str, Any]], *, include_reasons: bool = False) -> List[str]:
    if not candidates:
        return ["No candidates."]
    headers = ["symbol", "dir", "score", "spread", "cost", "lot", "sl_net_loss", "tp_net_profit"]
    if include_reasons:
        headers.append("reasons")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for item in candidates:
        row = [
            str(item.get("symbol") or "-"),
            str(item.get("direction") or "-"),
            _fmt(item.get("score")),
            _fmt(item.get("current_spread")),
            _fmt(item.get("current_cost")),
            _fmt(item.get("estimated_lot")),
            _fmt(item.get("estimated_sl_net_loss")),
            _fmt(item.get("estimated_tp_net_profit")),
        ]
        if include_reasons:
            row.append(", ".join(item.get("block_reasons") or []) or "-")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _current_symbols(data: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> List[str]:
    raw = _value(data, ("current_symbols", "symbols", "symbol_universe"))
    symbols = [_text(item, default="") for item in _sequence(raw)]
    if not symbols:
        symbols = [_text(item.get("symbol"), default="") for item in candidates]
    return sorted({symbol for symbol in symbols if symbol})


def _current_timeframe(data: Mapping[str, Any]) -> Optional[str]:
    text = _text(_value(data, ("current_timeframe", "timeframe")), default="")
    return text or None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _value(data: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _sequence_value(data: Mapping[str, Any], keys: Sequence[str]) -> Sequence[Any]:
    return _sequence(_value(data, keys, []))


def _sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Mapping):
        return [value]
    return [value]


def _normalize_direction(value: Any) -> str:
    text = str(getattr(value, "value", value) or "").strip().lower()
    if text in {"buy", "long", "bull", "bullish"}:
        return "long"
    if text in {"sell", "short", "bear", "bearish"}:
        return "short"
    return text or "unknown"


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _round_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 10)


def _fmt(value: Any) -> str:
    num = _finite_float(value)
    if num is None:
        return "-"
    return f"{num:.6g}"


def _text(value: Any, *, default: str) -> str:
    if value is None:
        return default
    text = str(getattr(value, "value", value)).strip()
    return text if text else default


def _reason_text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = _value(value, ("reason", "code", "name", "message"))
    return _text(value, default="")


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "allow", "allowed", "pass", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "block", "blocked", "reject", "rejected", "fail", "failed"}:
        return False
    return None
