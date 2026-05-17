from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


DEFAULT_MIN_OOS_TRADES = 20
DEFAULT_MIN_SHADOW_SAMPLES = 20
DEFAULT_MIN_WALK_FORWARD_WINDOWS = 2


def load_validation_report(path: str) -> Dict:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def check_live_readiness(report_path: str) -> Tuple[bool, str]:
    payload = load_validation_report(report_path)
    if not payload:
        return False, "validation_report_missing"
    if not bool(payload.get("oos_pass", False)):
        return False, str(payload.get("reason", "oos_failed"))

    thresholds = _mapping(payload.get("thresholds"))
    min_oos_trades = int(_first_number((payload, thresholds), ("min_oos_trades",), DEFAULT_MIN_OOS_TRADES) or 0)
    oos_trades = _first_number((payload,), ("oos_total_trades", "oos_trades", "out_of_sample_trades"), None)
    if oos_trades is None:
        return False, "oos_total_trades_missing"
    if int(oos_trades) < min_oos_trades:
        return False, "oos_total_trades_lt_min"

    min_oos_trading_days = int(_first_number((payload, thresholds), ("min_oos_trading_days",), 0) or 0)
    if min_oos_trading_days > 0:
        oos_trading_days = _first_number((payload,), ("oos_trading_day_count", "oos_active_days"), None)
        if oos_trading_days is None:
            trade_dates = payload.get("oos_trade_dates")
            if isinstance(trade_dates, (list, tuple, set)):
                oos_trading_days = len({str(item)[:10] for item in trade_dates if len(str(item or "")) >= 10})
        if oos_trading_days is None:
            return False, "oos_trading_day_count_missing"
        if int(oos_trading_days) < min_oos_trading_days:
            return False, "oos_trading_days_lt_min"

    symbol_metrics = _mapping(payload.get("symbol_metrics"))
    min_oos_trades_per_symbol = int(
        _first_number((payload, thresholds), ("min_oos_trades_per_symbol",), 0) or 0
    )
    min_oos_trading_days_per_symbol = int(
        _first_number((payload, thresholds), ("min_oos_trading_days_per_symbol",), 0) or 0
    )
    if min_oos_trades_per_symbol > 0 or min_oos_trading_days_per_symbol > 0:
        if not symbol_metrics:
            return False, "symbol_oos_metrics_missing"
        symbol_check = _check_symbol_oos_coverage(
            symbol_metrics,
            min_trades=min_oos_trades_per_symbol,
            min_trading_days=min_oos_trading_days_per_symbol,
        )
        if symbol_check:
            return False, symbol_check

    min_oos_trades_per_direction = int(
        _first_number((payload, thresholds), ("min_oos_trades_per_direction",), 0) or 0
    )
    if min_oos_trades_per_direction > 0:
        direction_check = _check_direction_oos_coverage(
            _mapping(payload.get("oos_direction_trade_counts")),
            min_trades=min_oos_trades_per_direction,
        )
        if direction_check:
            return False, direction_check

    if not bool(payload.get("walk_forward_pass", False)):
        return False, "walk_forward_not_passed"
    min_walk_forward_windows = int(
        _first_number(
            (payload, thresholds),
            ("min_walk_forward_windows", "min_walk_forward_folds"),
            DEFAULT_MIN_WALK_FORWARD_WINDOWS,
        )
        or 0
    )
    if min_walk_forward_windows > 0:
        walk_forward_windows = _walk_forward_window_count(payload)
        if walk_forward_windows is None:
            return False, "walk_forward_window_count_missing"
        if int(walk_forward_windows) < min_walk_forward_windows:
            return False, "walk_forward_windows_lt_min"
    walk_forward_detail_check = _check_walk_forward_window_details(payload, thresholds)
    if walk_forward_detail_check:
        return False, walk_forward_detail_check

    shadow_payload = _mapping(payload.get("shadow")) or _mapping(payload.get("shadow_evaluation")) or payload
    shadow_gate = _mapping(shadow_payload.get("promotion_gate"))
    shadow_gate_status = str(shadow_gate.get("status", "")).lower()
    if shadow_gate and (
        shadow_gate_status in {"blocked", "fail", "failed", "reject", "rejected"}
        or bool(shadow_gate.get("block_reasons"))
    ):
        return False, "shadow_promotion_gate_blocked"
    shadow_pass = (
        bool(payload.get("shadow_pass", False))
        or bool(shadow_payload.get("shadow_pass", False))
        or bool(shadow_payload.get("live_review_ready", False))
        or shadow_gate_status == "pass"
    )
    if not shadow_pass:
        return False, "shadow_not_passed"

    shadow_thresholds = _mapping(shadow_payload.get("thresholds"))
    min_shadow_samples = int(
        _first_number(
            (payload, thresholds, shadow_payload, shadow_thresholds),
            ("min_shadow_samples", "min_live_review_samples"),
            DEFAULT_MIN_SHADOW_SAMPLES,
        )
        or 0
    )
    shadow_samples = _first_number(
        (payload, shadow_payload),
        (
            "shadow_sample_count",
            "shadow_samples",
            "live_review_sample_count",
            "actual_trade_samples",
            "matched_trade_count",
            "candidate_matched_trade_count_sum",
            "blocked_total",
        ),
        None,
    )
    if shadow_samples is None:
        return False, "shadow_sample_count_missing"
    if int(shadow_samples) < min_shadow_samples:
        return False, "shadow_sample_count_lt_min"

    return True, "oos_walk_forward_shadow_pass"


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_number(sources: Iterable[Mapping[str, Any]], keys: Iterable[str], default: Optional[float]) -> Optional[float]:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool) or value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _check_symbol_oos_coverage(
    symbol_metrics: Mapping[str, Any],
    *,
    min_trades: int,
    min_trading_days: int,
) -> str:
    for raw_payload in symbol_metrics.values():
        payload = _mapping(raw_payload)
        oos_payload = _mapping(payload.get("oos")) or payload
        if min_trades > 0:
            trades = _first_number(
                (oos_payload,),
                ("total_trades", "oos_total_trades", "oos_trades", "out_of_sample_trades"),
                None,
            )
            if trades is None:
                return "symbol_oos_total_trades_missing"
            if int(trades) < min_trades:
                return "symbol_oos_total_trades_lt_min"
        if min_trading_days > 0:
            days = _first_number((oos_payload,), ("trading_day_count", "oos_trading_day_count", "oos_active_days"), None)
            if days is None:
                trade_dates = oos_payload.get("trade_dates") or oos_payload.get("oos_trade_dates")
                if isinstance(trade_dates, (list, tuple, set)):
                    days = len({str(item)[:10] for item in trade_dates if len(str(item or "")) >= 10})
            if days is None:
                return "symbol_oos_trading_day_count_missing"
            if int(days) < min_trading_days:
                return "symbol_oos_trading_days_lt_min"
    return ""


def _check_direction_oos_coverage(direction_counts: Mapping[str, Any], *, min_trades: int) -> str:
    if not direction_counts:
        return "oos_direction_trade_counts_missing"
    long_count = _direction_count(direction_counts, "long", "buy", "bull")
    short_count = _direction_count(direction_counts, "short", "sell", "bear")
    if long_count is None or short_count is None:
        return "oos_direction_trade_counts_missing"
    if min(int(long_count), int(short_count)) < min_trades:
        return "oos_direction_trades_lt_min"
    return ""


def _walk_forward_window_count(payload: Mapping[str, Any]) -> Optional[int]:
    direct = _first_number(
        (payload,),
        (
            "walk_forward_window_count",
            "walk_forward_windows_count",
            "walk_forward_fold_count",
            "walk_forward_folds_count",
            "wf_window_count",
            "wf_fold_count",
        ),
        None,
    )
    if direct is not None:
        return int(direct)

    for key in ("walk_forward_windows", "walk_forward_results", "walk_forward_folds", "folds"):
        value = payload.get(key)
        if isinstance(value, (list, tuple, set)):
            return len(value)

    walk_forward = _mapping(payload.get("walk_forward"))
    if not walk_forward:
        return None
    direct = _first_number(
        (walk_forward,),
        (
            "window_count",
            "windows_count",
            "fold_count",
            "folds_count",
            "walk_forward_window_count",
            "walk_forward_fold_count",
        ),
        None,
    )
    if direct is not None:
        return int(direct)
    for key in ("windows", "results", "folds"):
        value = walk_forward.get(key)
        if isinstance(value, (list, tuple, set)):
            return len(value)
    return None


def _walk_forward_window_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("walk_forward_windows", "walk_forward_results", "walk_forward_folds", "folds"):
        value = payload.get(key)
        if isinstance(value, (list, tuple, set)):
            rows.extend(_mapping(item) for item in value if isinstance(item, Mapping))

    walk_forward = _mapping(payload.get("walk_forward"))
    for key in ("windows", "results", "folds"):
        value = walk_forward.get(key)
        if isinstance(value, (list, tuple, set)):
            rows.extend(_mapping(item) for item in value if isinstance(item, Mapping))
    return rows


def _check_walk_forward_window_details(payload: Mapping[str, Any], thresholds: Mapping[str, Any]) -> str:
    rows = _walk_forward_window_rows(payload)
    if not rows:
        return ""

    min_window_oos_trades = int(
        _first_number(
            (payload, thresholds),
            (
                "min_walk_forward_oos_trades_per_window",
                "min_walk_forward_oos_trades_per_fold",
                "min_oos_trades_per_walk_forward_window",
                "min_oos_trades_per_walk_forward_fold",
            ),
            0,
        )
        or 0
    )
    for row in rows:
        if _window_failed(row):
            return "walk_forward_window_failed"
        if min_window_oos_trades > 0:
            trades = _first_number(
                (row,),
                ("oos_total_trades", "oos_trades", "out_of_sample_trades", "test_total_trades", "test_trades"),
                None,
            )
            if trades is None:
                return "walk_forward_window_oos_trades_missing"
            if int(trades) < min_window_oos_trades:
                return "walk_forward_window_oos_trades_lt_min"
    return ""


def _window_failed(row: Mapping[str, Any]) -> bool:
    for key in ("pass", "passed", "oos_pass", "validation_pass", "walk_forward_pass"):
        value = row.get(key)
        if isinstance(value, bool) and not value:
            return True
    status = str(row.get("status", row.get("verdict", "")) or "").strip().lower()
    return status in {"blocked", "fail", "failed", "reject", "rejected"}


def _direction_count(counts: Mapping[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = counts.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
