from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


PNL_KEYS = ("realized_pnl", "pnl", "profit", "net_pnl", "pnl_usd")
COST_KEYS = (
    "fee",
    "fees",
    "commission",
    "commissions",
    "swap",
    "spread_cost",
    "cost",
    "costs",
    "realized_cost_usd",
)
ENTRY_TIME_KEYS = ("entry_time", "open_time", "time_open", "time_open_utc", "opened_at")
EXIT_TIME_KEYS = ("exit_time", "close_time", "time_close", "time_close_utc", "closed_at")


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _first_float(record: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in record:
            value = _as_float(record.get(key))
            if value is not None:
                return value
    return None


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    numeric = _as_float(value)
    if numeric is not None:
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _hold_seconds(record: Dict[str, Any]) -> float:
    explicit = _first_float(record, ("hold_seconds", "duration_seconds", "seconds_in_trade"))
    if explicit is not None and explicit >= 0.0:
        return explicit
    entry = None
    exit_time = None
    for key in ENTRY_TIME_KEYS:
        entry = _parse_time(record.get(key))
        if entry is not None:
            break
    for key in EXIT_TIME_KEYS:
        exit_time = _parse_time(record.get(key))
        if exit_time is not None:
            break
    if entry is None or exit_time is None:
        return 0.0
    return max(0.0, (exit_time - entry).total_seconds())


def _normalize_trade(record: Any) -> Optional[Dict[str, float]]:
    if isinstance(record, dict):
        raw = dict(record)
    else:
        pnl = _as_float(record)
        if pnl is None:
            return None
        raw = {"pnl": pnl}

    pnl = _first_float(raw, PNL_KEYS)
    if pnl is None:
        return None

    cost = 0.0
    for key in COST_KEYS:
        value = _as_float(raw.get(key))
        if value is not None:
            cost += abs(value)

    return {
        "pnl": float(pnl),
        "cost": float(cost),
        "hold_seconds": float(_hold_seconds(raw)),
    }


def compute_backtest_metrics(
    trades: Iterable[Any],
    *,
    tiny_pnl_threshold: float = 2.0,
    quick_exit_seconds: float = 300.0,
) -> Dict[str, Any]:
    normalized = [trade for trade in (_normalize_trade(item) for item in trades) if trade is not None]
    pnl_values = [float(item["pnl"]) for item in normalized]
    count = len(pnl_values)
    wins = [value for value in pnl_values if value > 0.0]
    losses = [value for value in pnl_values if value < 0.0]
    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    net_pnl = float(sum(pnl_values))

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    quick_exit_count = 0
    churn_count = 0
    for item in normalized:
        hold_seconds = float(item["hold_seconds"])
        pnl_abs = abs(float(item["pnl"]))
        if hold_seconds <= quick_exit_seconds:
            quick_exit_count += 1
            if pnl_abs <= tiny_pnl_threshold:
                churn_count += 1

    profit_factor: Any
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "trades": int(count),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / count) if count else 0.0,
        "avg_win": float(sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "expectancy": float(net_pnl / count) if count else 0.0,
        "net_pnl": float(net_pnl),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": profit_factor,
        "max_drawdown": float(max_drawdown),
        "fees_cost_estimate": float(sum(float(item["cost"]) for item in normalized)),
        "exposure_seconds": float(sum(float(item["hold_seconds"]) for item in normalized)),
        "avg_hold_seconds": float(sum(float(item["hold_seconds"]) for item in normalized) / count) if count else 0.0,
        "quick_exit_count": int(quick_exit_count),
        "churn_count": int(churn_count),
    }


def walk_forward_splits(
    *,
    total_rows: int,
    train_rows: int,
    test_rows: int,
    step_rows: Optional[int] = None,
) -> List[Dict[str, int]]:
    total = max(0, int(total_rows))
    train = max(1, int(train_rows))
    test = max(1, int(test_rows))
    step = max(1, int(step_rows if step_rows is not None else test))

    splits: List[Dict[str, int]] = []
    train_start = 0
    fold = 1
    while True:
        train_end = train_start + train
        test_start = train_end
        test_end = test_start + test
        if test_end > total:
            break
        splits.append(
            {
                "fold": int(fold),
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
        train_start += step
        fold += 1
    return splits
