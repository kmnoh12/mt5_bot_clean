from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


def train_oos_split(frame: Any, train_ratio: float = 0.7) -> Tuple[Any, Any]:
    if frame is None or len(frame) == 0:
        raise ValueError("frame must not be empty")
    ratio = min(0.95, max(0.1, float(train_ratio)))
    split = max(1, min(len(frame) - 1, int(len(frame) * ratio)))
    return frame.iloc[:split].reset_index(drop=True), frame.iloc[split:].reset_index(drop=True)


def attach_oos_metrics(train_metrics: Mapping[str, Any], oos_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    train_score = _simple_score(train_metrics)
    oos_score = _simple_score(oos_metrics)
    train_expectancy = float(train_metrics.get("expectancy_net", 0.0) or 0.0)
    oos_expectancy = float(oos_metrics.get("expectancy_net", 0.0) or 0.0)
    train_trades = int(train_metrics.get("total_trades", 0) or 0)
    oos_trades = int(oos_metrics.get("total_trades", 0) or 0)
    total_split_trades = train_trades + oos_trades
    oos_trade_share_pct = (oos_trades / total_split_trades * 100.0) if total_split_trades else 0.0
    oos_trade_dates = list(oos_metrics.get("trade_dates", []) or [])
    decay = 0.0
    if train_expectancy > 0:
        decay = max(0.0, (train_expectancy - oos_expectancy) / train_expectancy * 100.0)
    merged = dict(train_metrics)
    merged.update(
        {
            "train_score": train_score,
            "oos_score": oos_score,
            "oos_decay_pct": decay,
            "train_total_trades": train_trades,
            "total_split_trades": total_split_trades,
            "oos_trade_share_pct": oos_trade_share_pct,
            "oos_total_trades": int(oos_metrics.get("total_trades", 0) or 0),
            "oos_trading_day_count": int(oos_metrics.get("trading_day_count", len(oos_trade_dates)) or 0),
            "oos_trade_dates": oos_trade_dates,
            "oos_net_profit_factor": float(oos_metrics.get("net_profit_factor", 0.0) or 0.0),
            "oos_expectancy_net": oos_expectancy,
            "oos_max_drawdown_pct": float(oos_metrics.get("max_drawdown_pct", 0.0) or 0.0),
            "oos_no_trade_days_pct": float(oos_metrics.get("no_trade_days_pct", 0.0) or 0.0),
        }
    )
    return merged


def _simple_score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics.get("total_net_pnl", 0.0) or 0.0)
        + float(metrics.get("net_profit_factor", 0.0) or 0.0)
        + float(metrics.get("expectancy_net", 0.0) or 0.0)
        - float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
        - float(metrics.get("no_trade_days_pct", 0.0) or 0.0) * 0.05
    )
