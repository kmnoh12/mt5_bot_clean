from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping


REQUIRED_METRIC_KEYS = (
    "total_net_pnl",
    "gross_pnl",
    "total_cost",
    "expectancy_net",
    "net_profit_factor",
    "total_trades",
    "win_rate",
    "avg_win",
    "avg_loss",
    "payoff_ratio",
    "median_trades_per_day",
    "no_trade_days",
    "no_trade_days_pct",
    "unexplained_no_trade_days",
    "max_drawdown",
    "max_drawdown_pct",
    "max_single_trade_net_loss",
    "hard_loss_breach_count",
    "daily_loss_breach_count",
    "daily_bleed_halt_count",
    "consecutive_losses_max",
    "spread_cost",
    "commission_cost",
    "slippage_cost",
    "effective_leverage_max",
    "margin_used_pct_max",
    "rejected_by_effective_leverage_count",
    "rejected_by_margin_count",
    "profit_lock_move_count",
    "profit_lock_saved_pnl",
    "breakeven_move_count",
    "runner_tp_extension_count",
    "winner_capture_ratio",
    "mfe_mae_summary",
    "raw_signal_count",
    "scored_signal_count",
    "eligible_signal_count",
    "executed_trade_count",
    "block_reason_distribution",
    "train_score",
    "oos_score",
    "oos_decay_pct",
    "oos_total_trades",
    "oos_net_profit_factor",
    "oos_expectancy_net",
)


def metrics_from_backtest(
    result: Mapping[str, Any],
    *,
    hard_max_net_loss_usd: float,
    max_daily_loss_usd: float,
    max_effective_leverage: float,
    max_margin_used_pct: float,
    leverage_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    source = dict(result.get("metrics", {}) or {})
    trades = list(result.get("trades", []) or [])
    leverage = dict(leverage_metrics or {})
    total_net = float(source.get("total_net_pnl", 0.0) or 0.0)
    gross_pnl = sum(float(t.get("gross_pnl", 0.0) or 0.0) for t in trades if isinstance(t, Mapping))
    total_cost = sum(float(t.get("cost_usd", 0.0) or 0.0) for t in trades if isinstance(t, Mapping))
    net_pnls = [float(t.get("net_pnl", 0.0) or 0.0) for t in trades if isinstance(t, Mapping)]
    gross_profit = sum(p for p in net_pnls if p > 0.0)
    gross_loss = abs(sum(p for p in net_pnls if p < 0.0))
    win_count = sum(1 for p in net_pnls if p > 0.0)
    loss_count = sum(1 for p in net_pnls if p < 0.0)
    trade_count = int(source.get("executed_trade_count", len(trades)) or 0)
    max_loss = float(source.get("max_single_trade_loss", 0.0) or 0.0)
    daily_loss = float(source.get("daily_max_loss", 0.0) or 0.0)
    block_reasons = dict(source.get("block_reasons", {}) or {})
    daily_bleed_blocks = sum(
        int(count or 0)
        for reason, count in block_reasons.items()
        if str(reason).startswith("daily_bleed") or "DAILY_BLEED" in str(reason)
    )
    effective_max = float(leverage.get("effective_leverage_max", 0.0) or 0.0)
    margin_max = float(leverage.get("margin_used_pct_max", 0.0) or 0.0)
    out = {
        "total_net_pnl": total_net,
        "gross_pnl": gross_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "win_count": win_count,
        "loss_count": loss_count,
        "total_cost": total_cost,
        "expectancy_net": total_net / trade_count if trade_count else 0.0,
        "net_profit_factor": gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "total_trades": trade_count,
        "win_rate": float(source.get("win_rate", 0.0) or 0.0),
        "avg_win": float(source.get("avg_win", 0.0) or 0.0),
        "avg_loss": float(source.get("avg_loss", 0.0) or 0.0),
        "payoff_ratio": float(source.get("payoff_ratio", 0.0) or 0.0),
        "median_trades_per_day": float(source.get("median_trades_per_day", 0.0) or 0.0),
        "no_trade_days": int(source.get("no_trade_days_count", 0) or 0),
        "no_trade_days_pct": float(source.get("no_trade_days_pct", 0.0) or 0.0),
        "unexplained_no_trade_days": _unexplained_no_trade_days(source),
        "max_drawdown": float(source.get("max_drawdown", 0.0) or 0.0),
        "max_drawdown_pct": float(source.get("max_drawdown", 0.0) or 0.0) / 10000.0 * 100.0,
        "max_single_trade_net_loss": max_loss,
        "hard_loss_breach_count": sum(1 for t in trades if float(t.get("net_pnl", 0.0) or 0.0) < -abs(float(hard_max_net_loss_usd)) - 1e-9),
        "daily_loss_breach_count": 1 if daily_loss < -abs(float(max_daily_loss_usd)) - 1e-9 else 0,
        "daily_bleed_halt_count": int(source.get("daily_bleed_halt_count", daily_bleed_blocks) or daily_bleed_blocks),
        "consecutive_losses_max": int(source.get("consecutive_losses_max", 0) or 0),
        "spread_cost": float(leverage.get("spread_cost", 0.0) or 0.0),
        "commission_cost": float(leverage.get("commission_cost", total_cost) or 0.0),
        "slippage_cost": float(leverage.get("slippage_cost", 0.0) or 0.0),
        "effective_leverage_max": effective_max,
        "margin_used_pct_max": margin_max,
        "rejected_by_effective_leverage_count": 1 if effective_max > float(max_effective_leverage) + 1e-12 else 0,
        "rejected_by_margin_count": 1 if margin_max > float(max_margin_used_pct) + 1e-12 else 0,
        "profit_lock_move_count": int(source.get("profit_lock_sl_moves", 0) or 0),
        "profit_lock_saved_pnl": float(source.get("profit_lock_saved_pnl", 0.0) or 0.0),
        "breakeven_move_count": int(source.get("breakeven_move_count", 0) or 0),
        "runner_tp_extension_count": int(source.get("runner_tp_extension_count", 0) or 0),
        "winner_capture_ratio": float(source.get("winner_capture_ratio", 0.0) or 0.0),
        "mfe_mae_summary": {"mfe_max": source.get("mfe_max", 0.0), "mae_min": source.get("mae_min", 0.0)},
        "raw_signal_count": int(source.get("raw_signal_count", 0) or 0),
        "scored_signal_count": int(source.get("scored_signal_count", 0) or 0),
        "eligible_signal_count": int(source.get("eligible_signal_count", 0) or 0),
        "executed_trade_count": trade_count,
        "block_reason_distribution": block_reasons,
        "train_score": 0.0,
        "oos_score": 0.0,
        "oos_decay_pct": 0.0,
        "oos_total_trades": 0,
        "oos_net_profit_factor": 0.0,
        "oos_expectancy_net": 0.0,
    }
    return out


def aggregate_metrics(items: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [dict(item) for item in items]
    if not values:
        return {key: 0 for key in REQUIRED_METRIC_KEYS}
    total_trades = sum(int(v.get("total_trades", 0) or 0) for v in values)
    total_net = sum(float(v.get("total_net_pnl", 0.0) or 0.0) for v in values)
    gross_profit = sum(max(0.0, float(v.get("total_net_pnl", 0.0) or 0.0)) for v in values)
    gross_loss = abs(sum(min(0.0, float(v.get("total_net_pnl", 0.0) or 0.0)) for v in values))
    trade_gross_profit = sum(float(v.get("gross_profit", 0.0) or 0.0) for v in values)
    trade_gross_loss = sum(float(v.get("gross_loss", 0.0) or 0.0) for v in values)
    out = {key: 0 for key in REQUIRED_METRIC_KEYS}
    out.update(
        {
            "total_net_pnl": total_net,
            "gross_pnl": sum(float(v.get("gross_pnl", 0.0) or 0.0) for v in values),
            "gross_profit": trade_gross_profit,
            "gross_loss": trade_gross_loss,
            "win_count": sum(int(v.get("win_count", 0) or 0) for v in values),
            "loss_count": sum(int(v.get("loss_count", 0) or 0) for v in values),
            "total_cost": sum(float(v.get("total_cost", 0.0) or 0.0) for v in values),
            "expectancy_net": total_net / total_trades if total_trades else 0.0,
            "net_profit_factor": trade_gross_profit / trade_gross_loss if trade_gross_loss > 0 else (999.0 if trade_gross_profit > 0 else 0.0),
            "total_trades": total_trades,
            "executed_trade_count": total_trades,
            "win_rate": _weighted_avg(values, "win_rate", "total_trades"),
            "avg_win": _avg(values, "avg_win"),
            "avg_loss": _avg(values, "avg_loss"),
            "payoff_ratio": _avg(values, "payoff_ratio"),
            "median_trades_per_day": _avg(values, "median_trades_per_day"),
            "no_trade_days": sum(int(v.get("no_trade_days", 0) or 0) for v in values),
            "no_trade_days_pct": _avg(values, "no_trade_days_pct"),
            "unexplained_no_trade_days": sum(int(v.get("unexplained_no_trade_days", 0) or 0) for v in values),
            "max_drawdown": sum(float(v.get("max_drawdown", 0.0) or 0.0) for v in values),
            "max_drawdown_pct": sum(float(v.get("max_drawdown_pct", 0.0) or 0.0) for v in values),
            "max_single_trade_net_loss": min(float(v.get("max_single_trade_net_loss", 0.0) or 0.0) for v in values),
            "hard_loss_breach_count": sum(int(v.get("hard_loss_breach_count", 0) or 0) for v in values),
            "daily_loss_breach_count": sum(int(v.get("daily_loss_breach_count", 0) or 0) for v in values),
            "daily_bleed_halt_count": sum(int(v.get("daily_bleed_halt_count", 0) or 0) for v in values),
            "consecutive_losses_max": max(int(v.get("consecutive_losses_max", 0) or 0) for v in values),
            "effective_leverage_max": max(float(v.get("effective_leverage_max", 0.0) or 0.0) for v in values),
            "margin_used_pct_max": max(float(v.get("margin_used_pct_max", 0.0) or 0.0) for v in values),
            "rejected_by_effective_leverage_count": sum(int(v.get("rejected_by_effective_leverage_count", 0) or 0) for v in values),
            "rejected_by_margin_count": sum(int(v.get("rejected_by_margin_count", 0) or 0) for v in values),
            "profit_lock_move_count": sum(int(v.get("profit_lock_move_count", 0) or 0) for v in values),
            "profit_lock_saved_pnl": sum(float(v.get("profit_lock_saved_pnl", 0.0) or 0.0) for v in values),
            "winner_capture_ratio": _avg(values, "winner_capture_ratio"),
            "raw_signal_count": sum(int(v.get("raw_signal_count", 0) or 0) for v in values),
            "scored_signal_count": sum(int(v.get("scored_signal_count", 0) or 0) for v in values),
            "eligible_signal_count": sum(int(v.get("eligible_signal_count", 0) or 0) for v in values),
        }
    )
    reasons: Dict[str, int] = {}
    for value in values:
        for reason, count in dict(value.get("block_reason_distribution", {}) or {}).items():
            reasons[str(reason)] = reasons.get(str(reason), 0) + int(count or 0)
    out["block_reason_distribution"] = reasons
    return out


def _finite_pf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(out):
        return 999.0
    if math.isnan(out):
        return 0.0
    return out


def _unexplained_no_trade_days(source: Mapping[str, Any]) -> int:
    if int(source.get("executed_trade_count", 0) or 0) != 0:
        return 0
    return 1 if not dict(source.get("block_reasons", {}) or {}) else 0


def _avg(values: List[Mapping[str, Any]], key: str) -> float:
    nums = [float(v.get(key, 0.0) or 0.0) for v in values]
    return sum(nums) / len(nums) if nums else 0.0


def _weighted_avg(values: List[Mapping[str, Any]], key: str, weight_key: str) -> float:
    weighted = [(float(v.get(key, 0.0) or 0.0), float(v.get(weight_key, 0.0) or 0.0)) for v in values]
    total_weight = sum(w for _, w in weighted)
    if total_weight <= 0:
        return _avg(values, key)
    return sum(v * w for v, w in weighted) / total_weight
