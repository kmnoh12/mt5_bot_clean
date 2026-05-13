from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from core.optimization.trial_result import TrialResult


def hard_reject_reasons(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> List[str]:
    risk = dict(config.get("risk", {}) or {})
    validation = dict(config.get("validation", {}) or {})
    hard_max = float(risk.get("hard_max_net_loss_usd", 1.25) or 1.25)
    max_eff = float(risk.get("max_effective_leverage", 1.0) or 1.0)
    max_margin = float(risk.get("max_margin_used_pct", 5.0) or 5.0)
    min_total_trades = int(validation.get("min_total_trades", 100) or 100)
    min_oos_trades = int(validation.get("min_oos_trades", 100) or 100)
    min_oos_profit_factor = float(validation.get("min_oos_profit_factor", 1.05) or 1.05)
    min_oos_expectancy = float(validation.get("min_oos_expectancy_net", 0.0) or 0.0)
    max_no_trade_days_pct = float(validation.get("max_no_trade_days_pct", 60.0) or 60.0)
    min_train_profit_factor = float(validation.get("min_train_profit_factor", 0.0) or 0.0)
    require_train_net_pnl_positive = bool(validation.get("require_train_net_pnl_positive", False))
    reasons: List[str] = []
    if float(metrics.get("max_single_trade_net_loss", 0.0) or 0.0) < -hard_max - 1e-9:
        reasons.append("max_single_trade_net_loss_exceeds_hard_max")
    if int(metrics.get("hard_loss_breach_count", 0) or 0) > 0:
        reasons.append("hard_loss_breach")
    if int(metrics.get("daily_loss_breach_count", 0) or 0) > 0:
        reasons.append("daily_loss_breach")
    if float(metrics.get("no_trade_days_pct", 0.0) or 0.0) > max_no_trade_days_pct:
        reasons.append("no_trade_days_pct_gt_limit")
    if int(metrics.get("unexplained_no_trade_days", 0) or 0) >= 2:
        reasons.append("unexplained_no_trade_days")
    if int(metrics.get("oos_total_trades", metrics.get("total_trades", 0)) or 0) < min_oos_trades:
        reasons.append("oos_total_trades_lt_min")
    if float(metrics.get("oos_net_profit_factor", metrics.get("net_profit_factor", 0.0)) or 0.0) < min_oos_profit_factor:
        reasons.append("oos_net_profit_factor_lt_min")
    if float(metrics.get("oos_expectancy_net", metrics.get("expectancy_net", 0.0)) or 0.0) <= min_oos_expectancy:
        reasons.append("oos_expectancy_net_lte_min")
    if float(metrics.get("expectancy_net", 0.0) or 0.0) <= 0.0:
        reasons.append("cost_adjusted_expectancy_lte_0")
    if require_train_net_pnl_positive and float(metrics.get("total_net_pnl", 0.0) or 0.0) < 0.0:
        reasons.append("train_total_net_pnl_lt_0")
    if float(metrics.get("net_profit_factor", 0.0) or 0.0) <= min_train_profit_factor:
        reasons.append("train_net_profit_factor_lte_min")
    if float(metrics.get("effective_leverage_max", 0.0) or 0.0) > max_eff + 1e-12:
        reasons.append("effective_leverage_cap_exceeded")
    if float(metrics.get("margin_used_pct_max", 0.0) or 0.0) > max_margin + 1e-12:
        reasons.append("margin_cap_exceeded")
    if int(metrics.get("total_trades", 0) or 0) < min_total_trades:
        reasons.append("total_trades_lt_min")
    return reasons


def score_trials(trials: Iterable[TrialResult]) -> List[TrialResult]:
    items = list(trials)
    fields = {
        "total_net_pnl": 1.0,
        "net_profit_factor": 0.8,
        "winner_capture_ratio": 0.5,
        "median_trades_per_day": 0.3,
        "oos_expectancy_net": 0.3,
        "max_drawdown_pct": -1.5,
        "oos_decay_pct": -0.7,
        "margin_used_pct_max": -0.5,
    }
    ranges = {field: _range([float(t.metrics.get(field, 0.0) or 0.0) for t in items]) for field in fields}
    for trial in items:
        score = 0.0
        for field, weight in fields.items():
            score += weight * _normalize(float(trial.metrics.get(field, 0.0) or 0.0), *ranges[field])
        score -= 2.0 if int(trial.metrics.get("hard_loss_breach_count", 0) or 0) else 0.0
        score -= min(1.0, float(trial.metrics.get("daily_bleed_halt_count", 0) or 0) / 5.0)
        score -= min(1.0, float(trial.metrics.get("no_trade_days_pct", 0.0) or 0.0) / 100.0)
        score -= min(0.5, max(0.0, float(trial.metrics.get("total_trades", 0) or 0) - 800.0) / 1600.0)
        if trial.rejected:
            score -= 100.0
        trial.robust_score = score
    return sorted(items, key=lambda t: t.robust_score, reverse=True)


def _range(values: List[float]) -> tuple[float, float]:
    return (min(values), max(values)) if values else (0.0, 0.0)


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return (value - low) / (high - low)
