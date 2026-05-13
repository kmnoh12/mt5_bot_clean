from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from core.optimization.trial_result import TrialResult


PARETO_AXES: Dict[str, str] = {
    "total_net_pnl": "max",
    "net_profit_factor": "max",
    "max_drawdown_pct": "min",
    "no_trade_days_pct": "min",
    "max_single_trade_net_loss": "max",
    "oos_decay_pct": "min",
    "median_trades_per_day": "max",
    "winner_capture_ratio": "max",
}


def pareto_frontier(trials: Iterable[TrialResult]) -> List[TrialResult]:
    candidates = [t for t in trials if not t.rejected]
    frontier: List[TrialResult] = []
    for trial in candidates:
        if not any(_dominates(other, trial) for other in candidates if other is not trial):
            frontier.append(trial)
    return sorted(frontier, key=lambda t: t.robust_score, reverse=True)


def select_recommendations(trials: Iterable[TrialResult]) -> Dict[str, Optional[TrialResult]]:
    candidates = [t for t in trials if not t.rejected]
    if not candidates:
        return {"aggressive": None, "balanced": None, "conservative": None}
    aggressive = max(candidates, key=lambda t: (t.metrics.get("total_net_pnl", 0.0), t.metrics.get("median_trades_per_day", 0.0), t.robust_score))
    remaining = [t for t in candidates if t is not aggressive] or candidates
    balanced = max(remaining, key=lambda t: t.robust_score)
    remaining = [t for t in candidates if t is not aggressive and t is not balanced] or [t for t in candidates if t is not aggressive] or candidates
    conservative = max(
        remaining,
        key=lambda t: (
            -float(t.metrics.get("max_drawdown_pct", 0.0) or 0.0),
            -float(t.metrics.get("max_single_trade_net_loss", 0.0) or 0.0),
            float(t.metrics.get("oos_net_profit_factor", 0.0) or 0.0),
            t.robust_score,
        ),
    )
    aggressive.rank_bucket = "aggressive"
    balanced.rank_bucket = "balanced"
    conservative.rank_bucket = "conservative"
    return {"aggressive": aggressive, "balanced": balanced, "conservative": conservative}


def _dominates(left: TrialResult, right: TrialResult) -> bool:
    better_or_equal = True
    strictly_better = False
    for field, direction in PARETO_AXES.items():
        a = float(left.metrics.get(field, 0.0) or 0.0)
        b = float(right.metrics.get(field, 0.0) or 0.0)
        if direction == "max":
            if a < b:
                better_or_equal = False
                break
            strictly_better = strictly_better or a > b
        else:
            if a > b:
                better_or_equal = False
                break
            strictly_better = strictly_better or a < b
    return better_or_equal and strictly_better
