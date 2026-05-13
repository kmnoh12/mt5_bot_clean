from core.optimization.pareto_ranker import pareto_frontier, select_recommendations
from core.optimization.trial_result import TrialResult


def _trial(trial_id: int, pnl: float, dd: float, trades: int) -> TrialResult:
    t = TrialResult(
        trial_id,
        1,
        {"risk": {"max_effective_leverage": 2, "max_margin_used_pct": 10}},
        {
            "total_net_pnl": pnl,
            "net_profit_factor": 1.5,
            "max_drawdown_pct": dd,
            "no_trade_days_pct": 5,
            "max_single_trade_net_loss": -1,
            "oos_decay_pct": 10,
            "median_trades_per_day": trades,
            "winner_capture_ratio": 0.5,
        },
    )
    t.robust_score = pnl - dd
    return t


def test_pareto_frontier_and_recommendations() -> None:
    trials = [_trial(1, 10, 5, 3), _trial(2, 12, 10, 5), _trial(3, 8, 2, 2)]

    frontier = pareto_frontier(trials)
    recs = select_recommendations(trials)

    assert frontier
    assert recs["aggressive"] is not None
    assert recs["balanced"] is not None
    assert recs["conservative"] is not None

