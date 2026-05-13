from core.optimization.objective_v4 import hard_reject_reasons, score_trials
from core.optimization.trial_result import TrialResult


def _config():
    return {"risk": {"hard_max_net_loss_usd": 1.25, "max_effective_leverage": 1.0, "max_margin_used_pct": 5}}


def test_hard_reject_blocks_loss_and_oos_failures() -> None:
    metrics = {
        "max_single_trade_net_loss": -2.0,
        "hard_loss_breach_count": 1,
        "daily_loss_breach_count": 0,
        "no_trade_days_pct": 10,
        "unexplained_no_trade_days": 0,
        "oos_total_trades": 120,
        "oos_net_profit_factor": 1.2,
        "oos_expectancy_net": 0.1,
        "expectancy_net": 0.1,
        "effective_leverage_max": 0.5,
        "margin_used_pct_max": 1.0,
        "total_trades": 120,
    }

    reasons = hard_reject_reasons(metrics, _config())

    assert "max_single_trade_net_loss_exceeds_hard_max" in reasons
    assert "hard_loss_breach" in reasons


def test_score_trials_penalizes_rejected() -> None:
    good = TrialResult(1, 1, _config(), {"total_net_pnl": 10, "net_profit_factor": 2, "total_trades": 200, "oos_expectancy_net": 1})
    bad = TrialResult(2, 1, _config(), {"total_net_pnl": 999, "net_profit_factor": 3, "total_trades": 200}, rejected=True)

    scored = score_trials([bad, good])

    assert scored[0].trial_id == 1

