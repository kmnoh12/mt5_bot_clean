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


def test_hard_reject_blocks_tiny_oos_trade_share() -> None:
    metrics = {
        "max_single_trade_net_loss": -0.5,
        "hard_loss_breach_count": 0,
        "daily_loss_breach_count": 0,
        "no_trade_days_pct": 10,
        "unexplained_no_trade_days": 0,
        "oos_total_trades": 5,
        "oos_trade_share_pct": 5.0,
        "oos_net_profit_factor": 2.0,
        "oos_expectancy_net": 0.2,
        "expectancy_net": 0.1,
        "net_profit_factor": 1.2,
        "effective_leverage_max": 0.5,
        "margin_used_pct_max": 1.0,
        "total_trades": 95,
    }
    config = {
        **_config(),
        "validation": {
            "min_total_trades": 50,
            "min_oos_trades": 1,
            "min_oos_trade_share_pct": 20.0,
            "min_oos_profit_factor": 1.05,
        },
    }

    reasons = hard_reject_reasons(metrics, config)

    assert reasons == ["oos_trade_share_pct_lt_min"]


def test_hard_reject_blocks_clustered_oos_trading_days() -> None:
    metrics = {
        "max_single_trade_net_loss": -0.5,
        "hard_loss_breach_count": 0,
        "daily_loss_breach_count": 0,
        "no_trade_days_pct": 10,
        "unexplained_no_trade_days": 0,
        "oos_total_trades": 120,
        "oos_trading_day_count": 1,
        "oos_trade_share_pct": 30.0,
        "oos_net_profit_factor": 2.0,
        "oos_expectancy_net": 0.2,
        "expectancy_net": 0.1,
        "net_profit_factor": 1.2,
        "effective_leverage_max": 0.5,
        "margin_used_pct_max": 1.0,
        "total_trades": 280,
    }
    config = {
        **_config(),
        "validation": {
            "min_total_trades": 50,
            "min_oos_trades": 100,
            "min_oos_trading_days": 3,
            "min_oos_trade_share_pct": 20.0,
            "min_oos_profit_factor": 1.05,
        },
    }

    reasons = hard_reject_reasons(metrics, config)

    assert reasons == ["oos_trading_days_lt_min"]


def test_hard_reject_blocks_symbol_without_oos_coverage() -> None:
    metrics = {
        "max_single_trade_net_loss": -0.5,
        "hard_loss_breach_count": 0,
        "daily_loss_breach_count": 0,
        "no_trade_days_pct": 10,
        "unexplained_no_trade_days": 0,
        "oos_total_trades": 140,
        "oos_trading_day_count": 4,
        "oos_min_symbol_trades": 0,
        "oos_min_symbol_trading_days": 0,
        "oos_trade_share_pct": 30.0,
        "oos_net_profit_factor": 2.0,
        "oos_expectancy_net": 0.2,
        "expectancy_net": 0.1,
        "net_profit_factor": 1.2,
        "effective_leverage_max": 0.5,
        "margin_used_pct_max": 1.0,
        "total_trades": 280,
    }
    config = {
        **_config(),
        "validation": {
            "min_total_trades": 50,
            "min_oos_trades": 100,
            "min_oos_trading_days": 3,
            "min_oos_trades_per_symbol": 20,
            "min_oos_trading_days_per_symbol": 1,
            "min_oos_trade_share_pct": 20.0,
            "min_oos_profit_factor": 1.05,
        },
    }

    reasons = hard_reject_reasons(metrics, config)

    assert reasons == ["oos_symbol_trades_lt_min", "oos_symbol_trading_days_lt_min"]


def test_hard_reject_blocks_one_sided_oos_when_direction_floor_is_set() -> None:
    metrics = {
        "max_single_trade_net_loss": -0.5,
        "hard_loss_breach_count": 0,
        "daily_loss_breach_count": 0,
        "no_trade_days_pct": 10,
        "unexplained_no_trade_days": 0,
        "oos_total_trades": 140,
        "oos_trading_day_count": 4,
        "oos_min_direction_trades": 0,
        "oos_trade_share_pct": 30.0,
        "oos_net_profit_factor": 2.0,
        "oos_expectancy_net": 0.2,
        "expectancy_net": 0.1,
        "net_profit_factor": 1.2,
        "effective_leverage_max": 0.5,
        "margin_used_pct_max": 1.0,
        "total_trades": 280,
    }
    config = {
        **_config(),
        "validation": {
            "min_total_trades": 50,
            "min_oos_trades": 100,
            "min_oos_trading_days": 3,
            "min_oos_trade_share_pct": 20.0,
            "min_oos_trades_per_direction": 5,
            "min_oos_profit_factor": 1.05,
        },
    }

    reasons = hard_reject_reasons(metrics, config)

    assert reasons == ["oos_direction_trades_lt_min"]


def test_score_trials_penalizes_rejected() -> None:
    good = TrialResult(1, 1, _config(), {"total_net_pnl": 10, "net_profit_factor": 2, "total_trades": 200, "oos_expectancy_net": 1})
    bad = TrialResult(2, 1, _config(), {"total_net_pnl": 999, "net_profit_factor": 3, "total_trades": 200}, rejected=True)

    scored = score_trials([bad, good])

    assert scored[0].trial_id == 1


def test_score_trials_penalizes_execution_drag_against_gross_edge() -> None:
    clean = TrialResult(
        1,
        1,
        _config(),
        {
            "total_net_pnl": 10,
            "net_profit_factor": 2,
            "total_trades": 200,
            "oos_expectancy_net": 1,
            "cost_drag_to_gross_ratio": 0.10,
            "implementation_shortfall_to_gross_ratio": 0.02,
            "gross_positive_net_nonpositive_rate": 0.0,
        },
    )
    dragged = TrialResult(
        2,
        1,
        _config(),
        {
            "total_net_pnl": 10,
            "net_profit_factor": 2,
            "total_trades": 200,
            "oos_expectancy_net": 1,
            "cost_drag_to_gross_ratio": 0.80,
            "implementation_shortfall_to_gross_ratio": 0.35,
            "gross_positive_net_nonpositive_rate": 0.50,
        },
    )

    scored = score_trials([dragged, clean])

    assert scored[0].trial_id == 1
    assert clean.robust_score > dragged.robust_score
