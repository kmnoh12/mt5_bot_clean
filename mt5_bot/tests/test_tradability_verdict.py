from __future__ import annotations

from utils.execution_style_backtest import build_tradability_verdict


def test_tradability_verdict_kills_unexplained_zero_trade() -> None:
    verdict = build_tradability_verdict(
        {
            "executed_trade_count": 0,
            "median_trades_per_week": 0,
            "no_trade_days_pct": 100.0,
            "max_single_trade_loss": 0.0,
            "daily_max_loss": 0.0,
            "profit_factor": 0.0,
            "block_reasons": {},
        }
    )

    assert verdict["verdict"] == "KILL"
    assert "unexplained_zero_trade_period" in verdict["failures"]


def test_tradability_verdict_flags_hard_max_loss() -> None:
    verdict = build_tradability_verdict(
        {
            "executed_trade_count": 250,
            "median_trades_per_week": 10,
            "no_trade_days_pct": 10.0,
            "max_single_trade_loss": -10.0,
            "daily_max_loss": -2.0,
            "profit_factor": 1.4,
            "block_reasons": {"fee_adjusted_rr_too_low": 1},
        }
    )

    assert verdict["verdict"] == "REDESIGN"
    assert "hard_max_single_trade_loss_exceeded" in verdict["failures"]
