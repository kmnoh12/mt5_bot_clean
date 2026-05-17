import pytest

from core.optimization.execution_metrics import aggregate_metrics, metrics_from_backtest


def test_metrics_from_backtest_maps_required_values() -> None:
    result = {
        "metrics": {"total_net_pnl": 2.0, "profit_factor": 1.5, "executed_trade_count": 2, "max_single_trade_loss": -1.0},
        "trades": [
            {
                "gross_pnl": 3.0,
                "cost_usd": 0.5,
                "spread_cost_usd": 0.2,
                "slippage_cost_usd": 0.1,
                "commission_cost_usd": 0.2,
                "net_pnl": 2.5,
            },
            {
                "gross_pnl": 1.0,
                "cost_usd": 1.5,
                "spread_cost_usd": 0.3,
                "slippage_cost_usd": 0.4,
                "commission_cost_usd": 0.8,
                "net_pnl": -0.5,
                "exit_time": "2026-01-02T12:00:00+00:00",
            },
        ],
    }

    metrics = metrics_from_backtest(
        result,
        hard_max_net_loss_usd=1.25,
        max_daily_loss_usd=3.0,
        max_effective_leverage=2.0,
        max_margin_used_pct=10.0,
        leverage_metrics={"effective_leverage_max": 1.0, "margin_used_pct_max": 2.0},
    )

    assert metrics["total_trades"] == 2
    assert metrics["total_cost"] == 2.0
    assert metrics["signal_gross_expectancy"] == 2.0
    assert metrics["cost_drag_per_trade"] == 1.0
    assert metrics["cost_drag_to_gross_ratio"] == 0.5
    assert metrics["spread_cost"] == pytest.approx(0.5)
    assert metrics["slippage_cost"] == pytest.approx(0.5)
    assert metrics["commission_cost"] == pytest.approx(1.0)
    assert metrics["implementation_shortfall_cost"] == pytest.approx(0.5)
    assert metrics["implementation_shortfall_per_trade"] == pytest.approx(0.25)
    assert metrics["implementation_shortfall_to_gross_ratio"] == pytest.approx(0.125)
    assert metrics["net_to_gross_expectancy_ratio"] == 0.5
    assert metrics["gross_positive_trade_count"] == 2
    assert metrics["gross_positive_net_nonpositive_count"] == 1
    assert metrics["gross_positive_net_nonpositive_rate"] == 0.5
    assert metrics["trading_day_count"] == 1
    assert metrics["trade_dates"] == ["2026-01-02"]
    assert metrics["hard_loss_breach_count"] == 0


def test_aggregate_metrics_combines_counts_and_caps() -> None:
    agg = aggregate_metrics(
        [
            {"total_net_pnl": 1, "total_trades": 1, "effective_leverage_max": 1, "margin_used_pct_max": 2},
            {
                "total_net_pnl": 2,
                "gross_pnl": 5,
                "total_cost": 3,
                "implementation_shortfall_cost": 1,
                "total_trades": 2,
                "effective_leverage_max": 3,
                "margin_used_pct_max": 1,
                "spread_cost": 1,
                "slippage_cost": 1,
                "commission_cost": 1,
                "gross_positive_trade_count": 2,
                "gross_positive_net_nonpositive_count": 1,
            },
        ]
    )

    assert agg["total_net_pnl"] == 3
    assert agg["total_trades"] == 3
    assert agg["effective_leverage_max"] == 3
    assert agg["signal_gross_expectancy"] == pytest.approx(5 / 3)
    assert agg["cost_drag_per_trade"] == 1.0
    assert agg["implementation_shortfall_per_trade"] == pytest.approx(1 / 3)
    assert agg["implementation_shortfall_to_gross_ratio"] == pytest.approx(0.2)
    assert agg["spread_cost"] == 1
    assert agg["slippage_cost"] == 1
    assert agg["commission_cost"] == 1
    assert agg["gross_positive_net_nonpositive_rate"] == 0.5


def test_aggregate_metrics_unions_trade_dates() -> None:
    agg = aggregate_metrics(
        [
            {"total_net_pnl": 1, "total_trades": 1, "trade_dates": ["2026-01-01", "2026-01-02"]},
            {"total_net_pnl": 2, "total_trades": 2, "trade_dates": ["2026-01-02", "2026-01-03"]},
        ]
    )

    assert agg["trading_day_count"] == 3
    assert agg["trade_dates"] == ["2026-01-01", "2026-01-02", "2026-01-03"]
