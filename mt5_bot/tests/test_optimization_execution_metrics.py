from core.optimization.execution_metrics import aggregate_metrics, metrics_from_backtest


def test_metrics_from_backtest_maps_required_values() -> None:
    result = {
        "metrics": {"total_net_pnl": 2.0, "profit_factor": 1.5, "executed_trade_count": 2, "max_single_trade_loss": -1.0},
        "trades": [
            {"gross_pnl": 3.0, "cost_usd": 0.5, "net_pnl": 2.5},
            {"gross_pnl": -1.0, "cost_usd": 0.5, "net_pnl": -1.0},
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
    assert metrics["total_cost"] == 1.0
    assert metrics["hard_loss_breach_count"] == 0


def test_aggregate_metrics_combines_counts_and_caps() -> None:
    agg = aggregate_metrics(
        [
            {"total_net_pnl": 1, "total_trades": 1, "effective_leverage_max": 1, "margin_used_pct_max": 2},
            {"total_net_pnl": 2, "total_trades": 2, "effective_leverage_max": 3, "margin_used_pct_max": 1},
        ]
    )

    assert agg["total_net_pnl"] == 3
    assert agg["total_trades"] == 3
    assert agg["effective_leverage_max"] == 3

