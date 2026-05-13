import pandas as pd

from core.optimization.oos_validator import attach_oos_metrics, train_oos_split


def test_train_oos_split_and_decay_metrics() -> None:
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=10, freq="min"), "open": range(10)})
    train, oos = train_oos_split(frame, 0.6)

    merged = attach_oos_metrics(
        {"expectancy_net": 2.0, "total_net_pnl": 10, "net_profit_factor": 2, "max_drawdown_pct": 1, "no_trade_days_pct": 0},
        {"expectancy_net": 1.0, "total_trades": 120, "net_profit_factor": 1.2, "total_net_pnl": 5},
    )

    assert len(train) == 6
    assert len(oos) == 4
    assert merged["oos_decay_pct"] == 50.0
    assert merged["oos_total_trades"] == 120

