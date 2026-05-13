from __future__ import annotations

import pandas as pd

from utils.execution_style_backtest import ExecutionStyleBacktestConfig, run_cost_stress_scenarios


def _bars() -> pd.DataFrame:
    rows = []
    for idx, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="min")):
        close = 100.0 + ((idx % 5) * 0.05)
        rows.append({"time": timestamp, "open": close - 0.02, "high": 101.0, "low": 99.0, "close": close})
    rows.append({"time": pd.Timestamp("2026-01-01T00:20:00Z"), "open": 98.9, "high": 101.2, "low": 98.55, "close": 99.35})
    rows.append({"time": pd.Timestamp("2026-01-01T00:21:00Z"), "open": 99.35, "high": 106.0, "low": 99.2, "close": 105.0})
    return pd.DataFrame(rows)


def test_cost_stress_execution_style_reports_32_scenarios() -> None:
    result = run_cost_stress_scenarios(
        _bars(),
        ExecutionStyleBacktestConfig(
            symbol="XAUUSD",
            min_reward_to_net_risk_ratio=1.0,
            scanner_atr_period=5,
            tick_size=0.01,
            tick_value=0.1,
            spread_points=1.0,
            commission_per_lot=0.1,
        ),
    )

    assert result["scenario_count"] == 32
    assert all("trade_count" in item for item in result["scenarios"])
    assert all("max_single_trade_loss" in item for item in result["scenarios"])
