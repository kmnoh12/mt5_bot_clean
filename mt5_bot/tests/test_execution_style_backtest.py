from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.execution_style_backtest import (
    ExecutionStyleBacktestConfig,
    build_tradability_verdict,
    run_cost_stress_scenarios,
    run_execution_style_backtest,
    write_execution_style_reports,
)


def _base_rows(periods: int = 20, base: float = 100.0):
    rows = []
    for idx, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="min")):
        close = base + ((idx % 5) * 0.05)
        rows.append({"time": timestamp, "open": close - 0.02, "high": base + 1.0, "low": base - 1.0, "close": close})
    return rows


def _bars_with_profitable_long_setups(repeats: int = 8) -> pd.DataFrame:
    rows = _base_rows()
    ts = pd.Timestamp("2026-01-01T00:20:00Z")
    for cycle in range(repeats):
        rows.append({"time": ts, "open": 98.9, "high": 101.2, "low": 98.55, "close": 99.35})
        ts += pd.Timedelta(minutes=1)
        # move enough to trigger profit lock and TP extension on tiny contract values
        rows.append({"time": ts, "open": 99.35, "high": 106.0, "low": 99.2, "close": 105.0})
        ts += pd.Timedelta(minutes=1)
        rows.append({"time": ts, "open": 105.0, "high": 106.5, "low": 104.0, "close": 106.0})
        ts += pd.Timedelta(minutes=1)
        rows.extend(_base_rows(periods=20, base=100.0 + (cycle % 2) * 0.1))
        ts += pd.Timedelta(minutes=20)
    return pd.DataFrame(rows)


def _config(**overrides) -> ExecutionStyleBacktestConfig:
    values = {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "tick_size": 0.01,
        "tick_value": 0.1,
        "contract_size": 1.0,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "min_reward_to_net_risk_ratio": 1.5,
        "min_signal_score": 70.0,
        "spread_points": 0.0,
        "commission_per_lot": 0.0,
        "expected_slippage_points": 0.0,
        "scanner_atr_period": 5,
        "scanner_lookback_bars": 20,
        "max_bars_in_trade": 5,
    }
    values.update(overrides)
    return ExecutionStyleBacktestConfig(**values)


def test_execution_style_backtest_tracks_profit_lock_and_risk_metrics() -> None:
    result = run_execution_style_backtest(
        _bars_with_profitable_long_setups(),
        _config(spread_points=2.0, commission_per_lot=0.2, expected_slippage_points=1.0),
    )
    metrics = result["metrics"]

    assert metrics["raw_signal_count"] >= 1
    assert metrics["executed_trade_count"] >= 1
    assert metrics["max_single_trade_loss"] >= -1.25
    assert metrics["hard_max_respected"] is True
    assert "profit_lock_saved_pnl" in metrics
    assert metrics["spread_cost"] > 0
    assert metrics["slippage_cost"] > 0
    assert metrics["commission_cost"] > 0
    assert metrics["implementation_shortfall_cost"] == metrics["slippage_cost"]
    assert "no_trade_days_count" in metrics
    assert isinstance(result["trades"], list)
    assert {"spread_cost_usd", "slippage_cost_usd", "commission_cost_usd", "implementation_shortfall_usd"} <= set(
        result["trades"][0]
    )


def test_execution_style_backtest_blocks_min_lot_when_hard_max_exceeded() -> None:
    result = run_execution_style_backtest(
        _bars_with_profitable_long_setups(repeats=2),
        _config(volume_min=10.0, tick_value=10.0, min_reward_to_net_risk_ratio=1.0),
    )
    metrics = result["metrics"]

    assert metrics["executed_trade_count"] == 0
    assert metrics["block_reasons"].get("min_lot_risk_exceeds_hard_max", 0) >= 1
    assert metrics["no_trade_guard"]["zero_trade_success"] is False


def test_cost_stress_runs_32_scenarios() -> None:
    stress = run_cost_stress_scenarios(
        _bars_with_profitable_long_setups(repeats=2),
        _config(spread_points=1.0, commission_per_lot=0.1, min_reward_to_net_risk_ratio=1.0),
    )

    assert stress["scenario_count"] == 32
    assert len(stress["scenarios"]) == 32
    assert {"net_profit_factor", "trade_count", "no_trade_days", "max_single_trade_loss"} <= set(stress["scenarios"][0])


def test_tradability_verdict_reports_go_redesign_or_kill() -> None:
    verdict = build_tradability_verdict(
        {
            "executed_trade_count": 250,
            "median_trades_per_week": 10,
            "no_trade_days_pct": 10.0,
            "max_single_trade_loss": -1.0,
            "daily_max_loss": -2.0,
            "profit_factor": 1.2,
            "block_reasons": {"fee_adjusted_rr_too_low": 3},
        }
    )

    assert verdict["verdict"] == "GO"
    assert verdict["failures"] == []


def test_execution_style_reports_are_written(tmp_path: Path) -> None:
    result = run_execution_style_backtest(_bars_with_profitable_long_setups(repeats=1), _config(min_reward_to_net_risk_ratio=1.0))
    paths = write_execution_style_reports(result, tmp_path)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["verdict"].exists()
    assert "Execution Style Backtest" in paths["markdown"].read_text(encoding="utf-8")
