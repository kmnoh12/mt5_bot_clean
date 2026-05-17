import json
from pathlib import Path

import pytest

from core.optimization.trial_result import TrialResult
from tools.optimize_execution_v4 import (
    _oos_direction_coverage_metrics,
    _promotion_gate_payload,
    _symbol_oos_coverage_metrics,
    main,
)


def _write_csv(path: Path, periods: int = 80) -> None:
    from datetime import datetime, timedelta, timezone

    rows = ["time,open,high,low,close"]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for idx in range(periods):
        timestamp = start + timedelta(minutes=idx)
        close = 100.0 + ((idx % 10) * 0.1)
        rows.append(f"{timestamp.isoformat()},{close - 0.02},{close + 0.8},{close - 0.8},{close}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_optimize_execution_v4_cli_writes_reports(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "reports"
    data_dir.mkdir()
    _write_csv(data_dir / "BTCUSD_TIMEFRAME_M1.csv")

    code = main([
        "--trials", "2",
        "--seed", "7",
        "--mode", "random",
        "--symbols", "BTCUSD",
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--max-rows", "80",
    ])

    assert code == 0
    summary = json.loads((out_dir / "optimization_v4_summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["trial_count"] == 2
    assert (out_dir / "recommended_configs_v4.md").exists()
    assert (out_dir / "recommended_configs_v4_promotion_gate.json").exists()
    assert (out_dir / "optimization_v4_trials.csv").exists()


def test_symbol_oos_coverage_metrics_exposes_weakest_symbol() -> None:
    metrics = _symbol_oos_coverage_metrics(
        {
            "BTCUSD": {"oos": {"total_trades": 42, "trading_day_count": 3}},
            "ETHUSD": {"oos": {"total_trades": 0, "trading_day_count": 0}},
        }
    )

    assert metrics["oos_symbol_count"] == 2
    assert metrics["oos_min_symbol_trades"] == 0
    assert metrics["oos_min_symbol_trading_days"] == 0
    assert metrics["oos_symbols_with_no_trades"] == ["ETHUSD"]


def test_oos_direction_coverage_metrics_exposes_one_sided_oos() -> None:
    metrics = _oos_direction_coverage_metrics(
        {
            "BTCUSD": [{"direction": "long"}, {"direction": "BUY"}],
            "ETHUSD": [{"direction": "short"}],
        }
    )

    assert metrics["oos_direction_trade_counts"] == {"long": 2, "short": 1}
    assert metrics["oos_symbol_direction_trade_counts"]["BTCUSD"] == {"long": 2, "short": 0}
    assert metrics["oos_min_direction_trades"] == 1
    assert metrics["oos_single_direction_only"] is False
    assert metrics["oos_direction_imbalance_pct"] == 2 / 3 * 100.0


def test_promotion_gate_blocks_optimizer_recommendation_until_walk_forward_and_shadow() -> None:
    class Args:
        min_oos_trades = 100
        min_oos_trading_days = 3
        min_oos_trades_per_symbol = 20
        min_oos_trading_days_per_symbol = 1
        min_oos_trades_per_direction = 5
        min_oos_trade_share_pct = 20.0
        min_oos_profit_factor = 1.05
        min_oos_expectancy_net = 0.0

    trial = TrialResult(
        trial_id=11,
        seed=7,
        config={},
        metrics={
            "oos_total_trades": 140,
            "oos_trading_day_count": 4,
            "oos_trade_share_pct": 30.0,
            "oos_min_symbol_trades": 22,
            "oos_min_direction_trades": 9,
        },
    )

    payload = _promotion_gate_payload({"balanced": trial}, Args())

    assert payload["live_promotion_allowed"] is False
    assert payload["status"] == "blocked"
    assert payload["block_reasons"] == ["walk_forward_stage_missing", "shadow_stage_missing"]
    balanced = payload["recommendations"]["balanced"]
    assert balanced["trial_id"] == 11
    assert balanced["live_promotion_allowed"] is False
    assert balanced["block_reasons"] == ["walk_forward_stage_missing", "shadow_stage_missing"]
    assert balanced["observed"]["oos_total_trades"] == 140
