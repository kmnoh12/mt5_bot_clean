from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.run_execution_style_backtest import main


def _write_csv(path: Path) -> None:
    rows = []
    for idx, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="min")):
        close = 100.0 + ((idx % 5) * 0.05)
        rows.append({"time": timestamp.isoformat(), "open": close - 0.02, "high": 101.0, "low": 99.0, "close": close, "volume": 1.0})
    rows.append({"time": pd.Timestamp("2026-01-01T00:25:00Z").isoformat(), "open": 98.9, "high": 101.2, "low": 98.55, "close": 99.35, "volume": 1.0})
    rows.append({"time": pd.Timestamp("2026-01-01T00:26:00Z").isoformat(), "open": 99.35, "high": 106.0, "low": 99.2, "close": 105.0, "volume": 1.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_execution_style_backtest_cli_writes_combined_reports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "reports"
    data_dir.mkdir()
    _write_csv(data_dir / "BTCUSD_TIMEFRAME_M1.csv")

    code = main([
        "--data-dir", str(data_dir),
        "--symbols", "BTCUSD",
        "--output-dir", str(out_dir),
        "--max-rows", "0",
        "--min-rr", "1.0",
        "--stress",
        "--stress-max-rows", "10",
    ])

    assert code == 0
    summary_path = out_dir / "execution_style_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["completed_symbol_count"] == 1
    assert (out_dir / "BTCUSD" / "execution_style_backtest.json").exists()
    assert (out_dir / "BTCUSD" / "cost_stress_execution_style.json").exists()
    assert summary["symbols"]["BTCUSD"]["stress"]["rows_used"] == 10
