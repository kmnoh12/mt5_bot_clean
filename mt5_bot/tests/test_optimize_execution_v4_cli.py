import json
from pathlib import Path

import pandas as pd

from tools.optimize_execution_v4 import main


def _write_csv(path: Path, periods: int = 80) -> None:
    rows = []
    for idx, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="min")):
        close = 100.0 + ((idx % 10) * 0.1)
        rows.append({"time": timestamp.isoformat(), "open": close - 0.02, "high": close + 0.8, "low": close - 0.8, "close": close})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_optimize_execution_v4_cli_writes_reports(tmp_path: Path) -> None:
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
    assert (out_dir / "optimization_v4_trials.csv").exists()

