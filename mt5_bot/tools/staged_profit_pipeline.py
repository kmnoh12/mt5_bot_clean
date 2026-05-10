from __future__ import annotations

import json
import math
import shutil
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "pipeline_runs"
BASE_CONFIG = ROOT / "config.yaml"
DATA_DIR = ROOT / "data" / "aggressive_20260106_20260206"


@dataclass
class Scenario:
    name: str
    patch: Dict[str, Any]


def deep_update(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def compute_metrics(events_path: Path) -> Dict[str, Any]:
    trades: List[float] = []
    if not events_path.exists():
        return {"trades": 0}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "trade_ledger":
                continue
            pnl = row.get("realized_pnl")
            try:
                pnl_f = float(pnl)
            except (TypeError, ValueError):
                continue
            if math.isfinite(pnl_f):
                trades.append(pnl_f)

    if not trades:
        return {"trades": 0}

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(trades)
    avg_pnl = sum(trades) / len(trades)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    max_consecutive_loss = 0
    cur_consecutive_loss = 0
    for t in trades:
        equity += t
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
        if t < 0:
            cur_consecutive_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, cur_consecutive_loss)
        else:
            cur_consecutive_loss = 0

    return {
        "trades": len(trades),
        "net_pnl": round(sum(trades), 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": ("inf" if math.isinf(pf) else round(pf, 4)),
        "avg_pnl": round(avg_pnl, 4),
        "max_drawdown_abs": round(max_dd, 4),
        "max_consecutive_loss": int(max_consecutive_loss),
    }


def run_scenario(base: Dict[str, Any], scenario: Scenario) -> Dict[str, Any]:
    run_dir = RUN_ROOT / scenario.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(json.dumps(base))
    deep_update(cfg, scenario.patch)

    cfg_path = run_dir / "config.yaml"
    save_yaml(cfg_path, cfg)

    cmd = ["python", str(ROOT / "runner.py"), "--config", str(cfg_path), "--mode", "backtest"]
    try:
        proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired as exc:
        return {
            "scenario": scenario.name,
            "returncode": 124,
            "metrics": compute_metrics(run_dir / "events.jsonl"),
            "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-20:]),
            "timed_out": True,
        }
    metrics = compute_metrics(run_dir / "events.jsonl")
    return {
        "scenario": scenario.name,
        "returncode": proc.returncode,
        "metrics": metrics,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="Run only one scenario name")
    args = parser.parse_args()

    base = load_yaml(BASE_CONFIG)
    scenarios = [
        Scenario(
            name="baseline_gold_lsr_tick",
            patch={
                "backtest": {
                    "data_dir": str(DATA_DIR),
                    "initial_balance": 1000.0,
                    "spread_points": 35.0,
                    "commission_per_lot": 7.0,
                    "contract_size": 100.0,
                },
                "general": {"dry_run": False, "poll_seconds": 0.0},
                "required_active_symbols": ["GOLD"],
                "universe": [{"symbol": "GOLD", "strategy": "liquidity_sweep_reversal_tick", "timeframe": "TIMEFRAME_M1", "volume": 0.01}],
                "auto_tuning": {"enabled": False},
            },
        ),
        Scenario(
            name="abl_mtf_off",
            patch={"mtf_confirm": {"enabled": False}},
        ),
        Scenario(
            name="abl_churn_relaxed",
            patch={
                "execution_churn_guard": {
                    "max_entries_per_symbol_per_hour": 12,
                    "max_entries_per_symbol_per_day": 50,
                    "max_entries_global_per_day": 100,
                    "reentry_cooldown_seconds": 60,
                    "loss_reentry_lock_seconds": 30,
                }
            },
        ),
        Scenario(
            name="abl_cost_edge_off",
            patch={"cost_edge_guard": {"enabled": False}},
        ),
        Scenario(
            name="abl_lsr_looser",
            patch={
                "strategies": {
                    "liquidity_sweep_reversal": {
                        "displacement_mult": 1.15,
                        "sweep_buffer_atr": 0.007,
                        "reclaim_window_sec": 900,
                    }
                }
            },
        ),
    ]

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    selected = scenarios
    if args.only:
        selected = [s for s in scenarios if s.name == args.only]
        if not selected:
            raise SystemExit(f"Unknown scenario: {args.only}")

    results = [run_scenario(base, s) for s in selected]
    out = {"generated_by": "staged_profit_pipeline.py", "results": results}
    out_path = RUN_ROOT / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    for r in results:
        print(f"- {r['scenario']} rc={r['returncode']} metrics={r['metrics']}")


if __name__ == "__main__":
    main()
