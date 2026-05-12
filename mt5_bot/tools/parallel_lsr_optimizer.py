from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.backtest_sim import _coerce_candidate_params, _simulate_symbol  # type: ignore


PARAM_SPACE: Dict[str, List[Any]] = {
    "atr_period": [7, 10, 14, 21, 28],
    "pivot_lookback_sec": [600, 900, 1200, 1800, 2700, 3600],
    "swing_window": [20, 30, 45, 60, 90, 120],
    "sweep_buffer_atr": [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35],
    "reclaim_buffer_atr": [0.0, 0.02, 0.04, 0.06, 0.10],
    "reclaim_window_sec": [30, 60, 120, 240, 480, 900, 1500],
    "displacement_mult": [1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0],
    "displacement_lookback": [8, 12, 20, 32, 48],
    "sl_atr_mult": [0.4, 0.55, 0.7, 0.85, 1.0, 1.25, 1.5],
    "stop_buffer_atr": [0.0, 0.03, 0.06, 0.1, 0.16],
    "tp_R1": [0.8, 1.0, 1.2, 1.5, 1.8, 2.2],
    "tp_R2": [1.6, 2.0, 2.5, 3.0, 3.8, 5.0],
    "be_at_R": [0.6, 0.8, 1.0, 1.2, 1.5],
    "max_hold_bars": [40, 80, 120, 180, 300, 480],
    "min_hold_bars": [1, 2, 3, 5, 8],
    "min_cooldown_bars": [2, 5, 8, 13, 21, 34],
    "fvg_enabled": [False],
    "retest_enabled": [False],
    "trail_tp_enabled": [False],
}

BASE_PARAMS: Dict[str, Any] = {
    "enabled": True,
    "atr_period": 14,
    "pivot_lookback_sec": 1800,
    "swing_window": 60,
    "sweep_buffer_atr": 0.25,
    "reclaim_buffer_atr": 0.05,
    "reclaim_window_sec": 120,
    "displacement_mult": 1.55,
    "displacement_lookback": 20,
    "sl_atr_mult": 0.95,
    "stop_buffer_atr": 0.08,
    "tp_R1": 1.3,
    "tp_R2": 2.8,
    "be_at_R": 1.1,
    "max_hold_bars": 180,
    "min_hold_bars": 2,
    "min_cooldown_bars": 8,
    "fvg_enabled": False,
    "retest_enabled": False,
    "trail_tp_enabled": False,
}


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return df


def make_folds(df: pd.DataFrame, window_rows: int, folds: int, stride_rows: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    end = len(df)
    while len(out) < folds and end - window_rows >= 0:
        out.append((end - window_rows, end))
        end -= stride_rows
    return list(reversed(out))


def rand_params(rng: random.Random) -> Dict[str, Any]:
    params = dict(BASE_PARAMS)
    for key, values in PARAM_SPACE.items():
        params[key] = rng.choice(values)
    if float(params["tp_R2"]) <= float(params["tp_R1"]):
        params["tp_R2"] = float(params["tp_R1"]) + rng.choice([0.5, 1.0, 1.5, 2.0])
    return _coerce_candidate_params(params)


def mutate(seed_params: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    params = dict(seed_params)
    keys = list(PARAM_SPACE.keys())
    for key in rng.sample(keys, k=rng.randint(2, 6)):
        params[key] = rng.choice(PARAM_SPACE[key])
    if float(params["tp_R2"]) <= float(params["tp_R1"]):
        params["tp_R2"] = float(params["tp_R1"]) + 1.0
    return _coerce_candidate_params(params)


def summarize(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(metrics)
    if n == 0:
        return {"folds": 0, "trades": 0, "total_r": 0.0, "score": -9999.0}
    trades = sum(int(m.get("trades", 0)) for m in metrics)
    total_r = sum(float(m.get("total_r", 0.0)) for m in metrics)
    dd = max(float(m.get("max_drawdown_r", 0.0)) for m in metrics)
    exp_vals = [float(m.get("expectancy_r", 0.0)) for m in metrics]
    pf_vals = [min(10.0, float(m.get("profit_factor", 0.0))) for m in metrics]
    win_vals = [float(m.get("win_rate", 0.0)) for m in metrics]
    pos_folds = sum(1 for m in metrics if float(m.get("total_r", 0.0)) > 0)
    # robust objective: don't accept one lucky fold; penalize drawdown and no-trade candidates.
    score = total_r + (sum(exp_vals) / n) * 8.0 + (sum(pf_vals) / n) * 0.8 + pos_folds * 1.2 - dd * 1.5
    if trades < max(5, n):
        score -= (max(5, n) - trades) * 2.0
    return {
        "folds": n,
        "trades": int(trades),
        "total_r": round(total_r, 6),
        "avg_expectancy_r": round(sum(exp_vals) / n, 6),
        "avg_profit_factor": round(sum(pf_vals) / n, 6),
        "avg_win_rate": round(sum(win_vals) / n, 6),
        "max_drawdown_r": round(dd, 6),
        "positive_folds": int(pos_folds),
        "score": round(score, 6),
    }


_WORKER_FOLDS: List[pd.DataFrame] = []
_WORKER_SPLIT_AT = 0


def _init_worker(csv_path: str, folds: List[Tuple[int, int]], split_at: int) -> None:
    global _WORKER_FOLDS, _WORKER_SPLIT_AT
    df = load_frame(Path(csv_path))
    _WORKER_FOLDS = [df.iloc[s:e].copy().reset_index(drop=True) for s, e in folds]
    _WORKER_SPLIT_AT = int(split_at)


def eval_candidate(task: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    cid, params = task
    train_metrics = []
    test_metrics = []
    for idx, frame in enumerate(_WORKER_FOLDS):
        m = _simulate_symbol("BTCUSD", frame, params)
        if idx < _WORKER_SPLIT_AT:
            train_metrics.append(m)
        else:
            test_metrics.append(m)
    train = summarize(train_metrics)
    test = summarize(test_metrics)
    # rank primarily by OOS/test, then train stability
    rank = float(test["score"]) + 0.25 * float(train["score"])
    if int(test["trades"]) < 3:
        rank -= 20.0
    return {"id": cid, "rank": round(rank, 6), "params": params, "train": train, "test": test}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/binance/BTCUSD_TIMEFRAME_M1.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=1600)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--window-rows", type=int, default=6000)
    ap.add_argument("--stride-rows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260513)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "optimization_runs/parallel_lsr_btcusd_summary.json"))
    args = ap.parse_args()

    df = load_frame(Path(args.csv))
    folds = make_folds(df, int(args.window_rows), int(args.folds), int(args.stride_rows))
    if len(folds) < 3:
        raise SystemExit(f"not enough data for folds: rows={len(df)} folds={len(folds)}")
    split_at = max(1, int(len(folds) * 0.7))

    rng = random.Random(int(args.seed))
    candidates: List[Dict[str, Any]] = [_coerce_candidate_params(BASE_PARAMS)]
    while len(candidates) < int(args.candidates):
        if len(candidates) < int(args.candidates) // 2:
            candidates.append(rand_params(rng))
        else:
            candidates.append(mutate(rng.choice(candidates[: max(1, len(candidates)//3)]), rng))

    tasks = [(i, p) for i, p in enumerate(candidates)]
    results: List[Dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    print(f"start candidates={len(tasks)} workers={args.workers} folds={len(folds)} split_at={split_at} rows={len(df)}", flush=True)
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_worker,
        initargs=(str(args.csv), folds, split_at),
    ) as ex:
        futs = [ex.submit(eval_candidate, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            if n % 50 == 0:
                best = max(results, key=lambda r: float(r["rank"]))
                print(f"done={n}/{len(tasks)} best_rank={best['rank']} test={best['test']}", flush=True)
    results.sort(key=lambda r: float(r["rank"]), reverse=True)
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "csv": str(Path(args.csv).resolve()),
        "rows": int(len(df)),
        "folds": [{"start": s, "end": e} for s, e in folds],
        "split_at": int(split_at),
        "workers": int(args.workers),
        "candidates": int(len(tasks)),
        "top": results[: int(args.top)],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={out_path}")
    print(json.dumps(out["top"][:5], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
