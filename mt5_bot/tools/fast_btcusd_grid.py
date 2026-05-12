from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SPACE = {
    "lookback": [30, 45, 60, 90, 120, 180, 240],
    "atr_period": [7, 10, 14, 21, 28, 42],
    "sweep_atr": [0.02, 0.04, 0.07, 0.10, 0.16, 0.24, 0.35],
    "reclaim_atr": [0.00, 0.02, 0.04, 0.07, 0.10, 0.16],
    "disp_atr": [0.05, 0.10, 0.18, 0.28, 0.42, 0.65],
    "sl_atr": [0.45, 0.60, 0.80, 1.00, 1.25, 1.60],
    "tp_r": [1.2, 1.6, 2.0, 2.6, 3.4, 4.5],
    "trail_start_r": [0.8, 1.0, 1.3, 1.7, 2.2],
    "trail_gap_r": [0.35, 0.50, 0.70, 1.00, 1.40],
    "max_hold": [60, 120, 240, 480, 720, 1440],
    "cooldown": [15, 30, 60, 120, 240, 480],
}

BASE = {
    "lookback": 120,
    "atr_period": 14,
    "sweep_atr": 0.10,
    "reclaim_atr": 0.04,
    "disp_atr": 0.18,
    "sl_atr": 0.80,
    "tp_r": 2.6,
    "trail_start_r": 1.3,
    "trail_gap_r": 0.7,
    "max_hold": 240,
    "cooldown": 60,
}

_DATA: Dict[str, Any] = {}


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)


def _init(path: str, max_rows: int) -> None:
    global _DATA
    df = _load_csv(Path(path))
    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atrs: Dict[int, np.ndarray] = {}
    for period in sorted(set(SPACE["atr_period"])):
        atrs[period] = pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy(float)
    roll_low: Dict[int, np.ndarray] = {}
    roll_high: Dict[int, np.ndarray] = {}
    for lb in sorted(set(SPACE["lookback"])):
        roll_low[lb] = pd.Series(l).rolling(lb, min_periods=lb).min().shift(1).to_numpy(float)
        roll_high[lb] = pd.Series(h).rolling(lb, min_periods=lb).max().shift(1).to_numpy(float)
    _DATA = {"open": o, "high": h, "low": l, "close": c, "atrs": atrs, "roll_low": roll_low, "roll_high": roll_high, "rows": len(df), "start": df['time'].iloc[0].isoformat(), "end": df['time'].iloc[-1].isoformat()}


def _candidate(rng: random.Random, seed: Dict[str, Any] | None = None) -> Dict[str, Any]:
    p = dict(BASE if seed is None else seed)
    keys = list(SPACE)
    if seed is None:
        for k in keys:
            p[k] = rng.choice(SPACE[k])
    else:
        for k in rng.sample(keys, rng.randint(2, 5)):
            p[k] = rng.choice(SPACE[k])
    return p


def _metrics(pnls: List[float], holds: List[int]) -> Dict[str, Any]:
    arr = np.array(pnls, dtype=float)
    trades = int(len(arr))
    if trades == 0:
        return {"trades": 0, "net_r": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "max_drawdown_r": 0.0, "avg_hold": 0.0}
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    dd = np.maximum(0.0, peak - eq)
    return {
        "trades": trades,
        "net_r": round(float(arr.sum()), 6),
        "expectancy_r": round(float(arr.mean()), 6),
        "profit_factor": round(float(gross_win / gross_loss), 6) if gross_loss > 1e-12 else (99.0 if gross_win > 0 else 0.0),
        "win_rate": round(float((arr > 0).mean()), 6),
        "max_drawdown_r": round(float(dd.max()) if len(dd) else 0.0, 6),
        "avg_hold": round(float(np.mean(holds)) if holds else 0.0, 3),
    }


def _score(m: Dict[str, Any]) -> float:
    trades = int(m.get("trades", 0))
    if trades < 6:
        return -1000.0 + trades
    return float(m["net_r"]) + float(m["expectancy_r"]) * 12.0 + min(5.0, float(m["profit_factor"])) * 1.5 + float(m["win_rate"]) * 2.0 - float(m["max_drawdown_r"]) * 1.8


def _simulate(params: Dict[str, Any], start: int, end: int, fee_r: float) -> Dict[str, Any]:
    h = _DATA["high"]; l = _DATA["low"]; c = _DATA["close"]
    atr = _DATA["atrs"][int(params["atr_period"])]
    rlow = _DATA["roll_low"][int(params["lookback"])]
    rhigh = _DATA["roll_high"][int(params["lookback"])]
    sweep_atr = float(params["sweep_atr"]); reclaim_atr = float(params["reclaim_atr"]); disp_atr = float(params["disp_atr"])
    sl_atr = float(params["sl_atr"]); tp_r = float(params["tp_r"]); trail_start = float(params["trail_start_r"]); trail_gap = float(params["trail_gap_r"])
    max_hold = int(params["max_hold"]); cooldown = int(params["cooldown"])
    pnls: List[float] = []; holds: List[int] = []
    pos = 0; entry = 0.0; sl = 0.0; tp = 0.0; risk = 0.0; bars_in = 0; cd = 0; best_r = 0.0
    i0 = max(start, int(params["lookback"]) + int(params["atr_period"]) + 2)
    for i in range(i0, end):
        if not math.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if cd > 0:
            cd -= 1
        if pos != 0:
            bars_in += 1
            if pos > 0:
                cur_r = (h[i] - entry) / risk
                best_r = max(best_r, cur_r)
                if best_r >= trail_start:
                    sl = max(sl, entry + (best_r - trail_gap) * risk)
                exit_price = None
                if l[i] <= sl: exit_price = sl
                elif h[i] >= tp: exit_price = tp
                elif bars_in >= max_hold: exit_price = c[i]
                if exit_price is not None:
                    pnls.append(((exit_price - entry) / risk) - fee_r); holds.append(bars_in); pos = 0; cd = cooldown
            else:
                cur_r = (entry - l[i]) / risk
                best_r = max(best_r, cur_r)
                if best_r >= trail_start:
                    sl = min(sl, entry - (best_r - trail_gap) * risk)
                exit_price = None
                if h[i] >= sl: exit_price = sl
                elif l[i] <= tp: exit_price = tp
                elif bars_in >= max_hold: exit_price = c[i]
                if exit_price is not None:
                    pnls.append(((entry - exit_price) / risk) - fee_r); holds.append(bars_in); pos = 0; cd = cooldown
            continue
        if cd > 0:
            continue
        # Long: sweep below previous range then reclaim above previous low with displacement.
        if math.isfinite(rlow[i]) and l[i] < rlow[i] - sweep_atr * atr[i] and c[i] > rlow[i] + reclaim_atr * atr[i] and (c[i] - l[i]) > disp_atr * atr[i]:
            entry = c[i]; risk = max(1e-9, sl_atr * atr[i]); sl = entry - risk; tp = entry + tp_r * risk; pos = 1; bars_in = 0; best_r = 0.0; continue
        # Short: sweep above previous range then reclaim below previous high with displacement.
        if math.isfinite(rhigh[i]) and h[i] > rhigh[i] + sweep_atr * atr[i] and c[i] < rhigh[i] - reclaim_atr * atr[i] and (h[i] - c[i]) > disp_atr * atr[i]:
            entry = c[i]; risk = max(1e-9, sl_atr * atr[i]); sl = entry + risk; tp = entry - tp_r * risk; pos = -1; bars_in = 0; best_r = 0.0; continue
    if pos != 0:
        if pos > 0: pnls.append(((c[end-1] - entry) / risk) - fee_r)
        else: pnls.append(((entry - c[end-1]) / risk) - fee_r)
        holds.append(bars_in)
    return _metrics(pnls, holds)


def _eval(task: Tuple[int, Dict[str, Any], List[Tuple[int, int]], int, float]) -> Dict[str, Any]:
    cid, params, folds, split_at, fee_r = task
    train = []; test = []
    for idx, (s, e) in enumerate(folds):
        m = _simulate(params, s, e, fee_r)
        (train if idx < split_at else test).append(m)
    tm = _combine(train); om = _combine(test)
    rank = _score(om) + 0.20 * _score(tm)
    return {"id": cid, "rank": round(rank, 6), "params": params, "train": tm, "test": om}


def _combine(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Combine by summing trade-level proxy metrics approximately across folds.
    trades = sum(int(x["trades"]) for x in items)
    net = sum(float(x["net_r"]) for x in items)
    dd = max((float(x["max_drawdown_r"]) for x in items), default=0.0)
    wr = sum(float(x["win_rate"]) * int(x["trades"]) for x in items) / trades if trades else 0.0
    pf = sum(min(20.0, float(x["profit_factor"])) for x in items) / len(items) if items else 0.0
    return {"folds": len(items), "trades": trades, "net_r": round(net, 6), "expectancy_r": round(net / trades, 6) if trades else 0.0, "profit_factor": round(pf, 6), "win_rate": round(wr, 6), "max_drawdown_r": round(dd, 6), "score": round(net + pf - dd, 6)}


def _folds(rows: int, folds: int, window: int, stride: int) -> List[Tuple[int, int]]:
    out = []
    end = rows
    while len(out) < folds and end - window >= 0:
        out.append((end-window, end)); end -= stride
    return list(reversed(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/binance/BTCUSD_TIMEFRAME_M1.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=5000)
    ap.add_argument("--folds", type=int, default=16)
    ap.add_argument("--window", type=int, default=7200)
    ap.add_argument("--stride", type=int, default=7200)
    ap.add_argument("--max-rows", type=int, default=140000)
    ap.add_argument("--seed", type=int, default=20260513)
    ap.add_argument("--fee-r", type=float, default=0.04, help="round-trip cost/slippage in R units")
    ap.add_argument("--out", default=str(ROOT / "optimization_runs/fast_btcusd_grid.json"))
    args = ap.parse_args()
    df = _load_csv(Path(args.csv))
    rows = min(len(df), int(args.max_rows) if args.max_rows > 0 else len(df))
    folds = _folds(rows, int(args.folds), int(args.window), int(args.stride))
    split_at = max(1, int(len(folds) * 0.70))
    rng = random.Random(int(args.seed))
    candidates = [dict(BASE)]
    while len(candidates) < int(args.candidates):
        if len(candidates) < int(args.candidates) * 0.75:
            candidates.append(_candidate(rng))
        else:
            candidates.append(_candidate(rng, seed=rng.choice(candidates[: max(1, len(candidates)//5)])))
    print(f"start rows={rows} folds={len(folds)} split_at={split_at} candidates={len(candidates)} workers={args.workers}", flush=True)
    started = datetime.now(timezone.utc)
    results = []
    tasks = [(i, p, folds, split_at, float(args.fee_r)) for i, p in enumerate(candidates)]
    with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_init, initargs=(str(args.csv), int(args.max_rows))) as ex:
        futs = [ex.submit(_eval, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); results.append(r)
            if n % 250 == 0:
                b = max(results, key=lambda x: float(x["rank"]))
                print(f"done={n}/{len(tasks)} best_rank={b['rank']} test={b['test']}", flush=True)
    results.sort(key=lambda x: float(x["rank"]), reverse=True)
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": (datetime.now(timezone.utc)-started).total_seconds(), "note": "BTCUSDT Binance 1m proxy; not MT5 broker truth. Strategy is fast LSR-like proxy for broad parameter search.", "csv": str(Path(args.csv).resolve()), "rows_used": rows, "folds": folds, "split_at": split_at, "fee_r": float(args.fee_r), "workers": int(args.workers), "candidates": len(candidates), "top": results[:50]}
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True); outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={outp}")
    print(json.dumps(out["top"][:10], ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
