from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _as_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out if out > 0 else None


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out else None


def load_mt5_ledger(path: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pos_id = _as_int(row.get("position_id"))
            symbol = str(row.get("symbol") or "").strip().upper()
            if pos_id is None or not symbol:
                continue
            out[(pos_id, symbol)] = row
    return out


def iter_local_ledgers(events_path: Path) -> Iterable[Dict[str, Any]]:
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "trade_ledger_normalized":
                yield obj


def load_local_ledger(events_path: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for obj in iter_local_ledgers(events_path):
        ticket = _as_int(obj.get("ticket"))
        symbol = str(obj.get("symbol") or "").strip().upper()
        if ticket is None or not symbol:
            continue
        # Keep last occurrence (latest write wins).
        out[(ticket, symbol)] = obj
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MT5 truth ledger CSV vs local trade_ledger_normalized events.")
    parser.add_argument("--mt5", required=True, help="Path to MT5 ledger CSV (from utils/mt5_history_ledger.py).")
    parser.add_argument("--events", default=str(Path(__file__).resolve().parents[1] / "events.jsonl"))
    parser.add_argument("--pnl-tol", type=float, default=0.01, help="PNL mismatch tolerance.")
    parser.add_argument("--limit", type=int, default=50, help="Max examples to print per category.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    mt5_path = Path(args.mt5).resolve()
    events_path = Path(args.events).resolve()
    if not mt5_path.exists():
        print(f"MT5 ledger not found: {mt5_path}", file=sys.stderr)
        return 2
    if not events_path.exists():
        print(f"events.jsonl not found: {events_path}", file=sys.stderr)
        return 2

    mt5 = load_mt5_ledger(mt5_path)
    local = load_local_ledger(events_path)

    mt5_keys = set(mt5.keys())
    local_keys = set(local.keys())

    missing_local = sorted(mt5_keys - local_keys)
    missing_mt5 = sorted(local_keys - mt5_keys)

    pnl_mismatch: List[Tuple[Tuple[int, str], float, float]] = []
    tol = max(0.0, float(args.pnl_tol))
    for key in sorted(mt5_keys & local_keys):
        mt5_pnl = _as_float(mt5[key].get("pnl"))
        local_pnl = _as_float(local[key].get("realized_pnl"))
        if mt5_pnl is None or local_pnl is None:
            continue
        if abs(float(mt5_pnl) - float(local_pnl)) > tol:
            pnl_mismatch.append((key, float(mt5_pnl), float(local_pnl)))

    print("Ledger Compare Summary")
    print(f"- MT5 positions: {len(mt5_keys)}")
    print(f"- Local ledgers: {len(local_keys)}")
    print(f"- Missing in local: {len(missing_local)}")
    print(f"- Missing in MT5: {len(missing_mt5)}")
    print(f"- PNL mismatches (tol={tol}): {len(pnl_mismatch)}")

    limit = max(1, int(args.limit))
    if missing_local:
        print("\nMissing In Local (ticket,symbol):")
        for ticket, symbol in missing_local[:limit]:
            print(f"- {ticket},{symbol}")
    if missing_mt5:
        print("\nMissing In MT5 (ticket,symbol):")
        for ticket, symbol in missing_mt5[:limit]:
            print(f"- {ticket},{symbol}")
    if pnl_mismatch:
        print("\nPNL Mismatch (ticket,symbol,mt5,local):")
        for (ticket, symbol), mt5_pnl, local_pnl in pnl_mismatch[:limit]:
            print(f"- {ticket},{symbol},{mt5_pnl},{local_pnl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

