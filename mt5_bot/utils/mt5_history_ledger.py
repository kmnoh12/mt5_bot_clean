from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from brokers.mt5_live import MT5LiveGateway, mt5
from core.config import load_config


def _parse_dt(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty datetime")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_symbols(raw: str) -> Optional[set[str]]:
    text = str(raw or "").strip()
    if not text:
        return None
    parts = [p.strip().upper() for p in text.replace(";", ",").split(",") if p.strip()]
    return set(parts) if parts else None


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if out == out else 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _deal_field(deal: Any, name: str) -> Any:
    try:
        return getattr(deal, name)
    except Exception:
        return None


def _deal_time_utc(deal: Any) -> Optional[datetime]:
    t = _deal_field(deal, "time")
    if isinstance(t, (int, float)) and t > 0:
        try:
            return datetime.fromtimestamp(float(t), tz=timezone.utc)
        except Exception:
            return None
    return None


def _close_reason_label(reason: Any) -> str:
    if mt5 is None:
        return "Other"
    r = _as_int(reason)
    if r == _as_int(getattr(mt5, "DEAL_REASON_SL", -999)):
        return "SL"
    if r == _as_int(getattr(mt5, "DEAL_REASON_TP", -999)):
        return "TP"
    if r in {
        _as_int(getattr(mt5, "DEAL_REASON_CLIENT", -999)),
        _as_int(getattr(mt5, "DEAL_REASON_MOBILE", -999)),
        _as_int(getattr(mt5, "DEAL_REASON_WEB", -999)),
        _as_int(getattr(mt5, "DEAL_REASON_EXPERT", -999)),
    }:
        return "Manual"
    return "Other"


def _deal_side_label(deal: Any) -> str:
    if mt5 is None:
        return ""
    t = _as_int(_deal_field(deal, "type"))
    if t == _as_int(getattr(mt5, "DEAL_TYPE_BUY", -999)):
        return "BUY"
    if t == _as_int(getattr(mt5, "DEAL_TYPE_SELL", -999)):
        return "SELL"
    return ""


def _is_entry_in(deal: Any) -> bool:
    if mt5 is None:
        return False
    entry = _as_int(_deal_field(deal, "entry"))
    return entry == _as_int(getattr(mt5, "DEAL_ENTRY_IN", -999))


def _is_entry_out(deal: Any) -> bool:
    if mt5 is None:
        return False
    entry = _as_int(_deal_field(deal, "entry"))
    return entry == _as_int(getattr(mt5, "DEAL_ENTRY_OUT", -999))


def _group_by_position_id(deals: Iterable[Any]) -> Dict[int, List[Any]]:
    out: Dict[int, List[Any]] = {}
    for deal in deals:
        pos_id = _as_int(_deal_field(deal, "position_id"))
        if pos_id <= 0:
            pos_id = _as_int(_deal_field(deal, "position"))
        if pos_id <= 0:
            continue
        out.setdefault(pos_id, []).append(deal)
    return out


def _vwavg_price(deals: List[Any]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for d in deals:
        price = _as_float(_deal_field(d, "price"))
        vol = _as_float(_deal_field(d, "volume"))
        if vol <= 0:
            continue
        num += price * vol
        den += vol
    if den <= 0:
        return None
    return num / den


def build_ledger_rows(
    *,
    deals: Iterable[Any],
    symbol_filter: Optional[set[str]],
    magic_filter: Optional[int],
) -> List[Dict[str, Any]]:
    grouped = _group_by_position_id(deals)
    rows: List[Dict[str, Any]] = []
    for position_id, items in grouped.items():
        items_sorted = sorted(items, key=lambda d: _as_int(_deal_field(d, "time_msc") or _deal_field(d, "time") or 0))
        symbol = str(_deal_field(items_sorted[0], "symbol") or "").upper()
        if symbol_filter is not None and symbol not in symbol_filter:
            continue
        if magic_filter is not None:
            if not any(_as_int(_deal_field(d, "magic")) == magic_filter for d in items_sorted):
                continue

        entry_deals = [d for d in items_sorted if _is_entry_in(d)]
        exit_deals = [d for d in items_sorted if _is_entry_out(d)]
        if not entry_deals or not exit_deals:
            continue

        side = _deal_side_label(entry_deals[0])
        volume = sum(_as_float(_deal_field(d, "volume")) for d in entry_deals)
        entry_price = _vwavg_price(entry_deals)
        exit_price = _vwavg_price(exit_deals)
        open_time = min((t for t in (_deal_time_utc(d) for d in entry_deals) if t is not None), default=None)
        close_time = max((t for t in (_deal_time_utc(d) for d in exit_deals) if t is not None), default=None)

        profit = sum(_as_float(_deal_field(d, "profit")) for d in items_sorted)
        swap = sum(_as_float(_deal_field(d, "swap")) for d in items_sorted)
        commission = sum(_as_float(_deal_field(d, "commission")) for d in items_sorted)
        fee = sum(_as_float(_deal_field(d, "fee")) for d in items_sorted)
        pnl = profit + swap + commission + fee

        last_exit = exit_deals[-1]
        close_reason = _close_reason_label(_deal_field(last_exit, "reason"))

        rows.append(
            {
                "position_id": int(position_id),
                "symbol": symbol,
                "side": side,
                "volume": float(volume),
                "open_time_utc": open_time.isoformat() if open_time else "",
                "close_time_utc": close_time.isoformat() if close_time else "",
                "entry_price": float(entry_price) if entry_price is not None else "",
                "exit_price": float(exit_price) if exit_price is not None else "",
                "pnl": float(pnl),
                "swap": float(swap),
                "commission": float(commission),
                "fee": float(fee),
                "close_reason": close_reason,
            }
        )
    rows.sort(key=lambda r: (str(r.get("close_time_utc") or ""), str(r.get("symbol") or ""), int(r.get("position_id") or 0)))
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MT5 truth ledger from history deals.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    parser.add_argument("--from", dest="from_dt", required=True, help="Start datetime (UTC). e.g. 2026-02-10 or 2026-02-10T00:00:00Z")
    parser.add_argument("--to", dest="to_dt", required=True, help="End datetime (UTC). e.g. 2026-02-16T00:00:00Z")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. e.g. BTCUSD,GOLD")
    parser.add_argument("--magic", type=int, default=None, help="Optional magic filter.")
    parser.add_argument("--out", default="", help="Output CSV path. Default: reports/mt5_ledger_YYYYMMDD_YYYYMMDD.csv")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    from_dt = _parse_dt(args.from_dt)
    to_dt = _parse_dt(args.to_dt)
    if to_dt <= from_dt:
        print("--to must be after --from", file=sys.stderr)
        return 2

    symbol_filter = _parse_symbols(args.symbols)
    cfg = load_config(config_path)
    gateway = MT5LiveGateway(cfg, notifier=None)
    try:
        if not gateway.connect():
            print("MT5 connect failed", file=sys.stderr)
            return 3
        if mt5 is None:
            print("MetaTrader5 package not installed", file=sys.stderr)
            return 3

        deals = mt5.history_deals_get(from_dt, to_dt)
        if deals is None:
            print("history_deals_get returned None", file=sys.stderr)
            return 4

        rows = build_ledger_rows(deals=deals, symbol_filter=symbol_filter, magic_filter=args.magic)
        stamp_from = from_dt.strftime("%Y%m%d")
        stamp_to = to_dt.strftime("%Y%m%d")
        out_path = Path(args.out) if args.out else (Path(__file__).resolve().parents[1] / "reports" / f"mt5_ledger_{stamp_from}_{stamp_to}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "position_id",
            "symbol",
            "side",
            "volume",
            "open_time_utc",
            "close_time_utc",
            "entry_price",
            "exit_price",
            "pnl",
            "swap",
            "commission",
            "fee",
            "close_reason",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"Wrote {len(rows)} rows -> {out_path}")
        return 0
    finally:
        try:
            gateway.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

