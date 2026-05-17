from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import struct
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.liquidity import classify_lsr_confirmation_quality


KST = timezone(timedelta(hours=9))
INDEX_FILE = "postmortem_index.jsonl"
LEARNING_FILE = "learning_samples.jsonl"
VISION_PROMPT_FILE = "vision_prompt.md"
BAR_SCHEMA_HELP = (
    "OHLC bar schema: CSV columns time,open,high,low,close with optional "
    "timeframe,symbol,tick_volume,spread,real_volume. JSON may be a list of bar "
    "objects, {'bars': [...]}, or {'M1': [...], 'M5': [...]}. time must be ISO-8601 "
    "or epoch seconds. timeframe defaults to --bars-timeframe when omitted."
)


@dataclass
class BarSource:
    bars: Dict[str, List[Dict[str, Any]]]
    source: str
    warnings: List[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only post-close trade analyzer. It parses bot event logs, "
            "optionally fetches MT5 OHLC bars, and writes chart/report/learning artifacts."
        )
    )
    parser.add_argument("--events", default="events.jsonl", help="Path to events.jsonl")
    parser.add_argument("--output-dir", default="reports/trade_postmortems")
    parser.add_argument("--symbol", default=None, help="Optional symbol filter, e.g. BTCUSD")
    parser.add_argument("--limit", type=int, default=10, help="Maximum newly analyzed trades")
    parser.add_argument("--trade-key", default=None, help="Analyze only this stable trade key")
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even if indexed")
    parser.add_argument(
        "--mt5",
        action="store_true",
        help="Try read-only MetaTrader5 copy_rates_range for M1/M5 chart context",
    )
    parser.add_argument(
        "--no-mt5",
        action="store_true",
        help="Disable MT5 access and use event-derived chart fallback only",
    )
    parser.add_argument(
        "--bars-csv",
        action="append",
        default=[],
        help=(
            "Read-only CSV OHLC bars file. May be passed multiple times. "
            + BAR_SCHEMA_HELP
        ),
    )
    parser.add_argument(
        "--bars-json",
        action="append",
        default=[],
        help=(
            "Read-only JSON OHLC bars file. May be passed multiple times. "
            + BAR_SCHEMA_HELP
        ),
    )
    parser.add_argument(
        "--bars-timeframe",
        default="M1",
        help="Default timeframe label for bar files whose rows omit timeframe. Default: M1",
    )
    return parser.parse_args(argv)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        text = str(value).strip()
        if not text:
            return None
        numeric = _safe_float(text)
        if numeric is not None and re.fullmatch(r"\d+(?:\.\d+)?", text):
            try:
                return datetime.fromtimestamp(numeric, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def event_dt(event: Dict[str, Any]) -> Optional[datetime]:
    return _parse_dt(event.get("ts_utc")) or _parse_dt(event.get("ts_kst"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return events


def _normalize_timeframe(value: Any, default: str = "M1") -> str:
    raw = str(value or default or "M1").strip().upper()
    aliases = {
        "1": "M1",
        "1M": "M1",
        "MIN1": "M1",
        "MINUTE1": "M1",
        "5": "M5",
        "5M": "M5",
        "MIN5": "M5",
        "MINUTE5": "M5",
    }
    return aliases.get(raw, raw)


def _normalize_bar(
    row: Dict[str, Any],
    *,
    default_timeframe: str,
    symbol_filter: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    symbol = str(row.get("symbol") or row.get("Symbol") or "").upper()
    if symbol_filter and symbol and symbol != symbol_filter.upper():
        return None, None, None

    time_value = row.get("time", row.get("Time", row.get("timestamp", row.get("datetime"))))
    dt = _parse_dt(time_value)
    if dt is None:
        return None, None, f"skipped bar with invalid time: {time_value!r}"

    def value_for(*names: str) -> Optional[float]:
        for name in names:
            if name in row:
                return _safe_float(row.get(name))
        return None

    open_p = value_for("open", "Open", "o")
    high = value_for("high", "High", "h")
    low = value_for("low", "Low", "l")
    close = value_for("close", "Close", "c")
    if None in (open_p, high, low, close):
        return None, None, f"skipped bar at {dt.isoformat()} with incomplete OHLC"

    timeframe = _normalize_timeframe(
        row.get("timeframe", row.get("Timeframe", row.get("tf"))),
        default_timeframe,
    )
    out: Dict[str, Any] = {
        "time": dt.astimezone(timezone.utc).isoformat(),
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
    }
    for key in ("tick_volume", "spread", "real_volume", "volume"):
        if key in row:
            out[key] = _safe_float(row.get(key))
    if symbol:
        out["symbol"] = symbol
    return timeframe, out, None


def _append_normalized_bars(
    rows: Iterable[Dict[str, Any]],
    bars: Dict[str, List[Dict[str, Any]]],
    warnings: List[str],
    *,
    default_timeframe: str,
    symbol_filter: Optional[str],
    source_name: str,
) -> None:
    skipped = 0
    for row in rows:
        timeframe, bar, warning = _normalize_bar(
            row,
            default_timeframe=default_timeframe,
            symbol_filter=symbol_filter,
        )
        if warning:
            skipped += 1
            continue
        if timeframe and bar:
            bars.setdefault(timeframe, []).append(bar)
    if skipped:
        warnings.append(f"{source_name}: skipped {skipped} invalid/incomplete bar rows.")


def _dedupe_sort_bars(bars: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    cleaned: Dict[str, List[Dict[str, Any]]] = {}
    for timeframe, rows in bars.items():
        by_time: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            dt = _bar_dt(row)
            if dt is None:
                continue
            by_time[dt.isoformat()] = row
        cleaned[timeframe] = [by_time[key] for key in sorted(by_time)]
    return cleaned


def _json_bar_rows(payload: Any) -> Iterable[Tuple[Optional[str], Dict[str, Any]]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield None, item
        return
    if not isinstance(payload, dict):
        return
    if isinstance(payload.get("bars"), list):
        for item in payload["bars"]:
            if isinstance(item, dict):
                yield None, item
    for timeframe, rows in payload.items():
        if timeframe == "bars" or not isinstance(rows, list):
            continue
        if re.fullmatch(r"[Mm]\d+|[Hh]\d+|[Dd]1", str(timeframe)):
            for item in rows:
                if isinstance(item, dict):
                    yield str(timeframe), item


def load_bars_from_files(
    csv_paths: Sequence[str],
    json_paths: Sequence[str],
    *,
    default_timeframe: str = "M1",
    symbol_filter: Optional[str] = None,
) -> BarSource:
    bars: Dict[str, List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    default_tf = _normalize_timeframe(default_timeframe)

    for raw_path in csv_paths:
        path = Path(raw_path)
        if not path.exists():
            warnings.append(f"{path}: bars CSV not found.")
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except Exception as exc:
            warnings.append(f"{path}: failed to read bars CSV: {exc}")
            continue
        _append_normalized_bars(
            rows,
            bars,
            warnings,
            default_timeframe=default_tf,
            symbol_filter=symbol_filter,
            source_name=path.as_posix(),
        )

    for raw_path in json_paths:
        path = Path(raw_path)
        if not path.exists():
            warnings.append(f"{path}: bars JSON not found.")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            warnings.append(f"{path}: failed to read bars JSON: {exc}")
            continue
        rows: List[Dict[str, Any]] = []
        for timeframe, row in _json_bar_rows(payload):
            item = dict(row)
            if timeframe and "timeframe" not in item:
                item["timeframe"] = timeframe
            rows.append(item)
        _append_normalized_bars(
            rows,
            bars,
            warnings,
            default_timeframe=default_tf,
            symbol_filter=symbol_filter,
            source_name=path.as_posix(),
        )

    bars = _dedupe_sort_bars(bars)
    if any(bars.values()):
        return BarSource(bars=bars, source="REAL_BARS_FILE", warnings=warnings)
    if csv_paths or json_paths:
        warnings.append("No usable bars loaded from supplied bar files; falling back to the next source.")
    return BarSource(bars={}, source="NO_BARS_FILE", warnings=warnings)


def _safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "trade"


def _ts_slug(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _nested_get(payload: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class ClosedTrade:
    trade_key: str
    symbol: str
    strategy: Optional[str]
    side: Optional[str]
    ticket: Optional[int]
    entry_deal: Optional[int]
    entry_order: Optional[int]
    exit_order: Optional[int]
    entry_time_utc: Optional[datetime]
    exit_time_utc: Optional[datetime]
    entry_price: Optional[float]
    exit_price: Optional[float]
    volume: Optional[float]
    pnl: Optional[float]
    reason: Optional[str]
    sl: Optional[float]
    tp: Optional[float]
    entry_event: Dict[str, Any]
    exit_event: Dict[str, Any]
    ledger_event: Dict[str, Any]
    decision_event: Optional[Dict[str, Any]]
    quality_event: Optional[Dict[str, Any]]
    manage_events: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def _ticket_from_entry(event: Dict[str, Any]) -> Optional[int]:
    return _safe_int(_nested_get(event, ["result", "ticket"]))


def _is_filled_entry(event: Dict[str, Any]) -> bool:
    if event.get("event") != "order_submit":
        return False
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    return bool(result.get("ok")) and str(result.get("status", "")).upper() == "FILLED"


def _entry_metadata(entry_event: Dict[str, Any], decision_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    intent = entry_event.get("intent") if isinstance(entry_event.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    merged: Dict[str, Any] = dict(metadata)
    if decision_event and isinstance(decision_event.get("metadata"), dict):
        for key, value in decision_event["metadata"].items():
            merged.setdefault(key, value)
    return merged


def _make_trade_key(
    *,
    symbol: str,
    entry_order: Optional[int],
    entry_deal: Optional[int],
    exit_order: Optional[int],
    entry_time_utc: Optional[datetime],
    exit_time_utc: Optional[datetime],
) -> str:
    raw = (
        f"{symbol}_entryOrder{entry_order or 'na'}_entryDeal{entry_deal or 'na'}_"
        f"exitOrder{exit_order or 'na'}_{_ts_slug(entry_time_utc)}_{_ts_slug(exit_time_utc)}"
    )
    return _safe_slug(raw)


def extract_closed_trades(events: Sequence[Dict[str, Any]], symbol: Optional[str] = None) -> List[ClosedTrade]:
    symbol_filter = symbol.upper() if symbol else None
    indexed: List[Tuple[int, Dict[str, Any]]] = list(enumerate(events))
    entries_by_ticket: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    latest_decision_by_symbol: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    latest_quality_by_symbol: Dict[str, Tuple[int, Dict[str, Any]]] = {}

    close_rows_by_ticket: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    exit_events_by_ticket: Dict[int, Tuple[int, Dict[str, Any]]] = {}

    for idx, event in indexed:
        ev_symbol = str(event.get("symbol") or "").upper()
        name = event.get("event")
        if symbol_filter and ev_symbol and ev_symbol != symbol_filter:
            continue
        if name == "decision" and ev_symbol:
            latest_decision_by_symbol[ev_symbol] = (idx, event)
        elif name == "entry_quality_score" and ev_symbol:
            latest_quality_by_symbol[ev_symbol] = (idx, event)
        elif _is_filled_entry(event):
            ticket = _ticket_from_entry(event)
            if ticket is not None:
                entries_by_ticket[ticket] = (idx, event)
        elif name == "position_exit":
            ticket = _safe_int(_nested_get(event, ["result", "ticket"]))
            if ticket is not None:
                exit_events_by_ticket[ticket] = (idx, event)
        elif name in {"trade_ledger", "trade_ledger_normalized"}:
            ticket = _safe_int(event.get("ticket"))
            if ticket is not None:
                # Prefer normalized rows when both are present.
                previous = close_rows_by_ticket.get(ticket)
                if previous is None or name == "trade_ledger_normalized":
                    close_rows_by_ticket[ticket] = (idx, event)

    trades: List[ClosedTrade] = []
    for ticket, (close_idx, ledger) in sorted(close_rows_by_ticket.items(), key=lambda item: item[1][0]):
        ev_symbol = str(ledger.get("symbol") or "").upper()
        if symbol_filter and ev_symbol != symbol_filter:
            continue
        entry_pair = entries_by_ticket.get(ticket)
        if entry_pair is None:
            continue
        entry_idx, entry = entry_pair
        exit_idx, exit_event = exit_events_by_ticket.get(ticket, (close_idx, ledger))
        decision = None
        quality = None
        for idx in range(entry_idx - 1, max(-1, entry_idx - 30), -1):
            candidate = events[idx]
            if str(candidate.get("symbol") or "").upper() != ev_symbol:
                continue
            if decision is None and candidate.get("event") == "decision":
                decision = candidate
            if quality is None and candidate.get("event") == "entry_quality_score":
                quality = candidate
            if decision is not None and quality is not None:
                break
        if decision is None:
            decision = latest_decision_by_symbol.get(ev_symbol, (None, None))[1]
        if quality is None:
            quality = latest_quality_by_symbol.get(ev_symbol, (None, None))[1]

        manage_events: List[Dict[str, Any]] = []
        for candidate in events[entry_idx + 1 : min(len(events), exit_idx + 1)]:
            if str(candidate.get("symbol") or "").upper() != ev_symbol:
                continue
            if candidate.get("event") == "decision" and candidate.get("state") == "IN_POSITION":
                manage_events.append(candidate)

        intent = entry.get("intent") if isinstance(entry.get("intent"), dict) else {}
        result_raw = _nested_get(entry, ["result", "raw"], {})
        if not isinstance(result_raw, dict):
            result_raw = {}
        entry_time = event_dt(entry)
        exit_time = event_dt(exit_event) or event_dt(ledger)
        entry_order = _safe_int(_nested_get(entry, ["result", "ticket"])) or _safe_int(result_raw.get("order"))
        entry_deal = _safe_int(result_raw.get("deal"))
        exit_order = _safe_int(ledger.get("ticket")) or _safe_int(_nested_get(exit_event, ["result", "ticket"]))
        trade_key = _make_trade_key(
            symbol=ev_symbol,
            entry_order=entry_order,
            entry_deal=entry_deal,
            exit_order=exit_order,
            entry_time_utc=entry_time,
            exit_time_utc=exit_time,
        )

        trades.append(
            ClosedTrade(
                trade_key=trade_key,
                symbol=ev_symbol,
                strategy=str(ledger.get("strategy") or entry.get("strategy") or "") or None,
                side=str(ledger.get("side") or intent.get("side") or entry.get("action") or "").upper() or None,
                ticket=ticket,
                entry_deal=entry_deal,
                entry_order=entry_order,
                exit_order=exit_order,
                entry_time_utc=entry_time,
                exit_time_utc=exit_time,
                entry_price=_safe_float(ledger.get("entry_price"))
                or _safe_float(_nested_get(entry, ["result", "filled_price"]))
                or _safe_float(result_raw.get("price")),
                exit_price=_safe_float(ledger.get("exit_price"))
                or _safe_float(_nested_get(exit_event, ["result", "filled_price"])),
                volume=_safe_float(ledger.get("volume")) or _safe_float(intent.get("volume")),
                pnl=_safe_float(ledger.get("realized_pnl")) or _safe_float(_nested_get(exit_event, ["result", "pnl"])),
                reason=str(ledger.get("reason") or exit_event.get("reason") or "") or None,
                sl=_safe_float(intent.get("sl")),
                tp=_safe_float(intent.get("tp")),
                entry_event=entry,
                exit_event=exit_event,
                ledger_event=ledger,
                decision_event=decision,
                quality_event=quality,
                manage_events=manage_events,
                metadata=_entry_metadata(entry, decision),
            )
        )
    return trades


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.{stamp}.{suffix}.bak")
        suffix += 1
    shutil.copy2(path, backup)


def _read_jsonl_keys(path: Path, key_field: str = "trade_key") -> set[str]:
    keys: set[str] = set()
    for row in read_jsonl(path):
        value = row.get(key_field)
        if value:
            keys.add(str(value))
    return keys


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup_existing(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup_existing(path)
    path.write_text(text, encoding="utf-8")


def _upsert_jsonl(path: Path, payload: Dict[str, Any], key_field: str = "trade_key") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(path)
    key = str(payload.get(key_field))
    kept = [row for row in rows if str(row.get(key_field)) != key]
    kept.append(payload)
    _backup_existing(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fetch_mt5_bars(
    symbol: str,
    entry_time_utc: Optional[datetime],
    exit_time_utc: Optional[datetime],
    *,
    enabled: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[str]]:
    if not enabled:
        return {}, "MT5 disabled; using event-derived fallback context."
    if entry_time_utc is None or exit_time_utc is None:
        return {}, "MT5 bars skipped because entry/exit timestamps are missing."
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local terminal package
        return {}, f"MetaTrader5 import unavailable: {exc}"

    if not mt5.initialize():  # pragma: no cover - depends on local terminal state
        return {}, f"MetaTrader5 initialize failed: {mt5.last_error()}"

    try:  # pragma: no cover - depends on local terminal state
        start = entry_time_utc - timedelta(minutes=75)
        end = exit_time_utc + timedelta(minutes=35)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for label, timeframe in {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5}.items():
            raw = mt5.copy_rates_range(symbol, timeframe, start, end)
            rows: List[Dict[str, Any]] = []
            if raw is not None:
                for item in raw:
                    rows.append(
                        {
                            "time": datetime.fromtimestamp(int(item["time"]), tz=timezone.utc).isoformat(),
                            "open": float(item["open"]),
                            "high": float(item["high"]),
                            "low": float(item["low"]),
                            "close": float(item["close"]),
                            "tick_volume": int(item["tick_volume"]),
                            "spread": int(item["spread"]),
                            "real_volume": int(item["real_volume"]),
                        }
                    )
            out[label] = rows
        return _dedupe_sort_bars(out), None
    finally:
        mt5.shutdown()


def _bar_dt(bar: Dict[str, Any]) -> Optional[datetime]:
    return _parse_dt(bar.get("time"))


def _bars_between(
    bars: Sequence[Dict[str, Any]], start: Optional[datetime], end: Optional[datetime]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bar in bars:
        dt = _bar_dt(bar)
        if dt is None:
            continue
        if start is not None and dt < start:
            continue
        if end is not None and dt > end:
            continue
        out.append(bar)
    return out


def _nearest_bar_spread(
    bars: Sequence[Dict[str, Any]],
    entry_time_utc: Optional[datetime],
) -> Optional[float]:
    fallback: Optional[float] = None
    best: Optional[Tuple[float, float]] = None
    for bar in bars:
        spread = _safe_float(bar.get("spread"))
        if spread is None:
            continue
        if fallback is None:
            fallback = spread
        dt = _bar_dt(bar)
        if entry_time_utc is None or dt is None:
            continue
        distance = abs((dt - entry_time_utc).total_seconds())
        if best is None or distance < best[0]:
            best = (distance, spread)
    if best is not None:
        return best[1]
    return fallback


def _ema(values: Sequence[float], period: int) -> Optional[float]:
    if not values:
        return None
    alpha = 2.0 / (period + 1.0)
    ema = values[0]
    for value in values[1:]:
        ema = (value * alpha) + (ema * (1.0 - alpha))
    return ema


def _analysis_bars(bars: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
    for timeframe in ("M1", "M5"):
        rows = bars.get(timeframe, [])
        if rows:
            return timeframe, rows
    for timeframe in sorted(bars):
        if bars[timeframe]:
            return timeframe, bars[timeframe]
    return "M1", []


def _swing_summary(
    bars: Sequence[Dict[str, Any]],
    *,
    entry_price: Optional[float],
    side: str,
) -> Dict[str, Any]:
    valid = [
        bar
        for bar in bars
        if _safe_float(bar.get("high")) is not None
        and _safe_float(bar.get("low")) is not None
        and _safe_float(bar.get("close")) is not None
    ]
    if len(valid) < 4:
        return {
            "last_swing_direction": None,
            "pullback_depth": None,
            "bar_chase_flag": None,
            "bar_exhaustion_flag": None,
        }

    highs = [_safe_float(bar.get("high")) for bar in valid]
    lows = [_safe_float(bar.get("low")) for bar in valid]
    closes = [_safe_float(bar.get("close")) for bar in valid]
    highs_f = [value for value in highs if value is not None]
    lows_f = [value for value in lows if value is not None]
    closes_f = [value for value in closes if value is not None]
    recent_high = max(highs_f)
    recent_low = min(lows_f)
    price_range = recent_high - recent_low

    swing_points: List[Tuple[int, str, float]] = []
    for idx in range(1, len(valid) - 1):
        high = _safe_float(valid[idx].get("high"))
        low = _safe_float(valid[idx].get("low"))
        prev_high = _safe_float(valid[idx - 1].get("high"))
        next_high = _safe_float(valid[idx + 1].get("high"))
        prev_low = _safe_float(valid[idx - 1].get("low"))
        next_low = _safe_float(valid[idx + 1].get("low"))
        if None not in (high, prev_high, next_high) and high >= prev_high and high >= next_high:
            swing_points.append((idx, "HIGH", high))
        if None not in (low, prev_low, next_low) and low <= prev_low and low <= next_low:
            swing_points.append((idx, "LOW", low))

    last_swing_direction = None
    if len(swing_points) >= 2:
        prev = swing_points[-2]
        last = swing_points[-1]
        if prev[1] == "LOW" and last[1] == "HIGH":
            last_swing_direction = "UP"
        elif prev[1] == "HIGH" and last[1] == "LOW":
            last_swing_direction = "DOWN"
        else:
            last_swing_direction = "HIGHER_HIGH" if last[2] > prev[2] else "LOWER_LOW"
    elif len(closes_f) >= 2:
        last_swing_direction = "UP" if closes_f[-1] > closes_f[0] else "DOWN" if closes_f[-1] < closes_f[0] else "FLAT"

    pullback_depth = None
    if entry_price is not None and price_range > 0:
        if last_swing_direction in {"UP", "HIGHER_HIGH"}:
            pullback_depth = max(0.0, recent_high - entry_price) / price_range
        elif last_swing_direction in {"DOWN", "LOWER_LOW"}:
            pullback_depth = max(0.0, entry_price - recent_low) / price_range

    entry_position = None
    if entry_price is not None and price_range > 0:
        entry_position = (entry_price - recent_low) / price_range
    bar_chase_flag = None
    if entry_position is not None:
        bar_chase_flag = (side == "BUY" and entry_position >= 0.80) or (
            side == "SELL" and entry_position <= 0.20
        )

    slope = None
    if len(closes_f) >= 2:
        slope = (closes_f[-1] - closes_f[0]) / float(len(closes_f) - 1)
    avg_range = sum((high - low) for high, low in zip(highs_f, lows_f)) / len(highs_f) if highs_f else 0.0
    last_range = highs_f[-1] - lows_f[-1] if highs_f and lows_f else 0.0
    bar_exhaustion_flag = bool(avg_range > 0 and last_range >= avg_range * 1.8)
    if slope is not None and price_range > 0:
        bar_exhaustion_flag = bar_exhaustion_flag or abs(slope) >= price_range / max(len(closes_f), 1) * 1.5

    return {
        "last_swing_direction": last_swing_direction,
        "pullback_depth": pullback_depth,
        "bar_chase_flag": bar_chase_flag,
        "bar_exhaustion_flag": bar_exhaustion_flag,
    }


def compute_features(trade: ClosedTrade, bars: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    side = (trade.side or "").upper()
    entry_price = trade.entry_price
    exit_price = trade.exit_price
    metadata = trade.metadata
    analysis_timeframe, analysis_rows = _analysis_bars(bars)
    m1 = bars.get("M1", [])
    m5 = bars.get("M5", [])
    pre_start = trade.entry_time_utc - timedelta(minutes=60) if trade.entry_time_utc else None
    pre_bars = _bars_between(analysis_rows, pre_start, trade.entry_time_utc)
    post_bars = _bars_between(analysis_rows, trade.entry_time_utc, trade.exit_time_utc)
    if not post_bars and entry_price is not None and exit_price is not None:
        post_bars = [
            {"time": trade.entry_time_utc.isoformat() if trade.entry_time_utc else "", "high": entry_price, "low": entry_price, "close": entry_price},
            {"time": trade.exit_time_utc.isoformat() if trade.exit_time_utc else "", "high": exit_price, "low": exit_price, "close": exit_price},
        ]

    recent_high = max((_safe_float(bar.get("high")) for bar in pre_bars), default=None)
    recent_low = min((_safe_float(bar.get("low")) for bar in pre_bars), default=None)
    entry_position = None
    if entry_price is not None and recent_high is not None and recent_low is not None and recent_high != recent_low:
        entry_position = (entry_price - recent_low) / (recent_high - recent_low)

    post_high = max((_safe_float(bar.get("high")) for bar in post_bars), default=None)
    post_low = min((_safe_float(bar.get("low")) for bar in post_bars), default=None)
    adverse = None
    favorable = None
    if entry_price is not None and post_high is not None and post_low is not None:
        if side == "SELL":
            adverse = max(0.0, post_high - entry_price)
            favorable = max(0.0, entry_price - post_low)
        else:
            adverse = max(0.0, entry_price - post_low)
            favorable = max(0.0, post_high - entry_price)

    closes = [_safe_float(bar.get("close")) for bar in pre_bars]
    closes = [value for value in closes if value is not None]
    slope = None
    if len(closes) >= 2:
        slope = (closes[-1] - closes[0]) / float(len(closes) - 1)
    ema20 = _ema(closes, 20) if closes else None
    ema50 = _ema(closes, 50) if closes else None
    close_vs_ema20 = None
    ema20_vs_ema50 = None
    if closes and ema20 is not None:
        close_vs_ema20 = closes[-1] - ema20
    if ema20 is not None and ema50 is not None:
        ema20_vs_ema50 = ema20 - ema50

    risk_per_unit = _safe_float(metadata.get("risk_per_unit"))
    if risk_per_unit is None and entry_price is not None and trade.sl is not None:
        risk_per_unit = abs(trade.sl - entry_price)
    side_move = None
    price_r = None
    if entry_price is not None and exit_price is not None:
        side_move = entry_price - exit_price if side == "SELL" else exit_price - entry_price
        if risk_per_unit and risk_per_unit > 0:
            price_r = side_move / risk_per_unit

    estimated_loss = _safe_float(metadata.get("estimated_net_loss")) or _safe_float(
        _nested_get(metadata, ["risk_model", "estimated_net_loss_usd"])
    )
    pnl_r = None
    if trade.pnl is not None and estimated_loss and estimated_loss > 0:
        pnl_r = trade.pnl / estimated_loss
    entry_reference_price = _safe_float(metadata.get("signal_close"))
    if entry_reference_price is None:
        entry_reference_price = _safe_float(metadata.get("entry_price"))
    if entry_reference_price is None:
        entry_reference_price = _safe_float(_nested_get(metadata, ["opportunity", "entry_price"]))

    entry_shortfall_price = None
    entry_shortfall_r = None
    if entry_reference_price is not None and entry_price is not None:
        if side == "SELL":
            entry_shortfall_price = entry_reference_price - entry_price
        elif side == "BUY":
            entry_shortfall_price = entry_price - entry_reference_price
        else:
            entry_shortfall_price = entry_price - entry_reference_price
        if risk_per_unit and risk_per_unit > 0:
            entry_shortfall_r = entry_shortfall_price / risk_per_unit

    net_execution_drag_r = None
    if price_r is not None and pnl_r is not None:
        net_execution_drag_r = price_r - pnl_r

    sweep_level = _safe_float(metadata.get("sweep_level"))
    sweep_extreme = _safe_float(metadata.get("sweep_extreme"))
    distance_from_sweep_level = None if entry_price is None or sweep_level is None else entry_price - sweep_level
    distance_from_sweep_extreme = None if entry_price is None or sweep_extreme is None else entry_price - sweep_extreme
    reclaim_quality = metadata.get("reclaim_quality") if isinstance(metadata.get("reclaim_quality"), dict) else {}
    reclaim_distance_atr = _safe_float(reclaim_quality.get("reclaim_distance_atr"))
    sweep_depth_atr = _safe_float(reclaim_quality.get("sweep_depth_atr"))
    reclaim_to_sweep_depth_ratio = _safe_float(reclaim_quality.get("reclaim_to_sweep_depth_ratio"))
    confirmation_path = str(reclaim_quality.get("confirmation_path") or metadata.get("confirmation_path") or "")
    retest_confirmed = bool(reclaim_quality.get("retest_confirmed", False))
    confirmation_path_key = confirmation_path.strip().lower()

    sweep_time = None
    sweep_key = str(metadata.get("sweep_event_key") or "")
    if "|" in sweep_key:
        sweep_time = _parse_dt(sweep_key.split("|", 1)[1])
    time_from_sweep_to_entry_sec = _safe_float(metadata.get("time_from_sweep_to_reclaim_sec"))
    if time_from_sweep_to_entry_sec is None:
        time_from_sweep_to_entry_sec = _safe_float(reclaim_quality.get("time_from_sweep_to_reclaim_sec"))
    if time_from_sweep_to_entry_sec is None and sweep_time is not None and trade.entry_time_utc is not None:
        time_from_sweep_to_entry_sec = (trade.entry_time_utc - sweep_time).total_seconds()

    displacement = _safe_float(metadata.get("displacement_ratio"))
    adx = _safe_float(metadata.get("adx_entry"))
    fee_rr = _safe_float(metadata.get("fee_adjusted_rr")) or _safe_float(metadata.get("expected_rr"))
    spread_points = _safe_float(metadata.get("current_spread"))
    if spread_points is None:
        spread_points = _safe_float(metadata.get("spread_points"))
    if spread_points is None:
        spread_points = _safe_float(metadata.get("spread"))
    if spread_points is None:
        spread_points = _nearest_bar_spread(m1 or analysis_rows, trade.entry_time_utc)
    estimated_cost_usd = _safe_float(metadata.get("estimated_cost_usd"))
    if estimated_cost_usd is None:
        estimated_cost_usd = _safe_float(_nested_get(metadata, ["risk_model", "estimated_cost_usd"]))
    estimated_cost_to_expected_loss_r = None
    if estimated_cost_usd is not None and estimated_loss and estimated_loss > 0:
        estimated_cost_to_expected_loss_r = estimated_cost_usd / estimated_loss

    explicit_cost_components = [
        _safe_float(trade.ledger_event.get("commission")),
        _safe_float(trade.ledger_event.get("swap")),
        _safe_float(trade.ledger_event.get("fee")),
    ]
    realized_explicit_cost_usd = sum(-value for value in explicit_cost_components if value is not None and value < 0.0)
    realized_explicit_cost_r = None
    if realized_explicit_cost_usd and estimated_loss and estimated_loss > 0:
        realized_explicit_cost_r = realized_explicit_cost_usd / estimated_loss
    quality_score = _safe_float(trade.quality_event.get("score")) if trade.quality_event else None
    quality_threshold = _safe_float(trade.quality_event.get("threshold")) if trade.quality_event else None
    m5_align = _safe_float(_nested_get(trade.quality_event or {}, ["features", "m5_align"]))

    entry_chased_extension = bool(displacement is not None and displacement >= 2.0)
    entered_against_short_momentum = None
    if slope is not None:
        entered_against_short_momentum = (side == "SELL" and slope > 0) or (side == "BUY" and slope < 0)
    entered_into_exhaustion = bool(
        (adx is not None and adx >= 45.0)
        and (displacement is not None and displacement >= 1.5)
        and (
            quality_score is None
            or quality_threshold is None
            or quality_score <= quality_threshold + 0.05
        )
    )
    geometry_enough_after_costs = None
    if fee_rr is not None:
        geometry_enough_after_costs = fee_rr >= 2.0
    swings = _swing_summary(pre_bars[-30:], entry_price=entry_price, side=side)
    is_lsr = str(trade.strategy or "").strip().lower().startswith("liquidity_sweep_reversal")
    confirmation_flags = classify_lsr_confirmation_quality(
        confirmation_path=confirmation_path_key,
        retest_confirmed=retest_confirmed,
        reclaim_distance_atr=reclaim_distance_atr,
        sweep_depth_atr=sweep_depth_atr,
        reclaim_to_sweep_depth_ratio=reclaim_to_sweep_depth_ratio,
        displacement_ratio=displacement,
        time_from_sweep_to_reclaim_sec=time_from_sweep_to_entry_sec,
        reclaim_window_sec=metadata.get("reclaim_window_sec") or reclaim_quality.get("reclaim_window_sec"),
        entered_into_exhaustion=entered_into_exhaustion,
        is_lsr=is_lsr,
    )
    lsr_unconfirmed_reclaim = bool(confirmation_flags["lsr_unconfirmed_reclaim"])
    shallow_reclaim_confirmation = bool(confirmation_flags["shallow_reclaim_confirmation"])
    weak_reclaim_after_deep_sweep = bool(confirmation_flags["weak_reclaim_after_deep_sweep"])
    lsr_unconfirmed_reclaim_chase = bool(confirmation_flags["lsr_unconfirmed_reclaim_chase"])
    late_window_reclaim = bool(confirmation_flags["late_window_reclaim"])
    invalid_reclaim_timing = bool(confirmation_flags["invalid_reclaim_timing"])
    lsr_confirmation_score = _safe_float(confirmation_flags.get("confirmation_score"))

    return {
        "bar_analysis_source_timeframe": analysis_timeframe,
        "entry_position_in_recent_range": entry_position,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "distance_from_sweep_level": distance_from_sweep_level,
        "distance_from_sweep_extreme": distance_from_sweep_extreme,
        "time_from_sweep_to_entry_sec": time_from_sweep_to_entry_sec,
        "confirmation_path": confirmation_path or None,
        "retest_confirmed": retest_confirmed,
        "clean_reclaim": retest_confirmed,
        "clean_reclaim_confirmed": retest_confirmed,
        "lsr_unconfirmed_reclaim": lsr_unconfirmed_reclaim,
        "shallow_reclaim_confirmation": shallow_reclaim_confirmation,
        "shallow_reclaim_threshold_atr": confirmation_flags.get("shallow_reclaim_threshold_atr"),
        "weak_reclaim_after_deep_sweep": weak_reclaim_after_deep_sweep,
        "lsr_unconfirmed_reclaim_chase": lsr_unconfirmed_reclaim_chase,
        "late_window_reclaim": late_window_reclaim,
        "invalid_reclaim_timing": invalid_reclaim_timing,
        "lsr_confirmation_score": lsr_confirmation_score,
        "lsr_confirmation_band": confirmation_flags.get("confirmation_band"),
        "lsr_confirmation_score_components": confirmation_flags.get("confirmation_score_components"),
        "reclaim_window_elapsed_ratio": confirmation_flags.get("reclaim_window_elapsed_ratio"),
        "reclaim_distance_atr": reclaim_distance_atr,
        "sweep_depth_atr": sweep_depth_atr,
        "reclaim_to_sweep_depth_ratio": reclaim_to_sweep_depth_ratio,
        "m1_slope_per_bar": slope,
        "ema20": ema20,
        "ema50": ema50,
        "close_vs_ema20": close_vs_ema20,
        "ema20_vs_ema50": ema20_vs_ema50,
        "adverse_excursion_price": adverse,
        "favorable_excursion_price": favorable,
        "risk_per_unit": risk_per_unit,
        "side_adjusted_price_move": side_move,
        "price_r_multiple": price_r,
        "pnl_r_multiple": pnl_r,
        "entry_implementation_shortfall_price": entry_shortfall_price,
        "entry_implementation_shortfall_r": entry_shortfall_r,
        "net_execution_drag_r": net_execution_drag_r,
        "expected_net_loss_usd": estimated_loss,
        "expected_net_profit_at_tp_usd": _safe_float(metadata.get("estimated_net_profit_at_tp"))
        or _safe_float(_nested_get(metadata, ["risk_model", "estimated_net_profit_at_tp_usd"])),
        "fee_adjusted_rr": fee_rr,
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_cost_to_expected_loss_r": estimated_cost_to_expected_loss_r,
        "realized_explicit_cost_usd": realized_explicit_cost_usd,
        "realized_explicit_cost_r": realized_explicit_cost_r,
        "spread_points": spread_points,
        "atr_regime_ratio": _safe_float(metadata.get("atr_regime_ratio")),
        "chop_score": _safe_float(metadata.get("chop_score")),
        "tp_profile": metadata.get("tp_profile"),
        "adx_entry": adx,
        "displacement_ratio": displacement,
        "m5_align": m5_align,
        "entry_quality_score": quality_score,
        "entry_quality_threshold": quality_threshold,
        "entry_chased_extension": entry_chased_extension,
        "entered_into_exhaustion": entered_into_exhaustion,
        "entered_against_short_term_momentum": entered_against_short_momentum,
        "last_swing_direction": swings.get("last_swing_direction"),
        "pullback_depth": swings.get("pullback_depth"),
        "bar_chase_flag": swings.get("bar_chase_flag"),
        "bar_exhaustion_flag": swings.get("bar_exhaustion_flag"),
        "sl_tp_geometry_enough_after_costs": geometry_enough_after_costs,
        "exit_inferred_type": infer_exit_type(trade),
        "bars_m1_count": len(m1),
        "bars_m5_count": len(m5),
        "post_entry_bars_count": len(post_bars),
    }


def infer_exit_type(trade: ClosedTrade) -> str:
    reason = str(trade.reason or "").upper()
    if "BROKER_AUTO_CLOSE" in reason:
        code = reason.split(":", 1)[1] if ":" in reason else ""
        if code == "4":
            return "BROKER_AUTO_CLOSE_SL_OR_SO_CODE_4"
        return f"BROKER_AUTO_CLOSE_{code or 'UNKNOWN'}"
    if "TP" in reason:
        return "TAKE_PROFIT_OR_TP_GUARD"
    if "SL" in reason or "STOP" in reason:
        return "STOP_LOSS"
    if "MANUAL" in reason:
        return "MANUAL"
    if "GUARD" in reason:
        return "GUARD_DRIVEN"
    return "UNKNOWN"


def quality_grade(trade: ClosedTrade, features: Dict[str, Any]) -> str:
    pnl = trade.pnl or 0.0
    if pnl > 0:
        flags = sum(
            1
            for key in ("entry_chased_extension", "entered_into_exhaustion", "entered_against_short_term_momentum")
            if features.get(key) is True
        )
        return "A" if flags == 0 else "B" if flags == 1 else "C"
    flags = sum(
        1
        for key in ("entry_chased_extension", "entered_into_exhaustion", "entered_against_short_term_momentum")
        if features.get(key) is True
    )
    if features.get("entry_quality_score") == features.get("entry_quality_threshold"):
        flags += 1
    if features.get("price_r_multiple") is not None and features["price_r_multiple"] <= -1.0:
        flags += 1
    if flags >= 3:
        return "F"
    if flags == 2:
        return "D"
    return "C"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_guard_candidates(trade: ClosedTrade, features: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    if features.get("lsr_unconfirmed_reclaim_chase"):
        candidates.append(
            "LSR reclaim_only/tick_reclaim 진입이 추격/약한 되돌림과 겹치면 retest 또는 확인봉을 shadow 검증 전까지 요구 후보로 표시한다."
        )
    elif features.get("weak_reclaim_after_deep_sweep"):
        candidates.append(
            "깊은 sweep 이후 reclaim 폭이 sweep 깊이보다 작으면 단일 sweep 진입으로 보지 말고 되돌림 품질을 추가 확인한다."
        )
    if features.get("entry_chased_extension"):
        candidates.append(
            "displacement_ratio가 2.0 이상이면 즉시 진입 금지 또는 추가 되돌림 확인을 요구한다."
        )
    if features.get("entered_into_exhaustion"):
        candidates.append(
            "ADX가 높고 변위가 큰 구간에서는 entry_quality_score가 임계값과 같을 때 통과시키지 않는다."
        )
    if features.get("entered_against_short_term_momentum"):
        candidates.append(
            "M1 기울기가 진입 방향과 반대면 M5 정렬만으로는 부족하므로 단기 모멘텀 필터를 추가한다."
        )
    if features.get("pnl_r_multiple") is not None and features["pnl_r_multiple"] <= -1.0:
        candidates.append(
            "손절 1R 이상으로 즉시 밀린 패턴은 동일 sweep 레벨 재진입 쿨다운을 더 길게 둔다."
        )
    if trade.quality_event and features.get("entry_quality_score") == features.get("entry_quality_threshold"):
        candidates.append(
            "entry_quality_score가 threshold와 정확히 같은 경계값이면 허용하지 않고 최소 여유분을 요구한다."
        )
    if not candidates:
        candidates.append("현재 샘플만으로 자동 강화하지 말고 동일 feature 조합의 반복 성과를 누적 확인한다.")
    return candidates


def build_vision_prompt(trade: ClosedTrade, chart_path: Path, bar_source: str) -> str:
    return (
        "Hermes/Codex vision review task:\n"
        f"1. Open chart image: {chart_path.as_posix()}\n"
        f"2. Bar source: {bar_source}. If this is REAL_BARS_MT5 or REAL_BARS_FILE, inspect actual candle bodies, wicks, swing waves, pullbacks, failed continuation, and exhaustion.\n"
        f"3. Inspect {trade.symbol} {trade.side} entry and exit timing against the visible candle sequence/waveform.\n"
        "4. Explain whether entry occurred after exhaustion, during momentum continuation, into a pullback, or at a clean reclaim.\n"
        "5. Compare entry, exit, SL, TP, sweep level/extreme, and stage target marks; call out any mark that price never respected.\n"
        "6. Append a visual critique section to the JSON/Markdown with evidence from candle shape, wick rejection, range expansion/compression, and local swing structure.\n"
        "7. Do not change live config, place orders, close positions, or edit secrets. Guard changes are review-only recommendations.\n"
        "8. For losses, be harsh and propose guard candidates only when the visual evidence supports them.\n"
    )


def generate_chart(
    trade: ClosedTrade,
    bars: Dict[str, List[Dict[str, Any]]],
    features: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as exc:
        _write_fallback_png(trade, bars, output_path)
        return f"matplotlib unavailable; wrote dependency-free fallback PNG: {exc}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _backup_existing(output_path)
    primary_timeframe, primary_bars = _analysis_bars(bars)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=120)
    plotted = False

    candle_bars = [bar for bar in primary_bars if _bar_dt(bar) is not None]
    if candle_bars:
        dates = [mdates.date2num(_bar_dt(bar)) for bar in candle_bars if _bar_dt(bar) is not None]
        if len(dates) >= 2:
            width = max((dates[-1] - dates[0]) / max(len(dates), 1) * 0.65, 0.0002)
        else:
            width = 0.0005
        for bar in candle_bars:
            dt = _bar_dt(bar)
            if dt is None:
                continue
            x = mdates.date2num(dt)
            open_p = _safe_float(bar.get("open"))
            high = _safe_float(bar.get("high"))
            low = _safe_float(bar.get("low"))
            close = _safe_float(bar.get("close"))
            if None in (open_p, high, low, close):
                continue
            color = "#16845b" if close >= open_p else "#c43c39"
            ax.vlines(x, low, high, color=color, linewidth=1.0, alpha=0.85)
            body_low = min(open_p, close)
            body_height = max(abs(close - open_p), 0.01)
            ax.add_patch(
                plt.Rectangle(
                    (x - width / 2.0, body_low),
                    width,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.75,
                )
            )
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=KST))
        plotted = True
        if primary_timeframe != "M5" and bars.get("M5"):
            m5_points = [
                (_bar_dt(bar), _safe_float(bar.get("close")))
                for bar in bars["M5"]
                if _bar_dt(bar) is not None and _safe_float(bar.get("close")) is not None
            ]
            if m5_points:
                ax.plot_date(
                    [mdates.date2num(dt) for dt, _ in m5_points if dt is not None],
                    [close for _, close in m5_points if close is not None],
                    "-",
                    color="#333333",
                    linewidth=1.0,
                    alpha=0.35,
                    label="M5 close",
                )
    else:
        points: List[Tuple[datetime, float, str]] = []
        signal_close = _safe_float(trade.metadata.get("signal_close"))
        if trade.decision_event and signal_close is not None:
            dt = event_dt(trade.decision_event)
            if dt is not None:
                points.append((dt, signal_close, "signal_close"))
        if trade.entry_time_utc is not None and trade.entry_price is not None:
            points.append((trade.entry_time_utc, trade.entry_price, "entry"))
        if trade.exit_time_utc is not None and trade.exit_price is not None:
            points.append((trade.exit_time_utc, trade.exit_price, "exit"))
        points.sort(key=lambda item: item[0])
        if points:
            xs = [mdates.date2num(item[0]) for item in points]
            ys = [item[1] for item in points]
            ax.plot_date(xs, ys, "-", color="#2d5aa7", linewidth=2.0, marker="o")
            for x, y, label in zip(xs, ys, [item[2] for item in points]):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 8), fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=KST))
            plotted = True

    def hline(value: Optional[float], label: str, color: str, style: str = "--") -> None:
        if value is not None:
            ax.axhline(value, color=color, linestyle=style, linewidth=1.2, alpha=0.85, label=label)

    hline(trade.sl, "SL", "#d62728")
    hline(trade.tp, "TP", "#2ca02c")
    hline(_safe_float(trade.metadata.get("sweep_level")), "sweep_level", "#9467bd", ":")
    hline(_safe_float(trade.metadata.get("sweep_extreme")), "sweep_extreme", "#8c564b", ":")
    hline(_safe_float(trade.metadata.get("stage_a_target")), "stage_a_target", "#ff7f0e", "-.")

    if trade.entry_time_utc is not None and trade.entry_price is not None:
        ax.scatter([mdates.date2num(trade.entry_time_utc)], [trade.entry_price], s=90, marker="^", color="#111111", label="entry", zorder=5)
        ax.annotate(
            f"ENTRY {trade.side} {_fmt(trade.entry_price, 2)}",
            (mdates.date2num(trade.entry_time_utc), trade.entry_price),
            textcoords="offset points",
            xytext=(8, 14),
            fontsize=9,
        )
    if trade.exit_time_utc is not None and trade.exit_price is not None:
        ax.scatter([mdates.date2num(trade.exit_time_utc)], [trade.exit_price], s=90, marker="v", color="#0000aa", label="exit", zorder=5)
        ax.annotate(
            f"EXIT {_fmt(trade.exit_price, 2)} PnL={_fmt(trade.pnl, 2)}",
            (mdates.date2num(trade.exit_time_utc), trade.exit_price),
            textcoords="offset points",
            xytext=(8, -20),
            fontsize=9,
        )

    title = (
        f"{trade.symbol} {trade.side or ''} postmortem | {primary_timeframe} candles | "
        f"{trade.trade_key}"
    )
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return None


def _write_png_rgb(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start : start + stride])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
    png.extend(chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup_existing(path)
    path.write_bytes(bytes(png))


def _write_fallback_png(trade: ClosedTrade, bars: Dict[str, List[Dict[str, Any]]], output_path: Path) -> None:
    width, height = 1100, 520
    pixels = bytearray([255, 255, 255] * width * height)

    def put(x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], thickness: int = 1) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for tx in range(-thickness + 1, thickness):
                for ty in range(-thickness + 1, thickness):
                    put(x0 + tx, y0 + ty, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        for yy in range(top, bottom + 1):
            for xx in range(left, right + 1):
                put(xx, yy, color)

    all_prices: List[float] = []
    all_times: List[datetime] = []
    _, primary_bars = _analysis_bars(bars)
    candle_bars = [bar for bar in primary_bars if _bar_dt(bar) is not None]
    for bar in candle_bars:
        all_times.append(_bar_dt(bar))  # type: ignore[arg-type]
        for key in ("open", "high", "low", "close"):
            value = _safe_float(bar.get(key))
            if value is not None:
                all_prices.append(value)
    for dt, price in (
        (trade.entry_time_utc, trade.entry_price),
        (trade.exit_time_utc, trade.exit_price),
    ):
        if dt is not None:
            all_times.append(dt)
        if price is not None:
            all_prices.append(price)
    for value in (
        trade.sl,
        trade.tp,
        _safe_float(trade.metadata.get("sweep_level")),
        _safe_float(trade.metadata.get("sweep_extreme")),
        _safe_float(trade.metadata.get("stage_a_target")),
    ):
        if value is not None:
            all_prices.append(value)

    if not all_prices:
        all_prices = [0.0, 1.0]
    min_price, max_price = min(all_prices), max(all_prices)
    pad = max((max_price - min_price) * 0.08, 1.0)
    min_price -= pad
    max_price += pad
    if not all_times:
        now = datetime.now(timezone.utc)
        all_times = [now, now + timedelta(minutes=1)]
    min_ts = min(all_times).timestamp()
    max_ts = max(all_times).timestamp()
    if min_ts == max_ts:
        min_ts -= 60.0
        max_ts += 60.0

    left, right, top, bottom = 72, width - 36, 36, height - 58

    def x_for(dt: Optional[datetime]) -> int:
        ts = (dt or all_times[0]).timestamp()
        return int(left + (ts - min_ts) / (max_ts - min_ts) * (right - left))

    def y_for(price: Optional[float]) -> int:
        p = float(price if price is not None else min_price)
        return int(bottom - (p - min_price) / (max_price - min_price) * (bottom - top))

    line(left, top, left, bottom, (80, 80, 80), 1)
    line(left, bottom, right, bottom, (80, 80, 80), 1)
    for frac in (0.25, 0.5, 0.75):
        y = int(top + frac * (bottom - top))
        line(left, y, right, y, (225, 225, 225), 1)

    def hline(value: Optional[float], color: Tuple[int, int, int]) -> None:
        if value is not None:
            y = y_for(value)
            line(left, y, right, y, color, 1)

    hline(trade.sl, (210, 55, 55))
    hline(trade.tp, (45, 150, 80))
    hline(_safe_float(trade.metadata.get("sweep_level")), (125, 70, 170))
    hline(_safe_float(trade.metadata.get("sweep_extreme")), (140, 85, 70))
    hline(_safe_float(trade.metadata.get("stage_a_target")), (230, 140, 30))

    previous: Optional[Tuple[int, int]] = None
    if candle_bars:
        candle_width = max(3, int((right - left) / max(len(candle_bars), 1) * 0.45))
        for bar in candle_bars:
            dt = _bar_dt(bar)
            open_p = _safe_float(bar.get("open"))
            high = _safe_float(bar.get("high"))
            low = _safe_float(bar.get("low"))
            close = _safe_float(bar.get("close"))
            if None in (dt, open_p, high, low, close):
                continue
            x = x_for(dt)
            color = (30, 135, 90) if close >= open_p else (195, 60, 55)
            line(x, y_for(low), x, y_for(high), color, 1)
            rect(x - candle_width, y_for(open_p), x + candle_width, y_for(close), color)
    else:
        point_values = [
            (event_dt(trade.decision_event or {}), _safe_float(trade.metadata.get("signal_close"))),
            (trade.entry_time_utc, trade.entry_price),
            (trade.exit_time_utc, trade.exit_price),
        ]
        for dt, price in point_values:
            if dt is None or price is None:
                continue
            current = (x_for(dt), y_for(price))
            rect(current[0] - 5, current[1] - 5, current[0] + 5, current[1] + 5, (35, 85, 170))
            if previous is not None:
                line(previous[0], previous[1], current[0], current[1], (35, 85, 170), 2)
            previous = current

    if trade.entry_time_utc is not None and trade.entry_price is not None:
        x, y = x_for(trade.entry_time_utc), y_for(trade.entry_price)
        rect(x - 7, y - 7, x + 7, y + 7, (10, 10, 10))
    if trade.exit_time_utc is not None and trade.exit_price is not None:
        x, y = x_for(trade.exit_time_utc), y_for(trade.exit_price)
        rect(x - 7, y - 7, x + 7, y + 7, (20, 20, 180))

    _write_png_rgb(output_path, width, height, pixels)


def build_report_payload(
    trade: ClosedTrade,
    bars: Dict[str, List[Dict[str, Any]]],
    features: Dict[str, Any],
    chart_path: Path,
    chart_warning: Optional[str],
    bar_source: str,
    source_warnings: Sequence[str],
) -> Dict[str, Any]:
    grade = quality_grade(trade, features)
    outcome = "WIN" if (trade.pnl or 0.0) > 0 else "LOSS" if (trade.pnl or 0.0) < 0 else "FLAT"
    guard_candidates = build_guard_candidates(trade, features)
    vision_prompt = build_vision_prompt(trade, chart_path, bar_source)
    metadata_keep = {
        key: trade.metadata.get(key)
        for key in (
            "entry_style",
            "signal_close",
            "risk_per_unit",
            "sweep_level",
            "sweep_extreme",
            "sweep_event_key",
            "stage_a_target",
            "adx_entry",
            "displacement_ratio",
            "reclaim_window_sec",
            "expected_rr",
            "fee_adjusted_rr",
            "estimated_net_loss",
            "estimated_net_profit_at_tp",
            "target_net_loss_usd",
            "hard_max_net_loss_usd",
            "tp_profile",
            "win_probability",
            "chop_score",
            "atr_regime_ratio",
        )
        if key in trade.metadata
    }
    return {
        "trade_key": trade.trade_key,
        "symbol": trade.symbol,
        "strategy": trade.strategy,
        "side": trade.side,
        "ticket": trade.ticket,
        "entry_deal": trade.entry_deal,
        "entry_order": trade.entry_order,
        "exit_order": trade.exit_order,
        "entry_time_utc": trade.entry_time_utc.isoformat() if trade.entry_time_utc else None,
        "entry_time_kst": trade.entry_time_utc.astimezone(KST).isoformat() if trade.entry_time_utc else None,
        "exit_time_utc": trade.exit_time_utc.isoformat() if trade.exit_time_utc else None,
        "exit_time_kst": trade.exit_time_utc.astimezone(KST).isoformat() if trade.exit_time_utc else None,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "volume": trade.volume,
        "pnl": trade.pnl,
        "outcome": outcome,
        "quality_grade": grade,
        "reason": trade.reason,
        "sl": trade.sl,
        "tp": trade.tp,
        "strategy_metadata": metadata_keep,
        "entry_quality": {
            "score": features.get("entry_quality_score"),
            "threshold": features.get("entry_quality_threshold"),
            "features": (trade.quality_event or {}).get("features") if trade.quality_event else None,
            "risk_mode": (trade.quality_event or {}).get("risk_mode") if trade.quality_event else None,
        },
        "features": features,
        "quality_flags": {
            "entry_chased_extension": features.get("entry_chased_extension"),
            "entered_into_exhaustion": features.get("entered_into_exhaustion"),
            "entered_against_short_term_momentum": features.get("entered_against_short_term_momentum"),
            "sl_tp_geometry_enough_after_costs": features.get("sl_tp_geometry_enough_after_costs"),
            "clean_reclaim": features.get("clean_reclaim"),
            "clean_reclaim_confirmed": features.get("clean_reclaim_confirmed"),
            "lsr_unconfirmed_reclaim": features.get("lsr_unconfirmed_reclaim"),
            "shallow_reclaim_confirmation": features.get("shallow_reclaim_confirmation"),
            "weak_reclaim_after_deep_sweep": features.get("weak_reclaim_after_deep_sweep"),
            "lsr_unconfirmed_reclaim_chase": features.get("lsr_unconfirmed_reclaim_chase"),
            "late_window_reclaim": features.get("late_window_reclaim"),
            "invalid_reclaim_timing": features.get("invalid_reclaim_timing"),
            "lsr_confirmation_score": features.get("lsr_confirmation_score"),
            "lsr_confirmation_band": features.get("lsr_confirmation_band"),
        },
        "guard_candidates_review_only": guard_candidates,
        "chart_path": chart_path.as_posix(),
        "bar_source": bar_source,
        "bars_summary": {key: len(value) for key, value in bars.items()},
        "bar_schema": BAR_SCHEMA_HELP,
        "warnings": [*source_warnings, *([chart_warning] if chart_warning else [])],
        "vision_prompt": vision_prompt,
    }


def build_markdown(payload: Dict[str, Any]) -> str:
    features = payload["features"]
    flags = payload["quality_flags"]
    guards = "\n".join(f"- {item}" for item in payload["guard_candidates_review_only"])
    outcome = payload["outcome"]
    grade = payload["quality_grade"]
    harsh = (
        "이 타점은 나쁘다. 경계값 통과와 고변위 진입이 겹쳤고, 청산은 빠르게 1R 안팎 손실로 이어졌다."
        if outcome == "LOSS" and grade in {"D", "F"}
        else "손실이지만 현재 계산된 위험 신호는 제한적이다. 같은 feature 조합 반복 여부를 더 봐야 한다."
        if outcome == "LOSS"
        else "수익 패턴은 별도 표본으로 누적해 같은 구조가 반복되는지 확인한다."
    )
    warnings = payload.get("warnings") or []
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 없음"
    metadata = payload.get("strategy_metadata") or {}
    metadata_lines = "\n".join(f"- {key}: `{_fmt(value)}`" for key, value in metadata.items()) or "- N/A"
    feature_lines = "\n".join(f"- {key}: `{_fmt(value)}`" for key, value in features.items())
    flag_lines = "\n".join(f"- {key}: `{_fmt(value)}`" for key, value in flags.items())
    bars_summary = payload.get("bars_summary") or {}
    bars_lines = "\n".join(f"- {key}: `{value}`" for key, value in bars_summary.items()) or "- N/A"

    return f"""# Trade Postmortem: {payload['trade_key']}

## 1. 결론
- 결과: **{outcome}**
- 품질 등급: **{grade}**
- 심볼/방향: `{payload['symbol']} {payload['side']}`
- 진입: `{payload['entry_time_kst']}` @ `{_fmt(payload['entry_price'], 2)}`
- 청산: `{payload['exit_time_kst']}` @ `{_fmt(payload['exit_price'], 2)}`
- PnL: `{_fmt(payload['pnl'], 2)}`
- 차트: `{payload['chart_path']}`
- 바 소스: `{payload.get('bar_source')}`

## 2. 왜 진입했나
- 전략: `{payload.get('strategy')}`
- 사유: `{payload.get('reason')}`
- SL/TP: `{_fmt(payload.get('sl'), 2)}` / `{_fmt(payload.get('tp'), 2)}`
- entry_quality: score=`{_fmt(payload['entry_quality'].get('score'))}`, threshold=`{_fmt(payload['entry_quality'].get('threshold'))}`, risk_mode=`{payload['entry_quality'].get('risk_mode')}`

핵심 메타데이터:
{metadata_lines}

## 3. 왜 수익/손실났나
- 가격 기준 R: `{_fmt(features.get('price_r_multiple'))}`
- PnL 기준 R: `{_fmt(features.get('pnl_r_multiple'))}`
- 진입 implementation shortfall: `{_fmt(features.get('entry_implementation_shortfall_price'), 2)}` price / `{_fmt(features.get('entry_implementation_shortfall_r'))}`R
- 순체결 드래그: `{_fmt(features.get('net_execution_drag_r'))}`R (price-only R - net PnL R)
- 비용 비중: expected `{_fmt(features.get('estimated_cost_to_expected_loss_r'))}`R / realized explicit `{_fmt(features.get('realized_explicit_cost_r'))}`R
- adverse excursion: `{_fmt(features.get('adverse_excursion_price'), 2)}`
- favorable excursion: `{_fmt(features.get('favorable_excursion_price'), 2)}`
- exit type: `{features.get('exit_inferred_type')}`
- displacement_ratio: `{_fmt(features.get('displacement_ratio'))}`
- ADX: `{_fmt(features.get('adx_entry'))}`

## 4. 나쁜 타점 여부
{harsh}

품질 플래그:
{flag_lines}

## 5. 다음에 더 엄격해야 할 조건
{guards}

주의: 위 조건은 검토 추천일 뿐이며, 이 도구는 live config나 주문 로직을 자동 변경하지 않는다.

## 6. 수학적/통계적 저장값
{feature_lines}

## 7. 차트/바 입력
- bar_source: `{payload.get('bar_source')}`
- expected bar schema: {payload.get('bar_schema')}

bars_summary:
{bars_lines}

## Vision Hook
```text
{payload['vision_prompt'].strip()}
```

## Warnings
{warning_text}
"""


def learning_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trade_key": payload["trade_key"],
        "symbol": payload["symbol"],
        "side": payload["side"],
        "strategy": payload["strategy"],
        "entry_time_utc": payload["entry_time_utc"],
        "exit_time_utc": payload["exit_time_utc"],
        "label": "win" if payload["outcome"] == "WIN" else "loss" if payload["outcome"] == "LOSS" else "flat",
        "pnl": payload["pnl"],
        "quality_grade": payload["quality_grade"],
        "bar_source": payload.get("bar_source"),
        "features": payload["features"],
        "strategy_metadata": payload.get("strategy_metadata"),
        "quality_flags": payload["quality_flags"],
    }


def index_row(payload: Dict[str, Any], json_path: Path, md_path: Path, chart_path: Path) -> Dict[str, Any]:
    return {
        "trade_key": payload["trade_key"],
        "symbol": payload["symbol"],
        "ticket": payload["ticket"],
        "entry_time_utc": payload["entry_time_utc"],
        "exit_time_utc": payload["exit_time_utc"],
        "outcome": payload["outcome"],
        "pnl": payload["pnl"],
        "quality_grade": payload["quality_grade"],
        "bar_source": payload.get("bar_source"),
        "json_path": json_path.as_posix(),
        "markdown_path": md_path.as_posix(),
        "chart_path": chart_path.as_posix(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_global_vision_prompt(output_dir: Path) -> Path:
    path = output_dir / VISION_PROMPT_FILE
    text = """# Hermes/Codex Vision Review Hook

For each generated postmortem, open the chart PNG referenced in the JSON/Markdown.
Prefer reports whose `bar_source` is `REAL_BARS_MT5` or `REAL_BARS_FILE`.
Inspect entry/exit timing against candle bodies, wicks, range expansion/compression, local swing waves, pullback depth, exhaustion, and failed continuation.
If `bar_source` is `FALLBACK_EVENT_PATH`, treat the image as a minimal event sketch and explicitly say visual candle evidence is unavailable.
Append visual critique with concrete evidence from the chart only.
Do not place orders, close positions, modify positions, or alter live trading config.
Guard changes are review recommendations until a human approves them.

Bar file input schema:
CSV columns: time,open,high,low,close with optional timeframe,symbol,tick_volume,spread,real_volume.
JSON: list of bars, {"bars": [...]}, or {"M1": [...], "M5": [...]}.
Time may be ISO-8601 or epoch seconds. Rows without timeframe use --bars-timeframe.
"""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return path
    _write_text(path, text)
    return path


def analyze_trades(args: argparse.Namespace) -> Dict[str, Any]:
    events_path = Path(args.events)
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    write_global_vision_prompt(output_dir)

    events = read_jsonl(events_path)
    trades = extract_closed_trades(events, symbol=args.symbol)
    if args.trade_key:
        trades = [trade for trade in trades if trade.trade_key == args.trade_key]

    index_path = output_dir / INDEX_FILE
    indexed_keys = _read_jsonl_keys(index_path)
    mt5_enabled = bool(args.mt5 and not args.no_mt5)
    bars_csv = list(getattr(args, "bars_csv", []) or [])
    bars_json = list(getattr(args, "bars_json", []) or [])
    bars_timeframe = str(getattr(args, "bars_timeframe", "M1") or "M1")
    analyzed: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for trade in reversed(trades):
        if len(analyzed) >= max(0, int(args.limit)):
            break
        if trade.trade_key in indexed_keys and not args.force:
            skipped.append(trade.trade_key)
            continue

        source_warnings: List[str] = []
        file_source = load_bars_from_files(
            bars_csv,
            bars_json,
            default_timeframe=bars_timeframe,
            symbol_filter=trade.symbol,
        )
        source_warnings.extend(file_source.warnings)
        if any(file_source.bars.values()):
            bars = file_source.bars
            bar_source = "REAL_BARS_FILE"
        else:
            bars, mt5_warning = fetch_mt5_bars(
                trade.symbol,
                trade.entry_time_utc,
                trade.exit_time_utc,
                enabled=mt5_enabled,
            )
            if mt5_warning:
                source_warnings.append(mt5_warning)
            if any(bars.values()):
                bar_source = "REAL_BARS_MT5"
            else:
                bar_source = "FALLBACK_EVENT_PATH"
                source_warnings.append(
                    "No real OHLC bars available; using event-derived fallback chart/features."
                )
        features = compute_features(trade, bars)
        chart_path = assets_dir / f"{trade.trade_key}.png"
        chart_warning = generate_chart(trade, bars, features, chart_path)
        json_path = output_dir / f"{trade.trade_key}.json"
        md_path = output_dir / f"{trade.trade_key}.md"
        payload = build_report_payload(
            trade,
            bars,
            features,
            chart_path,
            chart_warning,
            bar_source,
            source_warnings,
        )
        _write_json(json_path, payload)
        _write_text(md_path, build_markdown(payload))
        _upsert_jsonl(index_path, index_row(payload, json_path, md_path, chart_path))
        _upsert_jsonl(output_dir / LEARNING_FILE, learning_row(payload))
        analyzed.append(
            {
                "trade_key": trade.trade_key,
                "json": json_path.as_posix(),
                "markdown": md_path.as_posix(),
                "chart": chart_path.as_posix(),
                "outcome": payload["outcome"],
                "quality_grade": payload["quality_grade"],
                "bar_source": payload["bar_source"],
            }
        )
    return {
        "events_path": events_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "events_read": len(events),
        "closed_trades_found": len(trades),
        "analyzed": analyzed,
        "skipped_indexed": skipped,
        "mt5_enabled": mt5_enabled,
        "bars_csv": bars_csv,
        "bars_json": bars_json,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = analyze_trades(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
