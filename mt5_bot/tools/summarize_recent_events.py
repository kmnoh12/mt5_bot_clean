from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

INTERESTING_EVENTS = {
    "decision",
    "order_skip",
    "entry_quality_score",
    "order_submit",
    "order_filled",
    "runtime_config_applied",
}
REDACT_KEYS = {"account", "login", "server", "password", "token", "chat_id"}


def iter_recent_events(path: Path, *, max_bytes: int = 5_000_000) -> Iterable[dict[str, Any]]:
    """Yield JSON events from the end of an events.jsonl file without live broker calls."""
    data = path.read_bytes()
    if max_bytes > 0 and len(data) > max_bytes:
        data = data[-max_bytes:]
        first_newline = data.find(b"\n")
        if first_newline >= 0:
            data = data[first_newline + 1 :]
    for line in data.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            yield item


def summarize_events(
    path: Path,
    *,
    interesting_events: set[str] | None = None,
    symbol: str | None = None,
    max_bytes: int = 5_000_000,
    last: int = 12,
) -> dict[str, Any]:
    interesting_events = interesting_events or INTERESTING_EVENTS
    total = 0
    event_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    interesting: list[dict[str, Any]] = []
    for event in iter_recent_events(path, max_bytes=max_bytes):
        total += 1
        event_name = str(event.get("event") or event.get("type") or event.get("name") or "")
        event_counts[event_name] += 1
        if symbol and event.get("symbol") != symbol:
            continue
        if event_name not in interesting_events:
            continue
        reason = event.get("reason")
        if reason:
            reason_counts[str(reason)] += 1
        interesting.append(redact_event(event))
    return {
        "path": str(path),
        "events_scanned": total,
        "interesting_events": len(interesting),
        "event_counts": dict(event_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "recent_interesting": interesting[-last:],
    }


def redact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in event.items() if k.lower() not in REDACT_KEYS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize recent non-telemetry MT5 events from events.jsonl. "
            "Read-only; makes no broker/API calls."
        )
    )
    parser.add_argument("events", type=Path, help="Path to events.jsonl")
    parser.add_argument("--symbol", help="Optional symbol filter, e.g. BTCUSD")
    parser.add_argument("--max-bytes", type=int, default=5_000_000)
    parser.add_argument("--last", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Print full JSON summary")
    args = parser.parse_args(argv)

    summary = summarize_events(
        args.events,
        symbol=args.symbol,
        max_bytes=args.max_bytes,
        last=args.last,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            f"events_scanned={summary['events_scanned']} "
            f"interesting_events={summary['interesting_events']}"
        )
        print("event_counts=", summary["event_counts"])
        print("reason_counts=", summary["reason_counts"])
        for item in summary["recent_interesting"]:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
