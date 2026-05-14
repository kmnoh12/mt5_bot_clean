from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_recent_events import summarize_events


def test_summarize_events_filters_telemetry_and_counts_reasons(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    rows = [
        {"event": "v4_opportunity_reports_written", "symbol": "BTCUSD"},
        {"event": "decision", "symbol": "BTCUSD", "reason": "LSR_SELL_ENTRY", "account": "secret"},
        {"event": "order_skip", "symbol": "BTCUSD", "reason": "M5_CONFIRM_BLOCK", "server": "secret"},
        {"event": "decision", "symbol": "ETHUSD", "reason": "OTHER"},
    ]
    events.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = summarize_events(events, symbol="BTCUSD")

    assert summary["events_scanned"] == 4
    assert summary["interesting_events"] == 2
    assert summary["reason_counts"] == {"LSR_SELL_ENTRY": 1, "M5_CONFIRM_BLOCK": 1}
    assert summary["recent_interesting"][0]["event"] == "decision"
    assert "account" not in summary["recent_interesting"][0]
    assert "server" not in summary["recent_interesting"][1]
