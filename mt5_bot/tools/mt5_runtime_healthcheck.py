from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REDACT_KEYS = {"account", "login", "server", "password", "token", "chat_id", "api_key", "secret"}
IMPORTANT_EVENTS = {
    "decision",
    "order_skip",
    "order_submit",
    "order_filled",
    "runtime_config_applied",
    "desired_state_applied",
    "watchdog_action",
    "error",
    "exception",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # read-only diagnostic; keep going on partial/corrupt files
        return {"_error": f"{type(exc).__name__}: {exc}"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: ("<redacted>" if k.lower() in REDACT_KEYS else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 0:
        return "future timestamp"
    if seconds < 120:
        return f"{seconds:.1f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def heartbeat_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    summary: dict[str, Any] = {"path": str(path), "present": data is not None}
    if not isinstance(data, dict):
        return summary
    ts = data.get("ts") or data.get("timestamp") or data.get("updated_at_utc")
    age = None
    if isinstance(ts, (int, float)):
        age = time.time() - float(ts)
    elif isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc).timestamp() - parsed.timestamp()
        except ValueError:
            pass
    summary.update(
        {
            "state": data.get("state") or data.get("status"),
            "timestamp": ts,
            "age": format_age(age),
            "raw": redact(data),
        }
    )
    return summary


def iter_recent_events(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if max_bytes > 0 and len(data) > max_bytes:
        data = data[-max_bytes:]
        first_newline = data.find(b"\n")
        if first_newline >= 0:
            data = data[first_newline + 1 :]
    events: list[dict[str, Any]] = []
    for line in data.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def event_summary(path: Path, max_bytes: int, last: int) -> dict[str, Any]:
    events = iter_recent_events(path, max_bytes)
    event_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    important: list[dict[str, Any]] = []
    for item in events:
        event_name = str(item.get("event") or item.get("type") or item.get("name") or "")
        event_counts[event_name] += 1
        reason = item.get("reason") or item.get("skip_reason") or item.get("error")
        if reason:
            reason_counts[str(reason)] += 1
        if event_name in IMPORTANT_EVENTS or reason:
            important.append(redact(item))
    compact = []
    for item in important[-last:]:
        compact.append(
            {
                "ts": item.get("ts_utc") or item.get("ts_kst") or item.get("ts"),
                "event": item.get("event") or item.get("type") or item.get("name"),
                "reason": item.get("reason") or item.get("skip_reason") or item.get("error"),
                "state": item.get("state") or item.get("status"),
            }
        )
    return {
        "path": str(path),
        "present": path.exists(),
        "events_scanned": len(events),
        "event_counts": dict(event_counts.most_common(8)),
        "reason_counts": dict(reason_counts.most_common(8)),
        "recent_important": compact,
    }


def powershell_json(script: str, timeout: int = 5) -> Any:
    exe = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not exe:
        return {"available": False, "reason": "PowerShell not found"}
    try:
        result = subprocess.run(
            [exe, "-NoProfile", "-Command", script],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {"available": True, "error": result.stderr.strip() or result.stdout.strip()}
    output = result.stdout.strip()
    if not output:
        return {"available": True, "items": []}
    try:
        return {"available": True, "items": json.loads(output)}
    except json.JSONDecodeError:
        return {"available": True, "raw": output}


def windows_process_evidence() -> Any:
    script = r"""
$names = @('python','python3','terminal64','terminal','MetaTrader','metatrader','mt5')
Get-Process -ErrorAction SilentlyContinue |
  Where-Object { $names -contains $_.ProcessName } |
  Select-Object ProcessName, Id, StartTime |
  ConvertTo-Json -Depth 3
"""
    return powershell_json(script)


def battery_summary() -> dict[str, Any]:
    script = r"""
Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue |
  Select-Object EstimatedChargeRemaining, BatteryStatus |
  ConvertTo-Json -Depth 3
"""
    data = powershell_json(script)
    warning = None
    items = data.get("items") if isinstance(data, dict) else None
    batteries = items if isinstance(items, list) else ([items] if isinstance(items, dict) else [])
    for battery in batteries:
        charge = battery.get("EstimatedChargeRemaining")
        if isinstance(charge, int) and charge <= 15:
            warning = f"battery very low: {charge}%"
    return {"evidence": data, "warning": warning}


def print_section(title: str) -> None:
    print(f"\n## {title}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only MT5 runtime healthcheck. Summarizes local runtime files, "
            "recent event reasons, optional Windows process evidence, and battery state."
        )
    )
    parser.add_argument("--events", type=Path, default=repo_root() / "events.jsonl")
    parser.add_argument("--max-event-bytes", type=int, default=5_000_000)
    parser.add_argument("--last-events", type=int, default=8)
    parser.add_argument("--no-powershell", action="store_true", help="Skip Windows PowerShell probes")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown-ish text")
    args = parser.parse_args(argv)

    root = repo_root()
    runtime_dir = root / "runtime"
    report: dict[str, Any] = {
        "repo_root": str(root),
        "runtime_dir": str(runtime_dir),
        "paths": {
            "heartbeat": str(runtime_dir / "heartbeat.json"),
            "desired_state": str(runtime_dir / "desired_state.json"),
            "runtime_lock": str(root / "runtime.lock"),
            "runtime_control": str(root / "runtime_control.json"),
            "events": str(args.events),
        },
        "heartbeat": heartbeat_summary(runtime_dir / "heartbeat.json"),
        "desired_state": redact(read_json(runtime_dir / "desired_state.json")),
        "runtime_lock": redact(read_json(root / "runtime.lock")),
        "runtime_control": redact(read_json(root / "runtime_control.json")),
        "events": event_summary(args.events, args.max_event_bytes, args.last_events),
    }
    if not args.no_powershell:
        report["windows_processes"] = windows_process_evidence()
        report["battery"] = battery_summary()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("# MT5 Runtime Healthcheck")
    print("mode: read-only")
    print_section("Paths")
    for key, value in report["paths"].items():
        print(f"- {key}: {value}")
    print_section("Heartbeat")
    hb = report["heartbeat"]
    print(f"- present: {hb.get('present')}")
    print(f"- state: {hb.get('state')}")
    print(f"- timestamp: {hb.get('timestamp')}")
    print(f"- age: {hb.get('age')}")
    print_section("Desired State")
    print(json.dumps(report["desired_state"], ensure_ascii=False, indent=2, sort_keys=True))
    print_section("Runtime Lock")
    print(json.dumps(report["runtime_lock"], ensure_ascii=False, indent=2, sort_keys=True))
    print_section("Recent Event Reasons")
    events = report["events"]
    print(f"- events_scanned: {events['events_scanned']}")
    print(f"- event_counts: {events['event_counts']}")
    print(f"- reason_counts: {events['reason_counts']}")
    for item in events["recent_important"]:
        print(f"- {json.dumps(item, ensure_ascii=False, sort_keys=True)}")
    if "windows_processes" in report:
        print_section("Windows Process Evidence")
        print(json.dumps(report["windows_processes"], ensure_ascii=False, indent=2, sort_keys=True))
    if "battery" in report:
        print_section("Battery")
        print(json.dumps(report["battery"], ensure_ascii=False, indent=2, sort_keys=True))
        warning = report["battery"].get("warning")
        if warning:
            print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
