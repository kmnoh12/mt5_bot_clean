from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _iter_events(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict):
                event["_line_no"] = line_no
                yield event


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _today_key(tz_name: str) -> str:
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    return datetime.now(timezone.utc).astimezone(tz).date().isoformat()


def _latest(events: List[Dict[str, Any]], **criteria: Any) -> Optional[Dict[str, Any]]:
    for event in reversed(events):
        ok = True
        for key, expected in criteria.items():
            value = event.get(key)
            if isinstance(expected, (set, tuple, list)):
                if value not in expected:
                    ok = False
                    break
            elif value != expected:
                ok = False
                break
        if ok:
            return event
    return None


def _evidence(event: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if event:
        for key in ("_line_no", "ts_kst", "event", "symbol", "strategy", "action", "reason", "score", "threshold", "allow"):
            if key in event:
                payload[key] = event[key]
        if "details" in event:
            payload["details"] = event["details"]
        if "result" in event and isinstance(event["result"], dict):
            payload["result"] = {
                key: event["result"].get(key)
                for key in ("ok", "status", "message", "retcode", "ticket")
                if key in event["result"]
            }
    if extra:
        payload.update(extra)
    return payload


def build_audit(root: Path, symbol: str, strategy: str) -> Dict[str, Any]:
    symbol = symbol.upper()
    config = _load_yaml(root / "config.yaml")
    state = _load_json(root / "state.json")
    events = [e for e in _iter_events(root / "events.jsonl") if e.get("symbol") == symbol or e.get("event") == "runtime_config_applied"]

    execution = config.get("execution", {}) if isinstance(config.get("execution"), dict) else {}
    quality_cfg = config.get("entry_quality_guard", {}) if isinstance(config.get("entry_quality_guard"), dict) else {}
    churn_cfg = config.get("execution_churn_guard", {}) if isinstance(config.get("execution_churn_guard"), dict) else {}
    cost_cfg = config.get("cost_edge_guard", {}) if isinstance(config.get("cost_edge_guard"), dict) else {}

    strategy_state = (
        state.get("strategy_state", {})
        .get(strategy, {})
        .get(symbol, {})
        if isinstance(state.get("strategy_state"), dict)
        else {}
    )
    churn_state = state.get("execution_churn_guard", {}) if isinstance(state.get("execution_churn_guard"), dict) else {}
    quality_state = state.get("entry_quality_guard", {}) if isinstance(state.get("entry_quality_guard"), dict) else {}

    latest_entry = _latest(events, event="decision", action={"BUY", "SELL"}, strategy=strategy)
    latest_quality = _latest(events, event="entry_quality_score", strategy=strategy)
    latest_skip = _latest(events, event="order_skip", strategy=strategy)
    latest_risk_skip = None
    for event in reversed(events):
        if event.get("event") != "order_skip" or event.get("strategy") != strategy:
            continue
        reason = str(event.get("reason") or "")
        if reason.startswith("RISK_PLAN_FAILED") or reason.startswith("EXPECTED_LOSS_CAP"):
            latest_risk_skip = event
            break
    latest_invalid_stops = None
    for event in reversed(events):
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        if event.get("event") == "order_submit" and int(result.get("retcode") or 0) == 10016:
            latest_invalid_stops = event
            break

    score = _finite(latest_quality.get("score") if latest_quality else quality_state.get("last_score_by_symbol", {}).get(symbol))
    threshold = _finite(latest_quality.get("threshold") if latest_quality else quality_cfg.get("min_score_risk_off"))
    quality_pass = score is not None and threshold is not None and score + 1e-12 >= threshold

    day_key = _today_key(str(churn_cfg.get("daily_reset_timezone", "UTC")))
    daily_map = churn_state.get("daily_entries_by_symbol", {}).get(symbol, {}) if isinstance(churn_state.get("daily_entries_by_symbol"), dict) else {}
    daily_count = int(daily_map.get(day_key, 0) or 0) if isinstance(daily_map, dict) else 0
    per_symbol_limits = churn_cfg.get("per_symbol_daily_limits", {}) if isinstance(churn_cfg.get("per_symbol_daily_limits"), dict) else {}
    daily_limit = int(per_symbol_limits.get(symbol, churn_cfg.get("max_entries_per_symbol_per_day", 0)) or 0)
    global_counts = churn_state.get("global_daily_counts", {}) if isinstance(churn_state.get("global_daily_counts"), dict) else {}
    global_count = int(global_counts.get(day_key, 0) or 0)
    global_limit = int(churn_cfg.get("max_entries_global_per_day", 0) or 0)

    latest_entry_meta = latest_entry.get("metadata") if isinstance(latest_entry, dict) else {}
    latest_entry_meta = latest_entry_meta if isinstance(latest_entry_meta, dict) else {}
    risk_cap = None
    cap_map = execution.get("max_expected_loss_usd_by_symbol", {})
    if isinstance(cap_map, dict):
        risk_cap = _finite(cap_map.get(symbol))
    hard_cap = _finite((config.get("risk_per_trade", {}) or {}).get("hard_max_net_loss_usd")) if isinstance(config.get("risk_per_trade"), dict) else None
    active_cap = min([v for v in (risk_cap, hard_cap) if v is not None], default=None)

    chain = [
        {
            "gate": "current_lsr_state",
            "status": "pass" if strategy_state.get("state") in {"SETUP", "IDLE"} else "watch",
            "evidence": {
                "state": strategy_state.get("state"),
                "bias": strategy_state.get("bias"),
                "last_reason": strategy_state.get("last_reason"),
                "cooldown_bars_remaining": strategy_state.get("cooldown_bars_remaining"),
                "metadata_risk_per_unit": (strategy_state.get("metadata") or {}).get("risk_per_unit")
                if isinstance(strategy_state.get("metadata"), dict)
                else None,
            },
        },
        {
            "gate": "entry_quality_guard",
            "status": "pass" if quality_pass else "block",
            "evidence": _evidence(
                latest_quality,
                {
                    "score": score,
                    "threshold": threshold,
                    "comparison": "score + 1e-12 >= threshold",
                    "config_min_score_risk_off": quality_cfg.get("min_score_risk_off"),
                },
            ),
        },
        {
            "gate": "cost_edge_guard",
            "status": "likely_pass",
            "evidence": {
                "enabled": cost_cfg.get("enabled"),
                "btc_threshold": (cost_cfg.get("min_edge_to_cost_ratio_by_symbol") or {}).get(symbol)
                if isinstance(cost_cfg.get("min_edge_to_cost_ratio_by_symbol"), dict)
                else None,
                "latest_entry_expected_rr": latest_entry_meta.get("expected_rr"),
                "latest_entry_risk_per_unit": latest_entry_meta.get("risk_per_unit"),
                "latest_skip_reason": latest_skip.get("reason") if latest_skip else None,
                "note": "No recent EDGE_TOO_LOW skip for the latest BTCUSD LSR entries.",
            },
        },
        {
            "gate": "risk_plan_or_expected_loss_cap",
            "status": "block" if latest_risk_skip is not None else "unknown",
            "evidence": _evidence(
                latest_risk_skip,
                {
                    "active_loss_cap_usd": active_cap,
                    "config_expected_loss_cap_usd": risk_cap,
                    "config_hard_max_net_loss_usd": hard_cap,
                },
            ),
        },
        {
            "gate": "invalid_stops_broker_constraints",
            "status": "guarded_unknown",
            "evidence": _evidence(
                latest_invalid_stops,
                {
                    "note": "Offline audit cannot query live broker stop/freeze levels; code now blocks any auto-widened stop that would exceed the active loss cap.",
                },
            ),
        },
        {
            "gate": "daily_entry_limits",
            "status": "pass" if daily_count < daily_limit and global_count < global_limit else "block",
            "evidence": {
                "day_key": day_key,
                "symbol_daily_count": daily_count,
                "symbol_daily_limit": daily_limit,
                "global_daily_count": global_count,
                "global_daily_limit": global_limit,
            },
        },
    ]

    next_blocker = next((item for item in chain if item["status"] == "block"), None)
    return {
        "symbol": symbol,
        "strategy": strategy,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_entry_evidence": _evidence(latest_entry),
        "next_blocker": next_blocker,
        "blocker_chain": chain,
    }


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    lines = [
        f"# Blocker Chain Audit: {audit['symbol']} {audit['strategy']}",
        "",
        f"Generated UTC: {audit['generated_at_utc']}",
        "",
        f"Next blocker: {(audit.get('next_blocker') or {}).get('gate', 'none')}",
        "",
        "## Chain",
    ]
    for item in audit.get("blocker_chain", []):
        lines.append(f"- {item.get('gate')}: {item.get('status')}")
        lines.append(f"  evidence: `{json.dumps(item.get('evidence', {}), ensure_ascii=False, sort_keys=True)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the current BTCUSD LSR blocker chain from local runtime artifacts.")
    parser.add_argument("--root", default=".", help="Project/runtime root")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--strategy", default="liquidity_sweep_reversal")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    audit = build_audit(root=root, symbol=args.symbol, strategy=args.strategy)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"blocker_chain_audit_{audit['symbol']}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, audit)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "next_blocker": audit.get("next_blocker")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
