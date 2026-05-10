from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
BOT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BOT_DIR / "runtime"
STATE_FILE = RUNTIME_DIR / "openclaw_event_watcher.state.json"
TRADE_EVENT_FILE = RUNTIME_DIR / "trade_event.json"
HEALTH_ALERT_FILE = RUNTIME_DIR / "health_alert.json"
OPENCLAW_CONFIG_FILE = ROOT_DIR / "openclaw.json"
SESSIONS_FILE = ROOT_DIR / "agents" / "main" / "sessions" / "sessions.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_telegram_config() -> Tuple[Optional[str], Optional[str]]:
    cfg = _read_json(OPENCLAW_CONFIG_FILE)
    bot_token = str(cfg.get("channels", {}).get("telegram", {}).get("botToken", "")).strip()

    env_chat_id = str(os.getenv("OPENCLAW_TELEGRAM_CHAT_ID", "")).strip()
    if env_chat_id:
        return bot_token or None, env_chat_id

    sessions = _read_json(SESSIONS_FILE)
    main_session = sessions.get("agent:main:main", {}) if isinstance(sessions, dict) else {}
    for candidate in (main_session.get("lastTo"), main_session.get("deliveryContext", {}).get("to")):
        raw = str(candidate or "").strip()
        # Expected form: "telegram:6651944098"
        matched = re.match(r"^telegram:(-?\d+)$", raw)
        if matched:
            return bot_token or None, matched.group(1)
    return bot_token or None, None


def _post_telegram(bot_token: str, chat_id: str, text: str, timeout_sec: float = 6.0) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    body = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            return bool(parsed.get("ok", False))
    except Exception:
        return False


def _format_trade_message(payload: Dict[str, Any]) -> str:
    event_type = str(payload.get("event_type", "TRADE")).upper()
    symbol = str(payload.get("symbol", ""))
    side = str(payload.get("side", ""))
    volume = payload.get("volume")
    price = payload.get("price")
    sl = payload.get("sl")
    tp = payload.get("tp")
    expected = payload.get("expected_pnl_usd")
    strategy = str(payload.get("strategy_name", ""))
    ts = str(payload.get("timestamp_utc", ""))

    return (
        "[MT5 EVENT]\n"
        f"type={event_type} symbol={symbol} side={side} vol={volume}\n"
        f"price={price} sl={sl} tp={tp} expected_pnl_usd={expected}\n"
        f"strategy={strategy}\n"
        f"ts={ts}"
    )


def _format_health_message(payload: Dict[str, Any]) -> str:
    severity = str(payload.get("severity", "UNKNOWN")).upper()
    issues = payload.get("issues", [])
    issue_lines = []
    if isinstance(issues, list):
        for item in issues[:3]:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", ""))
            msg = str(item.get("message", ""))
            issue_lines.append(f"- {src}: {msg}")
    issue_text = "\n".join(issue_lines) if issue_lines else "- no issue details"
    ts = str(payload.get("issued_at_utc", ""))

    return (
        "[MT5 HEALTH]\n"
        f"severity={severity}\n"
        f"{issue_text}\n"
        f"ts={ts}"
    )


def _event_changed(state: Dict[str, Any], key: str, content_hash: str) -> bool:
    previous = str(state.get(key, {}).get("hash", ""))
    return previous != content_hash


def _update_state_entry(state: Dict[str, Any], key: str, content_hash: str) -> None:
    state[key] = {
        "hash": content_hash,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(send_health_on_warn: bool) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_json(STATE_FILE)
    if not isinstance(state, dict):
        state = {}

    bot_token, chat_id = _load_telegram_config()
    if not bot_token or not chat_id:
        print("openclaw_event_watcher: missing Telegram config (bot token or chat id).")
        return 1

    sends = 0

    trade_text = _read_text(TRADE_EVENT_FILE)
    if trade_text.strip():
        trade_hash = _sha256_text(trade_text)
        if _event_changed(state, "trade_event", trade_hash):
            trade_payload = _read_json(TRADE_EVENT_FILE)
            message = _format_trade_message(trade_payload)
            if _post_telegram(bot_token, chat_id, message):
                _update_state_entry(state, "trade_event", trade_hash)
                sends += 1

    health_text = _read_text(HEALTH_ALERT_FILE)
    if health_text.strip():
        health_payload = _read_json(HEALTH_ALERT_FILE)
        severity = str(health_payload.get("severity", "")).upper()
        should_send = severity == "BLOCK" or (send_health_on_warn and severity == "WARN")
        if should_send:
            health_hash = _sha256_text(health_text)
            if _event_changed(state, "health_alert", health_hash):
                message = _format_health_message(health_payload)
                if _post_telegram(bot_token, chat_id, message):
                    _update_state_entry(state, "health_alert", health_hash)
                    sends += 1

    _write_json_atomic(STATE_FILE, state)
    print(f"openclaw_event_watcher: sent={sends}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge MT5 runtime events to OpenClaw Telegram channel.")
    parser.add_argument(
        "--send-health-warn",
        action="store_true",
        help="Also send WARN severity from health_alert.json (default: only BLOCK).",
    )
    args = parser.parse_args()
    return run(send_health_on_warn=bool(args.send_health_warn))


if __name__ == "__main__":
    raise SystemExit(main())
