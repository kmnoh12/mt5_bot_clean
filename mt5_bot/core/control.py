from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONTROL_STATE: Dict[str, Any] = {
    "paused": False,
    "manual_halt": False,
    "flatten_requested": False,
    "resume_requested": False,
    "manual_entry": None,
    "intentional_stop_requested": False,
    "intentional_stop_reason": None,
    "intentional_stop_source": None,
    "intentional_stop_requested_at_utc": None,
    "updated_at_utc": None,
}

DESIRED_STATE_RUN = "RUN"
DESIRED_STATE_STOP = "STOP"
_ALLOWED_DESIRED_STATES = {DESIRED_STATE_RUN, DESIRED_STATE_STOP}
DESIRED_STATE_PATH = Path(__file__).resolve().parents[1] / "runtime" / "desired_state.json"
DEFAULT_RUNTIME_CONTROL_PATH = Path(__file__).resolve().parents[1] / "runtime_control.json"
DEFAULT_DESIRED_STATE: Dict[str, Any] = {
    "state": DESIRED_STATE_RUN,
    "updated_at_utc": None,
    "source": "bootstrap",
    "reason": "",
    "metadata": {},
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _normalize_desired_state(raw: Any) -> str:
    value = str(raw or "").strip().upper()
    return value if value in _ALLOWED_DESIRED_STATES else DESIRED_STATE_RUN


def _resolve_desired_state_path(path: Optional[str] = None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return DESIRED_STATE_PATH


def load_desired_state(path: Optional[str] = None) -> Dict[str, Any]:
    target = _resolve_desired_state_path(path)
    state = dict(DEFAULT_DESIRED_STATE)
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                state.update(payload)
        except Exception:
            return dict(DEFAULT_DESIRED_STATE)
    state["state"] = _normalize_desired_state(state.get("state"))
    state["metadata"] = dict(state.get("metadata", {})) if isinstance(state.get("metadata"), dict) else {}
    return state


def ensure_desired_state_file(path: Optional[str] = None) -> Dict[str, Any]:
    target = _resolve_desired_state_path(path)
    state = load_desired_state(path=str(target))
    if not target.exists():
        _atomic_json_write(target, state)
    return state


def write_desired_state(
    state: str,
    *,
    source: str,
    reason: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    target = _resolve_desired_state_path(path)
    payload = load_desired_state(path=str(target))
    normalized_state = _normalize_desired_state(state)
    metadata_payload = dict(metadata or {})

    if normalized_state == DESIRED_STATE_RUN and not bool(metadata_payload.get("force_run")):
        control_state = RuntimeControlChannel(path=str(RuntimeControlChannel.default_path())).load()
        # Keep STOP latched unless an explicit force-run override is provided.
        if bool(control_state.get("manual_halt")) or bool(control_state.get("intentional_stop_requested")):
            payload["state"] = DESIRED_STATE_STOP
            return payload

    payload["state"] = normalized_state
    payload["source"] = str(source or "unknown")
    payload["reason"] = str(reason or "")
    payload["metadata"] = metadata_payload
    payload["updated_at_utc"] = _utc_now_iso()
    _atomic_json_write(target, payload)
    return payload


class RuntimeControlChannel:
    @staticmethod
    def default_path() -> Path:
        return DEFAULT_RUNTIME_CONTROL_PATH

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(dict(DEFAULT_CONTROL_STATE))
        ensure_desired_state_file()

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return dict(DEFAULT_CONTROL_STATE)
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                return dict(DEFAULT_CONTROL_STATE)
        except Exception:
            return dict(DEFAULT_CONTROL_STATE)

        state = dict(DEFAULT_CONTROL_STATE)
        state.update(payload)
        return state

    def save(self, state: Dict[str, Any]) -> None:
        payload = dict(DEFAULT_CONTROL_STATE)
        payload.update(state or {})
        payload["updated_at_utc"] = _utc_now_iso()
        _atomic_json_write(self.path, payload)

    def set_paused(self, value: bool) -> None:
        state = self.load()
        state["paused"] = bool(value)
        self.save(state)

    def set_manual_halt(self, value: bool) -> None:
        state = self.load()
        is_halt = bool(value)
        state["manual_halt"] = is_halt
        if is_halt:
            state["paused"] = True
            state["resume_requested"] = False
        self.save(state)

    def request_manual_halt(self, source: str = "manual", reason: str = "manual_halt") -> None:
        state = self.load()
        state["manual_halt"] = True
        state["paused"] = True
        state["resume_requested"] = False
        state["intentional_stop_requested"] = True
        state["intentional_stop_source"] = str(source or "manual")
        state["intentional_stop_reason"] = str(reason or "manual_halt")
        state["intentional_stop_requested_at_utc"] = _utc_now_iso()
        self.save(state)

    def request_flatten(self) -> None:
        state = self.load()
        state["flatten_requested"] = True
        self.save(state)

    def clear_flatten(self) -> None:
        state = self.load()
        state["flatten_requested"] = False
        self.save(state)

    def request_resume(self) -> None:
        state = self.load()
        state["paused"] = False
        state["manual_halt"] = False
        state["resume_requested"] = True
        state["intentional_stop_requested"] = False
        state["intentional_stop_reason"] = None
        state["intentional_stop_source"] = None
        state["intentional_stop_requested_at_utc"] = None
        self.save(state)

    def request_manual_entry(self, symbol: str, action: str, source: str = "dashboard") -> None:
        state = self.load()
        state["manual_entry"] = {
            "symbol": str(symbol or "").strip().upper(),
            "action": str(action or "").strip().upper(),
            "source": str(source or "dashboard").strip(),
            "requested_at_utc": _utc_now_iso(),
        }
        self.save(state)

    def clear_manual_entry(self) -> None:
        state = self.load()
        state["manual_entry"] = None
        self.save(state)
