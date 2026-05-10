from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


class JsonStore:
    def __init__(self, state_path: Path, events_path: Path) -> None:
        self.state_path = Path(state_path)
        self.events_path = Path(events_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_write_attempts = 8
        self._base_retry_delay_seconds = 0.05

    @staticmethod
    def _is_retryable_write_error(exc: BaseException) -> bool:
        if isinstance(exc, PermissionError):
            return True
        if isinstance(exc, OSError):
            if getattr(exc, "winerror", None) in {5, 32}:
                return True
            if getattr(exc, "errno", None) in {5, 13}:
                return True
        return False

    def _sleep_retry_backoff(self, attempt: int) -> None:
        time.sleep(self._base_retry_delay_seconds * max(1, attempt))

    def _temp_path(self, target: Path) -> Path:
        return target.with_name(
            f"{target.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        )

    def _atomic_dump_json(self, target: Path, payload: Dict[str, Any]) -> None:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self._max_write_attempts + 1):
            tmp = self._temp_path(target)
            try:
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(str(tmp), str(target))
                return
            except Exception as exc:
                last_exc = exc
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                if attempt >= self._max_write_attempts or not self._is_retryable_write_error(exc):
                    raise
                self._sleep_retry_backoff(attempt)
        if last_exc is not None:
            raise last_exc

    def load_state(self) -> Dict[str, Any]:
        for attempt in range(1, self._max_write_attempts + 1):
            if not self.state_path.exists():
                return {}
            try:
                with self.state_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
                return {}
            except FileNotFoundError:
                return {}
            except json.JSONDecodeError:
                LOGGER.warning("State file JSON decode error: %s", self.state_path)
                return {}
            except Exception as exc:
                if attempt >= self._max_write_attempts or not self._is_retryable_write_error(exc):
                    LOGGER.exception("Failed reading state file: %s", self.state_path)
                    return {}
                self._sleep_retry_backoff(attempt)
        return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        payload = dict(state or {})
        now_utc = datetime.now(timezone.utc)
        payload["updated_at_utc"] = now_utc.isoformat()
        payload["updated_at_kst"] = now_utc.astimezone(timezone(timedelta(hours=9))).isoformat()
        try:
            self._atomic_dump_json(self.state_path, payload)
        except Exception:
            LOGGER.exception("Failed writing state file: %s", self.state_path)

    @staticmethod
    def _should_persist_event(payload: Dict[str, Any]) -> bool:
        # Suppress high-frequency idle telemetry that bloats events.jsonl.
        return str(payload.get("reason", "") or "") != "NO_SWEEP_SETUP"

    def append_event(self, event: Dict[str, Any]) -> None:
        payload = dict(event or {})
        if not self._should_persist_event(payload):
            return
        now_utc = datetime.now(timezone.utc)
        payload["ts_utc"] = now_utc.isoformat()
        payload["ts_kst"] = now_utc.astimezone(timezone(timedelta(hours=9))).isoformat()
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        for attempt in range(1, self._max_write_attempts + 1):
            try:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                return
            except Exception as exc:
                if attempt >= self._max_write_attempts or not self._is_retryable_write_error(exc):
                    LOGGER.exception("Failed appending event log: %s", self.events_path)
                    return
                self._sleep_retry_backoff(attempt)
