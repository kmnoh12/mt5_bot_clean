from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class ExitRetryGuard:
    BACKOFF_SCHEDULE_SECONDS = (30.0, 60.0, 120.0, 300.0)

    def __init__(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        self._next_retry_ts_by_ticket: Dict[str, float] = {}
        self._attempts_by_ticket: Dict[str, int] = {}
        self._last_reason_by_ticket: Dict[str, str] = {}
        self._restore_snapshot(snapshot or {})

    @staticmethod
    def _ticket_key(ticket: Any) -> str:
        try:
            return str(int(ticket))
        except Exception:
            return str(ticket)

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        next_retry = snapshot.get("next_retry_ts_by_ticket")
        attempts = snapshot.get("attempts_by_ticket")
        reasons = snapshot.get("last_reason_by_ticket")
        if isinstance(next_retry, dict):
            for key, value in next_retry.items():
                try:
                    self._next_retry_ts_by_ticket[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        if isinstance(attempts, dict):
            for key, value in attempts.items():
                try:
                    self._attempts_by_ticket[str(key)] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
        if isinstance(reasons, dict):
            for key, value in reasons.items():
                self._last_reason_by_ticket[str(key)] = str(value or "")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "next_retry_ts_by_ticket": dict(self._next_retry_ts_by_ticket),
            "attempts_by_ticket": dict(self._attempts_by_ticket),
            "last_reason_by_ticket": dict(self._last_reason_by_ticket),
        }

    def clear(self, ticket: Any) -> None:
        key = self._ticket_key(ticket)
        self._next_retry_ts_by_ticket.pop(key, None)
        self._attempts_by_ticket.pop(key, None)
        self._last_reason_by_ticket.pop(key, None)

    def should_allow(self, ticket: Any, reason: str, now_ts: float) -> Tuple[bool, float, int]:
        key = self._ticket_key(ticket)
        reason_text = str(reason or "")
        last_reason = self._last_reason_by_ticket.get(key, "")
        next_ts = float(self._next_retry_ts_by_ticket.get(key, 0.0) or 0.0)
        attempts = int(self._attempts_by_ticket.get(key, 0))

        if reason_text != last_reason:
            self._last_reason_by_ticket[key] = reason_text
            self._next_retry_ts_by_ticket[key] = 0.0
            self._attempts_by_ticket[key] = 0
            return True, 0.0, 0

        if now_ts < next_ts:
            return False, max(0.0, next_ts - now_ts), attempts
        return True, 0.0, attempts

    def on_attempt(self, ticket: Any, reason: str, now_ts: float, success: bool) -> Dict[str, Any]:
        key = self._ticket_key(ticket)
        reason_text = str(reason or "")
        self._last_reason_by_ticket[key] = reason_text
        if success:
            self.clear(key)
            return {
                "ticket": key,
                "attempt": 0,
                "backoff_seconds": 0.0,
                "next_retry_ts": now_ts,
                "reason": reason_text,
                "success": True,
            }

        attempts = int(self._attempts_by_ticket.get(key, 0)) + 1
        self._attempts_by_ticket[key] = attempts
        idx = min(max(0, attempts - 1), len(self.BACKOFF_SCHEDULE_SECONDS) - 1)
        backoff = float(self.BACKOFF_SCHEDULE_SECONDS[idx])
        next_retry = float(now_ts + backoff)
        self._next_retry_ts_by_ticket[key] = next_retry
        return {
            "ticket": key,
            "attempt": attempts,
            "backoff_seconds": backoff,
            "next_retry_ts": next_retry,
            "reason": reason_text,
            "success": False,
        }
