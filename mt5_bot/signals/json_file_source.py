from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import ExternalSignal, parse_action
from signals.base import SignalSource


LOGGER = logging.getLogger(__name__)


class JsonFileSignalSource(SignalSource):
    def __init__(self, path: Path, consume_mode: str = "mark_used") -> None:
        self.path = Path(path)
        self.consume_mode = str(consume_mode or "mark_used").strip().lower()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_ids: List[str] = []
        self._seen_limit = 5000

    def _mark_seen(self, signal_id: str) -> None:
        self._seen_ids.append(signal_id)
        if len(self._seen_ids) > self._seen_limit:
            self._seen_ids = self._seen_ids[-self._seen_limit :]

    def _already_seen(self, signal_id: str) -> bool:
        return signal_id in self._seen_ids

    def _parse_datetime(self, payload: Dict[str, Any]) -> datetime:
        raw = payload.get("created_at") or payload.get("timestamp")
        if raw is None:
            return datetime.now(timezone.utc)
        text = str(raw).strip()
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _parse_signal(self, payload: Dict[str, Any]) -> Optional[ExternalSignal]:
        symbol = str(payload.get("symbol", "")).strip()
        if not symbol:
            return None

        action = parse_action(payload.get("action") or payload.get("side"))
        if action is None:
            return None

        signal_id = str(payload.get("id", "")).strip() or str(uuid.uuid4())
        return ExternalSignal(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            reason=str(payload.get("reason", "") or ""),
            strategy=str(payload.get("strategy", "external") or "external"),
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            volume=float(payload["volume"]) if payload.get("volume") is not None else None,
            ttl_seconds=int(payload.get("ttl_seconds", 300) or 300),
            created_at_utc=self._parse_datetime(payload),
            metadata={k: v for k, v in payload.items() if k not in {"id", "symbol", "action", "side"}},
        )

    def _extract_payloads(self, raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            if "signals" in raw and isinstance(raw["signals"], list):
                return [item for item in raw["signals"] if isinstance(item, dict)]
            return [raw]
        return []

    def poll(self) -> List[ExternalSignal]:
        if not self.path.exists():
            return []

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except json.JSONDecodeError:
            LOGGER.warning("Signal file JSON decode error: %s", self.path)
            return []
        except Exception:
            LOGGER.exception("Failed reading signal file: %s", self.path)
            return []

        payloads = self._extract_payloads(raw)
        parsed: List[ExternalSignal] = []
        for payload in payloads:
            signal = self._parse_signal(payload)
            if signal is None:
                continue
            if self._already_seen(signal.signal_id):
                continue
            self._mark_seen(signal.signal_id)
            parsed.append(signal)

        if parsed and self.consume_mode == "truncate":
            try:
                with self.path.open("w", encoding="utf-8") as handle:
                    json.dump([], handle, ensure_ascii=False, indent=2)
            except Exception:
                LOGGER.exception("Failed truncating signal file after consume: %s", self.path)

        return parsed
