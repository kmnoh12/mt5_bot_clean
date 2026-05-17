from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


class JsonStorage:
    def __init__(self, state_path: Path, events_path: Path) -> None:
        self.state_path = Path(state_path)
        self.events_path = Path(events_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
            LOGGER.warning("State file is not a dictionary. Resetting state.")
            return {}
        except json.JSONDecodeError:
            LOGGER.exception("State file JSON decode failed: %s", self.state_path)
            return {}
        except Exception:
            LOGGER.exception("Failed reading state file: %s", self.state_path)
            return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        payload = dict(state or {})
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            temp_path.replace(self.state_path)
        except Exception:
            LOGGER.exception("Failed writing state file: %s", self.state_path)
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                LOGGER.exception("Failed removing temp state file: %s", temp_path)

    def append_event(self, event: Dict[str, Any]) -> None:
        payload = dict(event or {})
        payload["ts_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception:
            LOGGER.exception("Failed appending event log: %s", self.events_path)
