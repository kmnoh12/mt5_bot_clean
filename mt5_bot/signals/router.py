from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from core.models import ExternalSignal
from signals.base import SignalSource
from signals.json_file_source import JsonFileSignalSource
from signals.socket_source import SocketSignalSource


LOGGER = logging.getLogger(__name__)


class ExternalSignalRouter:
    def __init__(self, config: Dict[str, Any], processed_ids: Optional[Iterable[str]] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.allowed_symbols: Set[str] = {str(v).strip() for v in cfg.get("allowed_symbols", []) if str(v).strip()}
        self._processed_ids: List[str] = []
        self._processed_limit = 5000
        for signal_id in processed_ids or []:
            self.mark_processed(str(signal_id))

        self.source_name = str(cfg.get("source", "none")).strip().lower()
        self.source: Optional[SignalSource] = None

        if not self.enabled:
            return

        if self.source_name == "json_file":
            json_cfg = cfg.get("json_file", {})
            path = Path(str(json_cfg.get("path", "./signals/inbox.json")))
            consume_mode = str(json_cfg.get("consume_mode", "mark_used"))
            self.source = JsonFileSignalSource(path=path, consume_mode=consume_mode)
        elif self.source_name == "socket":
            socket_cfg = cfg.get("socket", {})
            host = str(socket_cfg.get("host", "127.0.0.1"))
            port = int(socket_cfg.get("port", 8765))
            max_queue_size = int(socket_cfg.get("max_queue_size", 500))
            self.source = SocketSignalSource(host=host, port=port, max_queue_size=max_queue_size)
        elif self.source_name == "llm_assist":
            # Internal LLM assist is handled in runtime decision gating.
            self.source = None
            LOGGER.info("External signal source 'llm_assist' delegated to runtime internal service.")
        else:
            LOGGER.info("External signal source disabled (source=%s).", self.source_name)

    def mark_processed(self, signal_id: str) -> None:
        if not signal_id:
            return
        self._processed_ids.append(signal_id)
        if len(self._processed_ids) > self._processed_limit:
            self._processed_ids = self._processed_ids[-self._processed_limit :]

    def _is_processed(self, signal_id: str) -> bool:
        return signal_id in self._processed_ids

    def poll(self) -> List[ExternalSignal]:
        if not self.enabled or self.source is None:
            return []

        output: List[ExternalSignal] = []
        for signal in self.source.poll():
            if signal.is_expired():
                continue
            if self.allowed_symbols and signal.symbol not in self.allowed_symbols:
                continue
            if self._is_processed(signal.signal_id):
                continue
            output.append(signal)
        return output

    def processed_snapshot(self) -> List[str]:
        return list(self._processed_ids)

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
