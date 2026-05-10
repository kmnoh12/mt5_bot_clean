from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.models import ExternalSignal, parse_action
from signals.base import SignalSource


LOGGER = logging.getLogger(__name__)


class SocketSignalSource(SignalSource):
    """
    TCP JSON-lines signal ingestion endpoint.

    Each line must be one JSON object:
    {"id":"sig-1","symbol":"BTCUSD","action":"BUY","reason":"llm"}
    """

    def __init__(self, host: str, port: int, max_queue_size: int = 500) -> None:
        self.host = str(host)
        self.port = int(port)
        self.max_queue_size = max(10, int(max_queue_size))

        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True, name="SignalSocketServer")
        self._thread.start()

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(5)
            server.settimeout(1.0)
            LOGGER.info("Socket signal server listening on %s:%s", self.host, self.port)
            while not self._stop_event.is_set():
                try:
                    conn, _addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._read_connection(conn)
        except Exception:
            LOGGER.exception("Socket signal source failed.")
        finally:
            try:
                server.close()
            except Exception:
                pass

    def _read_connection(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(1.0)
            buffer = ""
            while not self._stop_event.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break

                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._push_line(line)

    def _push_line(self, line: str) -> None:
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(line)
            except queue.Full:
                # Drop if still full under pressure.
                return

    def _parse_signal(self, payload: Dict[str, Any]) -> Optional[ExternalSignal]:
        symbol = str(payload.get("symbol", "")).strip()
        if not symbol:
            return None
        action = parse_action(payload.get("action") or payload.get("side"))
        if action is None:
            return None
        signal_id = str(payload.get("id", "")).strip() or str(uuid.uuid4())

        raw_ts = payload.get("created_at") or payload.get("timestamp")
        created_at = datetime.now(timezone.utc)
        if raw_ts:
            ts = str(raw_ts).strip()
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(ts)
                created_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except Exception:
                created_at = datetime.now(timezone.utc)

        return ExternalSignal(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            reason=str(payload.get("reason", "") or ""),
            strategy=str(payload.get("strategy", "external") or "external"),
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            volume=float(payload["volume"]) if payload.get("volume") is not None else None,
            ttl_seconds=int(payload.get("ttl_seconds", 300) or 300),
            created_at_utc=created_at.astimezone(timezone.utc),
            metadata={k: v for k, v in payload.items() if k not in {"id", "symbol", "action", "side"}},
        )

    def poll(self) -> List[ExternalSignal]:
        lines: List[str] = []
        while True:
            try:
                lines.append(self._queue.get_nowait())
            except queue.Empty:
                break

        signals: List[ExternalSignal] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Invalid JSON line from socket signal source.")
                continue
            if not isinstance(payload, dict):
                continue
            signal = self._parse_signal(payload)
            if signal is not None:
                signals.append(signal)
        return signals

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

