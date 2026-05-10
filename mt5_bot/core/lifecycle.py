from __future__ import annotations

import atexit
import logging
import signal
import threading
from typing import Callable, Optional


LOGGER = logging.getLogger(__name__)


class LifecycleController:
    """Owns stop signaling and emergency shutdown execution."""

    def __init__(self, on_shutdown: Callable[[str], None]) -> None:
        self._on_shutdown = on_shutdown
        self._stop_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self._stop_reason: Optional[str] = None

        self._register_signal_handlers()
        atexit.register(self.execute_shutdown, "atexit")

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def stop_reason(self) -> Optional[str]:
        return self._stop_reason

    def request_stop(self, reason: str) -> None:
        if not self._stop_event.is_set():
            self._stop_reason = reason
            LOGGER.warning("Stop requested: %s", reason)
        self._stop_event.set()

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self.request_stop(f"signal_{signum}")

    def _register_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, self._handle_signal)
            except (ValueError, OSError):  # pragma: no cover
                # Can fail in worker threads or constrained runtimes.
                continue

    def execute_shutdown(self, reason: str) -> None:
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

        try:
            self._on_shutdown(reason)
        except Exception:  # pragma: no cover
            LOGGER.exception("Emergency shutdown handler failed.")

