from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.bot_token = str(cfg.get("bot_token", "")).strip()
        self.chat_id = str(cfg.get("chat_id", "")).strip()
        self.notify_trade = bool(cfg.get("notify_trade", True))
        self.notify_error = bool(cfg.get("notify_error", True))
        self.notify_system = bool(cfg.get("notify_system", True))
        self.timeout_seconds = 6

        if self.enabled and (not self.bot_token or not self.chat_id):
            LOGGER.warning("Telegram enabled but bot_token/chat_id not configured. Disabling alerts.")
            self.enabled = False

    def _post(self, text: str) -> None:
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        body = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            method="POST",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw)
                if not parsed.get("ok", False):
                    LOGGER.warning("Telegram API error response: %s", parsed)
        except Exception:
            LOGGER.exception("Failed to send Telegram message.")

    def send_trade(self, message: str) -> None:
        if self.notify_trade:
            self._post(f"*TRADE*\\n{message}")

    def send_error(self, message: str) -> None:
        if self.notify_error:
            self._post(f"*ERROR*\\n{message}")

    def send_system(self, message: str) -> None:
        if self.notify_system:
            self._post(f"*SYSTEM*\\n{message}")

