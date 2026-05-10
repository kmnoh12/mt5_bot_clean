from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class LlmRawDecision:
    action: str
    confidence: float
    reason: str
    ttl_seconds: int = 120


PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {
        "model": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "gemini": {
        "model": "gemini-3-flash",
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
}


class ChatCompletionClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_ms: int = 800,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_ms = max(100, int(timeout_ms))
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        text = str(content or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Best-effort extraction from markdown fenced block.
            left = text.find("{")
            right = text.rfind("}")
            if left >= 0 and right > left:
                try:
                    return json.loads(text[left : right + 1])
                except json.JSONDecodeError:
                    return None
            return None

    def infer(self, prompt_payload: Dict[str, Any]) -> Optional[LlmRawDecision]:
        if not self.api_key or not self.model:
            return None

        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a trading signal validator. Return strict JSON with keys: "
                        "action (BUY/SELL/HOLD), confidence (0..1), ttl_seconds (int), reason (short)."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
            ],
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        timeout_seconds = self.timeout_ms / 1000.0
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            LOGGER.warning("LLM API HTTPError: %s", detail)
            return None
        except Exception as exc:
            LOGGER.warning("LLM API request failed: %s", exc)
            return None

        try:
            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"]
        except Exception:
            return None

        parsed = self._extract_json(content)
        if not isinstance(parsed, dict):
            return None

        action = str(parsed.get("action", "HOLD") or "HOLD").upper()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        reason = str(parsed.get("reason", "") or "")
        ttl_seconds = int(parsed.get("ttl_seconds", 120) or 120)
        return LlmRawDecision(action=action, confidence=confidence, reason=reason, ttl_seconds=ttl_seconds)


def build_chat_client(config: Dict[str, Any]) -> Optional[ChatCompletionClient]:
    cfg = dict(config or {})
    provider = str(cfg.get("provider", "openai") or "openai").strip().lower()
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])

    api_key = str(cfg.get("api_key", "") or "").strip()
    env_name = str(cfg.get("api_key_env", defaults["api_key_env"]) or defaults["api_key_env"]).strip()
    if not api_key:
        api_key = str(os.getenv(env_name, "") or "").strip()
    if not api_key:
        return None

    model = str(cfg.get("model", defaults["model"]) or defaults["model"]).strip()
    timeout_ms = int(cfg.get("timeout_ms", 800) or 800)
    base_url = str(cfg.get("base_url", defaults["base_url"]) or defaults["base_url"])
    return ChatCompletionClient(api_key=api_key, model=model, timeout_ms=timeout_ms, base_url=base_url)


def build_openai_client(config: Dict[str, Any]) -> Optional[ChatCompletionClient]:
    cfg = dict(config or {})
    cfg["provider"] = "openai"
    return build_chat_client(cfg)
