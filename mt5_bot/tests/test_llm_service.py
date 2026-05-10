import unittest
from unittest.mock import patch

import pandas as pd

from core.models import DecisionAction, StrategyDecision
from llm.client import LlmRawDecision
from llm.service import LlmAssistService


class _FakeClient:
    def __init__(self, response: LlmRawDecision | None) -> None:
        self._response = response

    def infer(self, _prompt_payload):  # type: ignore[no-untyped-def]
        return self._response


def _sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:01:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "tick_volume": [10, 12],
        }
    )


def _sample_decision() -> StrategyDecision:
    return StrategyDecision(
        action=DecisionAction.BUY,
        reason="TEST_ENTRY",
        strategy="trend_regime_sm",
        confidence=0.7,
        volume=0.1,
        metadata={},
    )


class LlmAssistServiceTests(unittest.TestCase):
    def test_provider_unsupported(self) -> None:
        service = LlmAssistService({"enabled": True, "provider": "unknown"})
        decision, event = service.apply("BTCUSD", _sample_decision(), _sample_bars(), None)
        self.assertEqual(decision.action, DecisionAction.BUY)
        self.assertIsNotNone(event)
        self.assertEqual(event.get("status"), "provider_unsupported")

    def test_api_key_missing_event_kept_for_gemini(self) -> None:
        service = LlmAssistService({"enabled": True, "provider": "gemini"})
        with patch("llm.service.build_chat_client", return_value=None):
            decision, event = service.apply("BTCUSD", _sample_decision(), _sample_bars(), None)
        self.assertEqual(decision.action, DecisionAction.BUY)
        self.assertIsNotNone(event)
        self.assertEqual(event.get("status"), "api_key_missing")
        self.assertEqual(event.get("provider"), "gemini")

    def test_no_response_event_kept_for_openai(self) -> None:
        service = LlmAssistService({"enabled": True, "provider": "openai"})
        with patch("llm.service.build_chat_client", return_value=_FakeClient(None)):
            decision, event = service.apply("BTCUSD", _sample_decision(), _sample_bars(), None)
        self.assertEqual(decision.action, DecisionAction.BUY)
        self.assertIsNotNone(event)
        self.assertEqual(event.get("status"), "no_response")
        self.assertEqual(event.get("provider"), "openai")
        self.assertIn("latency_ms", event)


if __name__ == "__main__":
    unittest.main()
