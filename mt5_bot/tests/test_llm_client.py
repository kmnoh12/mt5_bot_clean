import unittest
from unittest.mock import patch

from llm.client import build_chat_client, build_openai_client


class LlmClientBuilderTests(unittest.TestCase):
    def test_build_chat_client_openai_defaults(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "openai-test-key"}, clear=True):
            client = build_chat_client({"provider": "openai"})

        self.assertIsNotNone(client)
        self.assertEqual(client.model, "gpt-5-mini")
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.timeout_ms, 800)

    def test_build_chat_client_gemini_defaults(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-test-key"}, clear=True):
            client = build_chat_client({"provider": "gemini"})

        self.assertIsNotNone(client)
        self.assertEqual(client.model, "gemini-3-flash")
        self.assertEqual(client.base_url, "https://generativelanguage.googleapis.com/v1beta/openai")
        self.assertEqual(client.timeout_ms, 800)

    def test_build_chat_client_missing_api_key_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = build_chat_client({"provider": "gemini"})
        self.assertIsNone(client)

    def test_build_openai_client_forces_openai_defaults(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "openai-test-key",
                "GEMINI_API_KEY": "gemini-test-key",
            },
            clear=True,
        ):
            client = build_openai_client({"provider": "gemini"})

        self.assertIsNotNone(client)
        self.assertEqual(client.model, "gpt-5-mini")
        self.assertEqual(client.base_url, "https://api.openai.com/v1")


if __name__ == "__main__":
    unittest.main()
