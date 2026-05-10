import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from dashboard.app import create_app


class DashboardSettingsTests(unittest.TestCase):
    def _write_config(self, tmpdir: str) -> tuple[Path, Path]:
        root = Path(tmpdir)
        settings_path = root / "dashboard_settings.json"
        control_path = root / "runtime_control.json"
        state_path = root / "state.json"
        events_path = root / "events.jsonl"
        cfg_path = root / "config.yaml"

        cfg_path.write_text(
            (
                "dashboard:\n"
                f"  control_path: \"{control_path.as_posix()}\"\n"
                f"  settings_path: \"{settings_path.as_posix()}\"\n"
                "storage:\n"
                f"  state_path: \"{state_path.as_posix()}\"\n"
                f"  events_path: \"{events_path.as_posix()}\"\n"
                "llm_assist:\n"
                "  enabled: false\n"
                "  provider: \"gemini\"\n"
                "  model: \"gemini-3-flash\"\n"
                "  api_key_env: \"GEMINI_API_KEY\"\n"
                "  base_url: \"https://generativelanguage.googleapis.com/v1beta/openai\"\n"
            ),
            encoding="utf-8",
        )
        return cfg_path, settings_path

    def test_status_exposes_llm_runtime_provider_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path, _ = self._write_config(tmpdir)
            client = TestClient(create_app(config_path=str(cfg_path)))

            payload = client.get("/api/status").json()
            llm = payload["runtime_settings"]["llm_assist"]

            self.assertIn("provider", llm)
            self.assertIn("api_key_env", llm)
            self.assertIn("base_url", llm)
            self.assertEqual(llm["provider"], "gemini")
            self.assertEqual(llm["api_key_env"], "GEMINI_API_KEY")
            self.assertEqual(llm["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai")

    def test_settings_payload_accepts_provider_fields_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path, settings_path = self._write_config(tmpdir)
            client = TestClient(create_app(config_path=str(cfg_path)))

            response = client.post(
                "/api/settings",
                json={
                    "llm_enabled": True,
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "do-not-persist",
                    "persist_api_key": False,
                },
            )
            payload = response.json()

            llm = payload["llm_assist"]
            self.assertEqual(llm["provider"], "openai")
            self.assertEqual(llm["api_key_env"], "OPENAI_API_KEY")
            self.assertEqual(llm["base_url"], "https://api.openai.com/v1")
            self.assertEqual(llm["model"], "gpt-5-mini")

            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["llm_assist"]["provider"], "openai")
            self.assertEqual(saved["llm_assist"]["api_key_env"], "OPENAI_API_KEY")
            self.assertEqual(saved["llm_assist"]["base_url"], "https://api.openai.com/v1")
            self.assertEqual(saved["llm_assist"]["api_key"], "")

    def test_dashboard_html_contains_provider_selector_and_gemini_quick_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path, _ = self._write_config(tmpdir)
            client = TestClient(create_app(config_path=str(cfg_path)))
            html = client.get("/").text

            self.assertIn('id="llm_provider"', html)
            self.assertIn("Gemini 3 Pro", html)
            self.assertIn("Gemini 3 Flash", html)


if __name__ == "__main__":
    unittest.main()
