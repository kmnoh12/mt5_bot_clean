import json
import tempfile
import unittest
from pathlib import Path

from core.config import load_config
from core.runtime import TradingRuntime


class ValidationGateTests(unittest.TestCase):
    def test_live_mode_blocked_without_oos_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = load_config(Path("config.yaml"))
            cfg["general"]["mode"] = "live"
            cfg["validation"]["require_oos_pass"] = True
            cfg["validation"]["report_path"] = str(Path(tmpdir) / "missing.json")

            with self.assertRaises(ValueError):
                TradingRuntime(config=cfg)

    def test_live_mode_allows_with_passing_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "oos_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "oos_pass": True,
                        "walk_forward_pass": True,
                        "walk_forward_window_count": 2,
                        "oos_total_trades": 125,
                        "thresholds": {"min_oos_trades": 100},
                        "shadow": {
                            "promotion_gate": {"status": "pass"},
                            "blocked_total": 25,
                            "thresholds": {"min_live_review_samples": 20},
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_config(Path("config.yaml"))
            cfg["general"]["mode"] = "live"
            cfg["validation"]["require_oos_pass"] = True
            cfg["validation"]["report_path"] = str(report_path)

            runtime = TradingRuntime(config=cfg)
            self.assertIsNotNone(runtime)


if __name__ == "__main__":
    unittest.main()
