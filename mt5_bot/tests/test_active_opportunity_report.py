import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from reports.active_opportunities import (
    build_active_opportunity_markdown,
    build_active_opportunity_report,
    write_active_opportunity_reports,
)


@dataclass(frozen=True)
class SampleOpportunity:
    symbol: str
    direction: str
    score: float
    eligible: bool
    entry_price: float
    stop_price: float
    take_profit_price: float
    estimated_lot: float
    tick_size: float
    tick_value: float
    current_spread: float
    commission_per_lot: float = 0.0


def _sample_source():
    return {
        "generated_at_utc": "2026-05-13T00:00:00Z",
        "current_symbols": ["BTCUSD", "ETHUSD", "XAUUSD"],
        "current_timeframe": "M5",
        "candidates": [
            {
                "symbol": "BTCUSD",
                "direction": "long",
                "score": 0.92,
                "eligible": True,
                "entry_price": 100.0,
                "stop_price": 95.0,
                "take_profit_price": 115.0,
                "estimated_lot": 2.0,
                "tick_size": 1.0,
                "tick_value": 1.0,
                "current_spread": 0.5,
                "commission_per_lot": 1.0,
            },
            {
                "symbol": "ETHUSD",
                "direction": "long",
                "score": 0.55,
                "eligible": True,
                "estimated_lot": 0.2,
                "estimated_sl_net_loss": 11.0,
                "estimated_tp_net_profit": 33.0,
                "current_cost": 1.5,
            },
            {
                "symbol": "XAUUSD",
                "direction": "short",
                "score": 0.81,
                "eligible": True,
                "entry_price": 2000.0,
                "stop_price": 2010.0,
                "take_profit_price": 1970.0,
                "estimated_lot": 0.1,
                "tick_size": 1.0,
                "tick_value": 10.0,
                "current_cost": 3.0,
            },
            {
                "symbol": "BTCUSD",
                "direction": "short",
                "score": 0.7,
                "eligible": False,
                "block_reasons": ["SPREAD_TOO_HIGH"],
                "estimated_lot": 0.1,
            },
        ],
        "rejected_candidates": [
            {
                "symbol": "ETHUSD",
                "direction": "short",
                "score": 0.2,
                "reject_reason": "RR_TOO_LOW",
            }
        ],
    }


class ActiveOpportunityReportTests(unittest.TestCase):
    def test_builds_top_long_and_short_candidates(self) -> None:
        report = build_active_opportunity_report(_sample_source())

        self.assertEqual(report["top_long_candidates"][0]["symbol"], "BTCUSD")
        self.assertEqual(report["top_short_candidates"][0]["symbol"], "XAUUSD")
        self.assertEqual(report["best_eligible_candidate"]["symbol"], "BTCUSD")

    def test_collects_rejected_candidates_and_block_reasons(self) -> None:
        report = build_active_opportunity_report(_sample_source())

        self.assertEqual(report["rejected_count"], 2)
        self.assertEqual(report["block_reasons"]["SPREAD_TOO_HIGH"], 1)
        self.assertEqual(report["block_reasons"]["RR_TOO_LOW"], 1)

    def test_computes_fee_aware_net_risk_from_price_inputs(self) -> None:
        report = build_active_opportunity_report(_sample_source())
        btc = report["top_long_candidates"][0]

        self.assertAlmostEqual(btc["current_cost"], 5.0)
        self.assertAlmostEqual(btc["estimated_sl_net_loss"], 15.0)
        self.assertAlmostEqual(btc["estimated_tp_net_profit"], 25.0)

    def test_profit_lock_plan_marks_reachable_stages(self) -> None:
        report = build_active_opportunity_report(_sample_source())
        btc = report["top_long_candidates"][0]
        plan = btc["profit_lock_plan"]

        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["reachable_stages_at_estimated_tp"][0]["trigger_net_profit"], 20.0)
        self.assertEqual(plan["reachable_stages_at_estimated_tp"][-1]["trigger_net_profit"], 2.0)

    def test_accepts_dataclass_candidates(self) -> None:
        source = {
            "current_timeframe": "M1",
            "candidates": [
                SampleOpportunity("EURUSD", "buy", 0.9, True, 1.1000, 1.0950, 1.1150, 1.0, 0.0001, 10.0, 0.0002)
            ],
        }
        report = build_active_opportunity_report(source, generated_at_utc="2026-05-13T01:00:00Z")

        candidate = report["top_long_candidates"][0]
        self.assertEqual(candidate["symbol"], "EURUSD")
        self.assertEqual(candidate["direction"], "long")
        self.assertAlmostEqual(candidate["estimated_sl_net_loss"], 520.0)

    def test_markdown_contains_required_sections(self) -> None:
        markdown = build_active_opportunity_markdown(_sample_source())

        self.assertIn("# Active Opportunity Report", markdown)
        self.assertIn("## Top Long Candidates", markdown)
        self.assertIn("## Top Short Candidates", markdown)
        self.assertIn("## Rejected Candidates", markdown)
        self.assertIn("SPREAD_TOO_HIGH", markdown)

    def test_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "active_opportunities.json"
            md_path = Path(tmp) / "active_opportunities.md"
            report = write_active_opportunity_reports(_sample_source(), json_path=json_path, markdown_path=md_path)

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["eligible_count"], report["eligible_count"])
            self.assertIn("Active Opportunity Report", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
