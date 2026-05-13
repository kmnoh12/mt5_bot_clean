import json
import unittest

from core.no_trade_guard import (
    FAILURE_NO_TRADE_2D,
    WARNING_NO_TRADE_24H,
    ZERO_TRADE_NOT_SUCCESS,
    NoTradeBiasGuard,
)
from reports.no_trade_report import build_no_trade_report_json, build_no_trade_report_markdown


def opp(idx, score=0.5, rr=1.0, symbol="BTCUSD"):
    return {
        "id": f"opp-{idx}",
        "symbol": symbol,
        "signal_score": score,
        "fee_adjusted_rr": rr,
    }


class NoTradeBiasGuardTests(unittest.TestCase):
    def test_tracks_core_counts(self):
        guard = NoTradeBiasGuard()
        guard.record_raw_signal(now_ts=0)
        guard.record_scored_signal(now_ts=1)
        guard.record_eligible_signal(opp(1), now_ts=2)
        guard.record_executed_trade(opp(1), now_ts=3)
        snap = guard.snapshot(now_ts=4)
        self.assertEqual(snap["raw_signal_count"], 1)
        self.assertEqual(snap["scored_signal_count"], 1)
        self.assertEqual(snap["eligible_signal_count"], 1)
        self.assertEqual(snap["executed_trade_count"], 1)

    def test_no_trade_hours_since_start_without_trade(self):
        guard = NoTradeBiasGuard()
        guard.record_raw_signal(now_ts=100)
        self.assertAlmostEqual(guard.snapshot(now_ts=100 + 7200)["no_trade_hours"], 2.0)

    def test_no_trade_hours_resets_from_last_executed_trade(self):
        guard = NoTradeBiasGuard()
        guard.record_raw_signal(now_ts=0)
        guard.record_executed_trade(now_ts=3600)
        self.assertAlmostEqual(guard.snapshot(now_ts=7200)["no_trade_hours"], 1.0)

    def test_24h_no_trade_warning(self):
        guard = NoTradeBiasGuard()
        guard.record_raw_signal(now_ts=0)
        snap = guard.snapshot(now_ts=24 * 3600)
        self.assertIn(WARNING_NO_TRADE_24H, snap["warnings"])
        self.assertEqual(snap["status"], "failure")

    def test_2d_no_trade_failure(self):
        guard = NoTradeBiasGuard()
        guard.record_raw_signal(now_ts=0)
        snap = guard.snapshot(now_ts=48 * 3600)
        self.assertIn(FAILURE_NO_TRADE_2D, snap["failures"])
        self.assertEqual(snap["no_trade_days_count"], 2)

    def test_zero_trade_is_not_success(self):
        guard = NoTradeBiasGuard()
        guard.record_raw_signal(now_ts=0)
        snap = guard.snapshot(now_ts=1)
        self.assertFalse(snap["zero_trade_success"])
        self.assertIn(ZERO_TRADE_NOT_SUCCESS, snap["failures"])

    def test_block_rate_by_reason(self):
        guard = NoTradeBiasGuard()
        guard.record_rejection(opp(1), "spread_too_wide", now_ts=0)
        guard.record_rejection(opp(2), ["spread_too_wide", "signal_score_too_low"], now_ts=1)
        snap = guard.snapshot(now_ts=2)
        self.assertEqual(snap["block_count_by_reason"]["spread_too_wide"], 2)
        self.assertAlmostEqual(snap["block_rate_by_reason"]["spread_too_wide"], 2 / 3)

    def test_top_rejected_opportunities_sorted_by_score(self):
        guard = NoTradeBiasGuard({"top_rejected_limit": 2})
        guard.record_rejection(opp(1, score=0.2), "a", now_ts=0)
        guard.record_rejection(opp(2, score=0.9), "b", now_ts=0)
        guard.record_rejection(opp(3, score=0.7), "c", now_ts=0)
        snap = guard.snapshot(now_ts=1)
        self.assertEqual([item["opportunity_id"] for item in snap["top_rejected_opportunities"]], ["opp-2", "opp-3"])

    def test_best_missed_opportunity_uses_eligible_and_rejected(self):
        guard = NoTradeBiasGuard()
        guard.record_eligible_signal(opp(1, score=0.8, rr=1.2), now_ts=0)
        guard.record_rejection(opp(2, score=0.7, rr=3.0), "late_entry", now_ts=1)
        snap = guard.snapshot(now_ts=2)
        self.assertEqual(snap["best_missed_opportunity"]["opportunity_id"], "opp-1")

    def test_record_filter_decision_routes_allow_and_block(self):
        guard = NoTradeBiasGuard()
        guard.record_filter_decision(opp(1), {"allow": True}, now_ts=0)
        guard.record_filter_decision(opp(2), {"allow": False, "reasons": ["data_gap"]}, now_ts=1)
        snap = guard.snapshot(now_ts=2)
        self.assertEqual(snap["eligible_signal_count"], 1)
        self.assertEqual(snap["block_count_by_reason"]["data_gap"], 1)

    def test_json_report_builder_is_parseable(self):
        guard = NoTradeBiasGuard()
        guard.record_raw_signal(now_ts=0)
        payload = build_no_trade_report_json(guard.snapshot(now_ts=1))
        parsed = json.loads(payload)
        self.assertEqual(parsed["raw_signal_count"], 1)

    def test_markdown_report_builder_contains_required_sections(self):
        guard = NoTradeBiasGuard()
        guard.record_rejection(opp(1, score=0.9), "spread_too_wide", now_ts=0)
        report = build_no_trade_report_markdown(guard.snapshot(now_ts=24 * 3600))
        self.assertIn("# No-Trade Bias Report", report)
        self.assertIn("Block Rate By Reason", report)
        self.assertIn("Best Missed Opportunity", report)
        self.assertIn("spread_too_wide", report)


if __name__ == "__main__":
    unittest.main()
