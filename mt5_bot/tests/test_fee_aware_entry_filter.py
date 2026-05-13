import unittest
from dataclasses import dataclass

from strategies.entry_filter import (
    BROKER_STOP_LEVEL_VIOLATION,
    CONSECUTIVE_LOSS_COOLDOWN,
    DAILY_BLEED_GUARD_ACTIVE,
    DATA_GAP,
    FEE_ADJUSTED_RR_TOO_LOW,
    FeeAwareEntryFilter,
    LATE_ENTRY,
    LIVE_GATE_CLOSED,
    LLM_VETO_FORBIDDEN_ATTEMPT,
    MAX_OPEN_POSITIONS_REACHED,
    MIN_LOT_RISK_EXCEEDS_HARD_MAX,
    OOS_NOT_READY,
    PAPER_ONLY_MODE,
    SIGNAL_SCORE_TOO_LOW,
    SPREAD_TOO_WIDE,
)


def good_opportunity(**overrides):
    data = {
        "id": "opp-1",
        "symbol": "BTCUSD",
        "signal_score": 0.85,
        "fee_adjusted_rr": 2.2,
        "estimated_sl_net_loss": 10.0,
        "min_lot_estimated_sl_net_loss": 8.0,
        "spread_points": 15.0,
        "volume": 0.02,
        "min_lot": 0.01,
        "max_lot": 10.0,
        "stop_distance_points": 80.0,
        "direction": "BUY",
        "setup_key": "breakout",
    }
    data.update(overrides)
    return data


def entry_filter(**overrides):
    cfg = {
        "min_signal_score": 0.7,
        "min_reward_to_net_risk_ratio": 1.6,
        "hard_max_net_loss_usd": 25.0,
        "max_spread_points": 20.0,
        "max_signal_age_seconds": 300.0,
        "max_bars_late": 1,
        "max_open_positions": 2,
        "broker_stop_level_points": 50.0,
        "max_data_gap_seconds": 30.0,
    }
    cfg.update(overrides)
    return FeeAwareEntryFilter(cfg)


@dataclass
class ObjectOpportunity:
    symbol: str = "BTCUSD"
    signal_score: float = 0.9
    fee_adjusted_rr: float = 2.0
    estimated_sl_net_loss: float = 10.0
    spread_points: float = 10.0
    volume: float = 0.02
    min_lot: float = 0.01
    max_lot: float = 1.0
    stop_distance_points: float = 80.0


class FakeDailyBleedGuard:
    def __init__(self, reason):
        self.reason = reason

    def entry_block(self, **kwargs):
        return self.reason


class FeeAwareEntryFilterTests(unittest.TestCase):
    def assertBlocked(self, opportunity, reason, context=None):
        out = entry_filter().evaluate(opportunity, context)
        self.assertFalse(out.allow)
        self.assertIn(reason, out.reasons)
        return out

    def test_good_setup_passes(self):
        out = entry_filter().evaluate(good_opportunity(), {"live_gate_open": True})
        self.assertTrue(out.allow)
        self.assertEqual(out.reason, "ok")

    def test_object_opportunity_passes(self):
        out = entry_filter().evaluate(ObjectOpportunity())
        self.assertTrue(out.passed)

    def test_blocks_spread_too_wide(self):
        self.assertBlocked(good_opportunity(spread_points=21.0), SPREAD_TOO_WIDE)

    def test_blocks_fee_adjusted_rr_too_low(self):
        self.assertBlocked(good_opportunity(fee_adjusted_rr=1.2), FEE_ADJUSTED_RR_TOO_LOW)

    def test_blocks_signal_score_too_low(self):
        self.assertBlocked(good_opportunity(signal_score=0.69), SIGNAL_SCORE_TOO_LOW)

    def test_blocks_late_entry_by_age(self):
        self.assertBlocked(good_opportunity(signal_age_seconds=301.0), LATE_ENTRY)

    def test_blocks_late_entry_by_flag(self):
        self.assertBlocked(good_opportunity(late_entry=True), LATE_ENTRY)

    def test_blocks_min_lot_risk_exceeds_hard_max(self):
        self.assertBlocked(good_opportunity(min_lot_estimated_sl_net_loss=26.0), MIN_LOT_RISK_EXCEEDS_HARD_MAX)

    def test_blocks_requested_size_outside_feasible_volume(self):
        self.assertBlocked(good_opportunity(volume=0.005), MIN_LOT_RISK_EXCEEDS_HARD_MAX)

    def test_blocks_daily_bleed_guard_active(self):
        self.assertBlocked(good_opportunity(), DAILY_BLEED_GUARD_ACTIVE, {"daily_bleed_guard_active": True})

    def test_maps_daily_bleed_consecutive_losses_to_cooldown_reason(self):
        self.assertBlocked(
            good_opportunity(),
            CONSECUTIVE_LOSS_COOLDOWN,
            {"daily_bleed_guard": FakeDailyBleedGuard("DAILY_BLEED_CONSECUTIVE_LOSSES")},
        )

    def test_blocks_max_open_positions_reached(self):
        self.assertBlocked(good_opportunity(), MAX_OPEN_POSITIONS_REACHED, {"open_positions_count": 2})

    def test_blocks_broker_stop_level_violation(self):
        self.assertBlocked(good_opportunity(stop_distance_points=49.0), BROKER_STOP_LEVEL_VIOLATION)

    def test_blocks_data_gap_flag_and_age(self):
        out = self.assertBlocked(good_opportunity(data_gap=True, data_gap_seconds=31.0), DATA_GAP)
        self.assertEqual(out.reasons.count(DATA_GAP), 1)

    def test_blocks_live_gate_closed(self):
        self.assertBlocked(good_opportunity(), LIVE_GATE_CLOSED, {"live_gate_open": False})

    def test_blocks_paper_only_mode(self):
        self.assertBlocked(good_opportunity(paper_only_mode=True), PAPER_ONLY_MODE)

    def test_blocks_oos_not_ready(self):
        out = FeeAwareEntryFilter({"require_oos_ready": True}).evaluate(good_opportunity())
        self.assertFalse(out.allow)
        self.assertIn(OOS_NOT_READY, out.reasons)

    def test_blocks_llm_veto_forbidden_attempt(self):
        self.assertBlocked(good_opportunity(llm_veto_forbidden_attempt=True), LLM_VETO_FORBIDDEN_ATTEMPT)

    def test_collects_multiple_reasons_in_priority_order(self):
        out = entry_filter().evaluate(good_opportunity(signal_score=0.1, spread_points=99.0))
        self.assertEqual(out.reason, SIGNAL_SCORE_TOO_LOW)
        self.assertIn(SPREAD_TOO_WIDE, out.reasons)


if __name__ == "__main__":
    unittest.main()
