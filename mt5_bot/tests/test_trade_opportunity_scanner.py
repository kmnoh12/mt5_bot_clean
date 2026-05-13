from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from strategies.opportunity_scanner import TradeOpportunity, TradeOpportunityScanner


def _base_frame(periods: int = 20) -> pd.DataFrame:
    rows = []
    for idx, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="min")):
        close = 100.0 + ((idx % 5) * 0.05)
        rows.append(
            {
                "time": timestamp,
                "open": close - 0.02,
                "high": 101.0,
                "low": 99.0,
                "close": close,
            }
        )
    return pd.DataFrame(rows)


def _with_signal(row: dict) -> pd.DataFrame:
    return pd.concat([_base_frame(), pd.DataFrame([row])], ignore_index=True)


def _scanner(**kwargs) -> TradeOpportunityScanner:
    params = {
        "lookback_bars": 20,
        "atr_period": 5,
        "min_signal_score": 70.0,
    }
    params.update(kwargs)
    return TradeOpportunityScanner(**params)


def _long_frame(**overrides) -> pd.DataFrame:
    row = {
        "time": pd.Timestamp("2026-01-01T00:20:00Z"),
        "open": 98.9,
        "high": 101.2,
        "low": 98.55,
        "close": 99.35,
    }
    row.update(overrides)
    return _with_signal(row)


def _short_frame(**overrides) -> pd.DataFrame:
    row = {
        "time": pd.Timestamp("2026-01-01T00:20:00Z"),
        "open": 101.1,
        "high": 101.45,
        "low": 98.8,
        "close": 100.65,
    }
    row.update(overrides)
    return _with_signal(row)


def test_long_liquidity_sweep_reclaim_generates_trade_opportunity() -> None:
    opportunities = _scanner().scan(symbol="XAUUSD", timeframe="M5", bars=_long_frame())

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert isinstance(opportunity, TradeOpportunity)
    assert opportunity.symbol == "XAUUSD"
    assert opportunity.timeframe == "M5"
    assert opportunity.direction == "long"
    assert opportunity.entry_price == 99.35
    assert opportunity.invalidation_price < opportunity.entry_price
    assert opportunity.target_reference_price > opportunity.entry_price
    assert opportunity.reason == "LIQUIDITY_SWEEP_RECLAIM_LONG"
    assert opportunity.signal_score >= 70.0


def test_short_liquidity_sweep_reclaim_generates_trade_opportunity() -> None:
    opportunities = _scanner().scan(symbol="XAUUSD", timeframe="M5", bars=_short_frame())

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity.direction == "short"
    assert opportunity.invalidation_price > opportunity.entry_price
    assert opportunity.target_reference_price < opportunity.entry_price
    assert opportunity.reason == "LIQUIDITY_SWEEP_RECLAIM_SHORT"


def test_outside_bar_can_generate_both_long_and_short_candidates() -> None:
    frame = _with_signal(
        {
            "time": pd.Timestamp("2026-01-01T00:20:00Z"),
            "open": 100.0,
            "high": 102.5,
            "low": 97.5,
            "close": 100.0,
        }
    )

    opportunities = _scanner(min_signal_score=0.0).scan(symbol="BTCUSD", timeframe="M1", bars=frame)

    assert {item.direction for item in opportunities} == {"long", "short"}
    assert opportunities[0].signal_score >= opportunities[1].signal_score


def test_components_include_required_scores_and_risk_fields() -> None:
    opportunity = _scanner().scan(symbol="XAUUSD", timeframe="M5", bars=_long_frame())[0]

    expected_score_keys = {
        "liquidity_sweep_quality": 25,
        "reclaim_or_rejection_strength": 20,
        "invalidation_distance_efficiency": 20,
        "fee_adjusted_rr": 20,
        "spread_quality": 10,
        "volatility_sufficiency": 5,
    }
    for key, max_value in expected_score_keys.items():
        assert key in opportunity.components
        assert 0.0 <= float(opportunity.components[key]) <= float(max_value)

    assert opportunity.components["invalidation_distance"] > 0.0
    assert opportunity.components["gross_rr"] > 0.0
    assert opportunity.components["fee_adjusted_rr_value"] > 0.0


def test_default_min_signal_score_threshold_filters_low_quality_candidate() -> None:
    opportunities = _scanner(min_signal_score=95.0).scan(symbol="XAUUSD", timeframe="M5", bars=_long_frame())

    assert opportunities == []


def test_scan_min_signal_score_override_can_include_lower_score_candidate() -> None:
    scanner = _scanner(min_signal_score=95.0)

    opportunities = scanner.scan(
        symbol="XAUUSD",
        timeframe="M5",
        bars=_long_frame(),
        min_signal_score=70.0,
    )

    assert len(opportunities) == 1
    assert 70.0 <= opportunities[0].signal_score < 95.0


def test_spread_and_round_turn_cost_reduce_quality_and_fee_adjusted_rr() -> None:
    clean = _scanner().scan(symbol="XAUUSD", timeframe="M5", bars=_long_frame())[0]
    costly = _scanner(min_signal_score=0.0).scan(
        symbol="XAUUSD",
        timeframe="M5",
        bars=_long_frame(),
        spread=0.20,
        round_turn_cost=0.30,
    )[0]

    assert costly.components["spread_quality"] < clean.components["spread_quality"]
    assert costly.components["fee_adjusted_rr_value"] < clean.components["fee_adjusted_rr_value"]
    assert costly.components["fee_adjusted_rr"] < clean.components["fee_adjusted_rr"]


def test_late_entry_detected_when_reclaim_is_extended_from_swept_level() -> None:
    frame = _long_frame(open=98.8, high=102.5, low=97.8, close=101.4)

    opportunity = _scanner().scan(symbol="XAUUSD", timeframe="M5", bars=frame)[0]

    assert opportunity.late_entry is True
    assert "reclaim_extended_from_swept_level" in opportunity.components["late_entry_reasons"]


def test_late_entry_detected_when_gross_rr_is_below_floor() -> None:
    frame = _long_frame(close=99.8)

    opportunity = _scanner(min_signal_score=0.0).scan(symbol="XAUUSD", timeframe="M5", bars=frame)[0]

    assert opportunity.late_entry is True
    assert opportunity.components["gross_rr"] < 1.0
    assert "gross_rr_below_late_entry_floor" in opportunity.components["late_entry_reasons"]


def test_not_late_when_reclaim_distance_and_rr_are_acceptable() -> None:
    opportunity = _scanner().scan(symbol="XAUUSD", timeframe="M5", bars=_long_frame())[0]

    assert opportunity.late_entry is False
    assert opportunity.components["late_entry_reasons"] == []


def test_no_candidate_when_sweep_does_not_reclaim_level() -> None:
    frame = _long_frame(high=100.8, close=98.8)

    opportunities = _scanner(min_signal_score=0.0).scan(symbol="XAUUSD", timeframe="M5", bars=frame)

    assert opportunities == []


def test_accepts_list_of_dict_bars_without_pandas_input_requirement() -> None:
    rows = _long_frame().to_dict("records")

    opportunities = _scanner().scan(symbol="XAUUSD", timeframe="M5", bars=rows)

    assert len(opportunities) == 1
    assert opportunities[0].direction == "long"


def test_detected_at_utc_can_be_supplied_by_caller() -> None:
    detected_at = datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)

    opportunity = _scanner().scan(
        symbol="XAUUSD",
        timeframe="M5",
        bars=_long_frame(),
        detected_at_utc=detected_at,
    )[0]

    assert opportunity.detected_at_utc == detected_at
