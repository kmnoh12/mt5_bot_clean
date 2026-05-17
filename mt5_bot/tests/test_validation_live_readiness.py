import json
from pathlib import Path

from core.validation import check_live_readiness


def _write_report(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "oos_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_live_readiness_rejects_boolean_only_oos_pass(tmp_path: Path) -> None:
    path = _write_report(tmp_path, {"oos_pass": True})

    assert check_live_readiness(str(path)) == (False, "oos_total_trades_missing")


def test_live_readiness_rejects_tiny_oos_sample(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "shadow_pass": True,
            "oos_total_trades": 5,
            "shadow_sample_count": 25,
            "thresholds": {"min_oos_trades": 20, "min_shadow_samples": 20},
        },
    )

    assert check_live_readiness(str(path)) == (False, "oos_total_trades_lt_min")


def test_live_readiness_requires_walk_forward_stage(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "shadow_pass": True,
            "oos_total_trades": 25,
            "shadow_sample_count": 25,
        },
    )

    assert check_live_readiness(str(path)) == (False, "walk_forward_not_passed")


def test_live_readiness_rejects_clustered_oos_day_sample_when_threshold_is_set(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "shadow_pass": True,
            "oos_total_trades": 125,
            "oos_trading_day_count": 1,
            "shadow_sample_count": 25,
            "thresholds": {"min_oos_trades": 100, "min_oos_trading_days": 3, "min_shadow_samples": 20},
        },
    )

    assert check_live_readiness(str(path)) == (False, "oos_trading_days_lt_min")


def test_live_readiness_counts_oos_trade_dates_when_day_count_is_missing(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_window_count": 2,
            "oos_total_trades": 125,
            "oos_trade_dates": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "thresholds": {"min_oos_trades": 100, "min_oos_trading_days": 3},
            "shadow": {
                "promotion_gate": {"status": "pass"},
                "blocked_total": 32,
                "thresholds": {"min_live_review_samples": 20},
            },
        },
    )

    assert check_live_readiness(str(path)) == (True, "oos_walk_forward_shadow_pass")


def test_live_readiness_rejects_boolean_only_walk_forward_pass(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "shadow_pass": True,
            "oos_total_trades": 25,
            "shadow_sample_count": 25,
        },
    )

    assert check_live_readiness(str(path)) == (False, "walk_forward_window_count_missing")


def test_live_readiness_rejects_single_walk_forward_window(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_window_count": 1,
            "shadow_pass": True,
            "oos_total_trades": 25,
            "shadow_sample_count": 25,
        },
    )

    assert check_live_readiness(str(path)) == (False, "walk_forward_windows_lt_min")


def test_live_readiness_rejects_failed_walk_forward_window_detail(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_windows": [
                {"id": 1, "status": "pass", "oos_total_trades": 32},
                {"id": 2, "status": "failed", "oos_total_trades": 44},
            ],
            "shadow_pass": True,
            "oos_total_trades": 125,
            "shadow_sample_count": 25,
            "thresholds": {"min_oos_trades": 100, "min_shadow_samples": 20},
        },
    )

    assert check_live_readiness(str(path)) == (False, "walk_forward_window_failed")


def test_live_readiness_rejects_tiny_walk_forward_oos_window_when_floor_is_set(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward": {
                "folds": [
                    {"id": 1, "passed": True, "oos_total_trades": 22},
                    {"id": 2, "passed": True, "oos_total_trades": 3},
                ],
            },
            "shadow_pass": True,
            "oos_total_trades": 125,
            "shadow_sample_count": 25,
            "thresholds": {
                "min_oos_trades": 100,
                "min_shadow_samples": 20,
                "min_walk_forward_oos_trades_per_window": 10,
            },
        },
    )

    assert check_live_readiness(str(path)) == (False, "walk_forward_window_oos_trades_lt_min")


def test_live_readiness_accepts_walk_forward_window_oos_floor_when_each_window_has_samples(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_windows": [
                {"id": 1, "status": "pass", "oos_total_trades": 22},
                {"id": 2, "status": "pass", "oos_total_trades": 18},
            ],
            "shadow_pass": True,
            "oos_total_trades": 125,
            "shadow_sample_count": 25,
            "thresholds": {
                "min_oos_trades": 100,
                "min_shadow_samples": 20,
                "min_walk_forward_oos_trades_per_window": 10,
            },
        },
    )

    assert check_live_readiness(str(path)) == (True, "oos_walk_forward_shadow_pass")


def test_live_readiness_requires_shadow_stage(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_window_count": 2,
            "oos_total_trades": 25,
            "shadow_sample_count": 25,
        },
    )

    assert check_live_readiness(str(path)) == (False, "shadow_not_passed")


def test_live_readiness_rejects_symbol_without_oos_coverage(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "shadow_pass": True,
            "oos_total_trades": 125,
            "oos_trading_day_count": 3,
            "shadow_sample_count": 25,
            "thresholds": {
                "min_oos_trades": 100,
                "min_oos_trading_days": 3,
                "min_oos_trades_per_symbol": 20,
                "min_oos_trading_days_per_symbol": 1,
                "min_shadow_samples": 20,
            },
            "symbol_metrics": {
                "BTCUSD": {"oos": {"total_trades": 125, "trading_day_count": 3}},
                "ETHUSD": {"oos": {"total_trades": 0, "trading_day_count": 0}},
            },
        },
    )

    assert check_live_readiness(str(path)) == (False, "symbol_oos_total_trades_lt_min")


def test_live_readiness_accepts_symbol_oos_coverage_from_trade_dates(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_window_count": 2,
            "shadow_pass": True,
            "oos_total_trades": 125,
            "oos_trading_day_count": 3,
            "shadow_sample_count": 25,
            "thresholds": {
                "min_oos_trades": 100,
                "min_oos_trades_per_symbol": 20,
                "min_oos_trading_days_per_symbol": 2,
                "min_shadow_samples": 20,
            },
            "symbol_metrics": {
                "BTCUSD": {"oos": {"total_trades": 70, "trade_dates": ["2026-01-01", "2026-01-02"]}},
                "ETHUSD": {"oos": {"total_trades": 55, "trade_dates": ["2026-01-02", "2026-01-03"]}},
            },
        },
    )

    assert check_live_readiness(str(path)) == (True, "oos_walk_forward_shadow_pass")


def test_live_readiness_rejects_blocked_shadow_promotion_gate(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_window_count": 2,
            "shadow_pass": True,
            "oos_total_trades": 125,
            "shadow": {
                "promotion_gate": {
                    "status": "blocked",
                    "block_reasons": ["non_positive_net_r_delta"],
                },
                "blocked_total": 40,
            },
        },
    )

    assert check_live_readiness(str(path)) == (False, "shadow_promotion_gate_blocked")


def test_live_readiness_rejects_one_sided_oos_direction_when_threshold_is_set(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "shadow_pass": True,
            "oos_total_trades": 125,
            "oos_direction_trade_counts": {"long": 125, "short": 0},
            "shadow_sample_count": 25,
            "thresholds": {
                "min_oos_trades": 100,
                "min_oos_trades_per_direction": 5,
                "min_shadow_samples": 20,
            },
        },
    )

    assert check_live_readiness(str(path)) == (False, "oos_direction_trades_lt_min")


def test_live_readiness_accepts_oos_direction_coverage_when_threshold_is_set(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward": {"folds": [{"id": 1}, {"id": 2}]},
            "shadow_pass": True,
            "oos_total_trades": 125,
            "oos_direction_trade_counts": {"buy": 70, "sell": 55},
            "shadow_sample_count": 25,
            "thresholds": {
                "min_oos_trades": 100,
                "min_oos_trades_per_direction": 5,
                "min_shadow_samples": 20,
            },
        },
    )

    assert check_live_readiness(str(path)) == (True, "oos_walk_forward_shadow_pass")


def test_live_readiness_rejects_inconsistent_shadow_gate_with_block_reasons(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_window_count": 2,
            "oos_total_trades": 125,
            "shadow": {
                "promotion_gate": {
                    "status": "pass",
                    "block_reasons": ["false_block_rate_above_limit"],
                },
                "blocked_total": 40,
            },
        },
    )

    assert check_live_readiness(str(path)) == (False, "shadow_promotion_gate_blocked")


def test_live_readiness_accepts_evidenced_walk_forward_oos_shadow_report(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path,
        {
            "oos_pass": True,
            "walk_forward_pass": True,
            "walk_forward_windows": [{"start": "2026-01-01"}, {"start": "2026-01-08"}],
            "oos_total_trades": 125,
            "thresholds": {"min_oos_trades": 100},
            "shadow": {
                "promotion_gate": {"status": "pass"},
                "blocked_total": 32,
                "thresholds": {"min_live_review_samples": 20},
            },
        },
    )

    assert check_live_readiness(str(path)) == (True, "oos_walk_forward_shadow_pass")
