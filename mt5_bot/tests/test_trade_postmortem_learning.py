from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.analyze_postmortem_learning import (
    analyze_learning,
    aggregate_patterns,
    execution_shortfall_summary,
    event_regime_context,
    generate_candidates,
    load_samples,
    lsr_confirmation_expectancy_summary,
    regime_expectancy_summary,
    shadow_evaluate,
)


def _sample(
    trade_key: str,
    *,
    label: str,
    r: float,
    displacement_ratio: float = 2.4,
    adx_entry: float = 50.0,
    score: float = 0.2,
    threshold: float = 0.2,
    chased: bool = True,
    exhaustion: bool = True,
    m5_align: float = 1.0,
    fee_adjusted_rr: float = 3.0,
    clean_reclaim: bool = False,
    confirmation_path: str | None = None,
    retest_confirmed: bool = False,
    lsr_unconfirmed_reclaim: bool = False,
    shallow_reclaim_confirmation: bool = False,
    weak_reclaim_after_deep_sweep: bool = False,
    lsr_unconfirmed_reclaim_chase: bool = False,
    late_window_reclaim: bool = False,
    reclaim_distance_atr: float | None = None,
    sweep_depth_atr: float | None = None,
    reclaim_to_sweep_depth_ratio: float | None = None,
    time_from_sweep_to_entry_sec: float | None = None,
    time_from_sweep_to_reclaim_sec: float | None = None,
    reclaim_window_sec: float | None = None,
    reclaim_window_elapsed_ratio: float | None = None,
    atr_regime_ratio: float = 1.0,
    spread_points: float = 10.0,
    price_r: float | None = None,
    entry_shortfall_r: float | None = None,
    execution_drag_r: float | None = None,
    estimated_cost_r: float | None = None,
    explicit_cost_r: float | None = None,
    estimated_cost_usd: float | None = None,
    explicit_cost_usd: float | None = None,
    expected_loss_usd: float | None = None,
) -> dict:
    if execution_drag_r is None and price_r is not None:
        execution_drag_r = price_r - r
    features = {
        "pnl_r_multiple": r,
        "displacement_ratio": displacement_ratio,
        "adx_entry": adx_entry,
        "entry_quality_score": score,
        "entry_quality_threshold": threshold,
        "entry_chased_extension": chased,
        "entered_into_exhaustion": exhaustion,
        "entered_against_short_term_momentum": False,
        "m5_align": m5_align,
        "fee_adjusted_rr": fee_adjusted_rr,
        "atr_regime_ratio": atr_regime_ratio,
        "spread_points": spread_points,
        "clean_reclaim": clean_reclaim,
        "exit_inferred_type": "STOP_LOSS" if label == "loss" else "TAKE_PROFIT_OR_TP_GUARD",
    }
    optional_lsr = {
        "confirmation_path": confirmation_path,
        "retest_confirmed": retest_confirmed,
        "reclaim_distance_atr": reclaim_distance_atr,
        "sweep_depth_atr": sweep_depth_atr,
        "reclaim_to_sweep_depth_ratio": reclaim_to_sweep_depth_ratio,
        "time_from_sweep_to_entry_sec": time_from_sweep_to_entry_sec,
        "time_from_sweep_to_reclaim_sec": time_from_sweep_to_reclaim_sec,
        "reclaim_window_sec": reclaim_window_sec,
        "reclaim_window_elapsed_ratio": reclaim_window_elapsed_ratio,
    }
    for key, value in optional_lsr.items():
        if value is not None:
            features[key] = value
    optional_tca = {
        "price_r_multiple": price_r,
        "entry_implementation_shortfall_r": entry_shortfall_r,
        "net_execution_drag_r": execution_drag_r,
        "estimated_cost_to_expected_loss_r": estimated_cost_r,
        "realized_explicit_cost_r": explicit_cost_r,
        "estimated_cost_usd": estimated_cost_usd,
        "realized_explicit_cost_usd": explicit_cost_usd,
        "expected_net_loss_usd": expected_loss_usd,
    }
    for key, value in optional_tca.items():
        if value is not None:
            features[key] = value
    quality_flags = {
        "entry_chased_extension": chased,
        "entered_into_exhaustion": exhaustion,
    }
    if lsr_unconfirmed_reclaim:
        features["lsr_unconfirmed_reclaim"] = True
        quality_flags["lsr_unconfirmed_reclaim"] = True
    if shallow_reclaim_confirmation:
        features["shallow_reclaim_confirmation"] = True
        features["shallow_reclaim_threshold_atr"] = 0.25
        quality_flags["shallow_reclaim_confirmation"] = True
    if weak_reclaim_after_deep_sweep:
        features["weak_reclaim_after_deep_sweep"] = True
        quality_flags["weak_reclaim_after_deep_sweep"] = True
    if lsr_unconfirmed_reclaim_chase:
        features["lsr_unconfirmed_reclaim_chase"] = True
        quality_flags["lsr_unconfirmed_reclaim_chase"] = True
    if late_window_reclaim:
        features["late_window_reclaim"] = True
        quality_flags["late_window_reclaim"] = True
    if retest_confirmed:
        quality_flags["retest_confirmed"] = True

    return {
        "trade_key": trade_key,
        "symbol": "BTCUSD",
        "side": "SELL",
        "strategy": "liquidity_sweep_reversal",
        "label": label,
        "pnl": r,
        "quality_grade": "F" if label == "loss" else "B",
        "bar_source": "REAL_BARS_FILE",
        "features": features,
        "quality_flags": quality_flags,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class TradePostmortemLearningTests(unittest.TestCase):
    def test_synthetic_samples_aggregate_wins_losses_and_r(self) -> None:
        rows = [
            _sample("loss-1", label="loss", r=-1.0),
            _sample("loss-2", label="loss", r=-0.5),
            _sample("loss-3", label="loss", r=-1.2),
            _sample("win-1", label="win", r=2.0),
            _sample("win-2", label="win", r=0.8, displacement_ratio=1.0, chased=False, exhaustion=False),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)
            patterns = aggregate_patterns(samples)

        displacement = next(
            pattern
            for pattern in patterns
            if pattern["feature"] == "displacement_ratio"
            and pattern["condition"] == {"feature": "displacement_ratio", "op": "gt", "value": 2.0}
        )
        self.assertEqual(displacement["sample_count"], 4)
        self.assertEqual(displacement["loss_count"], 3)
        self.assertEqual(displacement["win_count"], 1)
        self.assertAlmostEqual(displacement["avg_r"], (-1.0 - 0.5 - 1.2 + 2.0) / 4)
        self.assertLess(displacement["profit_factor"], 1.0)

    def test_duplicate_postmortem_json_enriches_sparse_learning_sample(self) -> None:
        sparse = _sample("same-trade", label="loss", r=-1.0)
        sparse["features"].pop("atr_regime_ratio")
        sparse["features"].pop("spread_points")
        rich = dict(sparse)
        rich["strategy_metadata"] = {"atr_regime_ratio": 1.35, "spread_points": 22.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "learning_samples.jsonl"
            _write_jsonl(input_path, [sparse])
            (tmp / "same-trade.json").write_text(json.dumps(rich), encoding="utf-8")
            samples = load_samples(input_path, postmortem_dir=tmp)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].features["atr_regime_ratio"], 1.35)
        self.assertEqual(samples[0].features["spread_points"], 22.0)

    def test_candidate_generation_respects_min_evidence_and_review_only(self) -> None:
        rows = [_sample(f"loss-{idx}", label="loss", r=-1.0) for idx in range(9)]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)
            patterns = aggregate_patterns(samples)

        self.assertEqual(generate_candidates(patterns), [])
        candidates = generate_candidates(patterns, min_samples_to_emit=3)
        self.assertGreater(candidates, [])
        for candidate in candidates:
            self.assertEqual(candidate["status"], "review_only")
            self.assertIn(candidate["candidate_action"], {"block", "delay_entry", "tighten_only_when_combo", "require_confirmation", "reduce_size"})
            self.assertFalse(candidate["safety"]["live_apply_allowed"])
            self.assertTrue(candidate["safety"]["needs_shadow_validation"])
            self.assertEqual(candidate["safety"]["min_samples_required"], 10)
            self.assertIn(candidate["safety"]["evidence_grade"], {"suspicion", "observation_only"})

    def test_shadow_evaluator_computes_blocked_counts_and_net_r(self) -> None:
        rows = [
            _sample("loss-1", label="loss", r=-1.0),
            _sample("loss-2", label="loss", r=-0.5),
            _sample("win-1", label="win", r=2.0),
            _sample("win-2", label="win", r=0.8, displacement_ratio=1.0, chased=False, exhaustion=False),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        candidate = {
            "candidate_id": "test-rule",
            "condition": {"feature": "displacement_ratio", "op": "gt", "value": 2.0},
        }
        shadow = shadow_evaluate(candidate, samples)
        self.assertEqual(shadow["blocked_losses"], 2)
        self.assertEqual(shadow["blocked_wins"], 1)
        self.assertEqual(shadow["blocked_losses_count"], 2)
        self.assertEqual(shadow["blocked_wins_count"], 1)
        self.assertEqual(shadow["trade_count_delta"], -3)
        self.assertAlmostEqual(shadow["net_r_delta"], -0.5)
        self.assertAlmostEqual(shadow["avoided_loss_r"], 1.5)
        self.assertAlmostEqual(shadow["sacrificed_win_r"], 2.0)
        self.assertAlmostEqual(shadow["missed_profit_r"], 2.0)
        self.assertAlmostEqual(shadow["false_block_rate"], 1 / 3)
        self.assertEqual(shadow["promotion_gate"]["status"], "blocked")
        self.assertIn("false_block_rate", shadow["promotion_gate"]["required_shadow_metrics"])
        self.assertIn("trade_frequency_delta", shadow["promotion_gate"]["required_shadow_metrics"])
        self.assertIn("insufficient_shadow_samples", shadow["promotion_gate"]["block_reasons"])
        self.assertIn("non_positive_net_r_delta", shadow["promotion_gate"]["block_reasons"])
        self.assertIn("표본", shadow["warning"])

    def test_regime_expectancy_splits_same_setup_by_volatility_and_spread(self) -> None:
        rows = [
            _sample("tight-low-win-1", label="win", r=1.0, atr_regime_ratio=0.8, spread_points=2.0),
            _sample("tight-low-win-2", label="win", r=0.6, atr_regime_ratio=0.85, spread_points=3.0),
            _sample("normal-mid-flat", label="flat", r=0.0, atr_regime_ratio=1.0, spread_points=8.0),
            _sample("wide-high-loss-1", label="loss", r=-1.0, atr_regime_ratio=1.4, spread_points=20.0),
            _sample("wide-high-loss-2", label="loss", r=-1.4, atr_regime_ratio=1.5, spread_points=25.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        summary = regime_expectancy_summary(samples)
        combined = {
            row["key"]: row for row in summary["dimensions"]["volatility_x_spread"]
        }
        setup_combined = {
            row["key"]: row for row in summary["dimensions"]["setup_x_volatility_spread"]
        }
        contrasts = {row["setup"]: row for row in summary["setup_regime_contrasts"]}
        self.assertEqual(summary["volatility_value_count"], 5)
        self.assertEqual(summary["spread_value_count"], 5)
        quality_by_symbol = {
            row["symbol"]: row for row in summary["data_quality"]["by_symbol"]
        }
        self.assertEqual(quality_by_symbol["BTCUSD"]["spread_threshold_status"], "quantile_thresholds")
        self.assertEqual(quality_by_symbol["BTCUSD"]["spread_missing_count"], 0)
        self.assertEqual(quality_by_symbol["BTCUSD"]["spread_coverage"], 1.0)
        self.assertAlmostEqual(
            combined["LOW_VOL|TIGHT_SPREAD"]["expectancy_r"],
            0.8,
        )
        self.assertAlmostEqual(
            combined["HIGH_VOL|WIDE_SPREAD"]["expectancy_r"],
            -1.2,
        )
        self.assertAlmostEqual(
            setup_combined["BTCUSD|liquidity_sweep_reversal|LOW_VOL|TIGHT_SPREAD"]["expectancy_r"],
            0.8,
        )
        self.assertAlmostEqual(
            setup_combined["BTCUSD|liquidity_sweep_reversal|HIGH_VOL|WIDE_SPREAD"]["expectancy_r"],
            -1.2,
        )
        self.assertAlmostEqual(
            contrasts["BTCUSD|liquidity_sweep_reversal"]["expectancy_gap_r"],
            2.0,
        )
        self.assertEqual(
            contrasts["BTCUSD|liquidity_sweep_reversal"]["best_regime"],
            "LOW_VOL|TIGHT_SPREAD",
        )

    def test_regime_expectancy_warns_on_partial_spread_coverage(self) -> None:
        rows = [
            _sample("spread-seen-1", label="win", r=0.8, spread_points=4.0),
            _sample("spread-seen-2", label="loss", r=-0.4, spread_points=24.0),
            _sample("spread-missing-1", label="win", r=0.3),
            _sample("spread-missing-2", label="loss", r=-0.6),
        ]
        rows[2]["features"].pop("spread_points")
        rows[3]["features"].pop("spread_points")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        summary = regime_expectancy_summary(samples)
        quality = summary["data_quality"]["by_symbol"][0]
        warnings = "\n".join(summary["warnings"])

        self.assertEqual(quality["symbol"], "BTCUSD")
        self.assertEqual(quality["spread_value_count"], 2)
        self.assertEqual(quality["spread_missing_count"], 2)
        self.assertEqual(quality["spread_threshold_status"], "observed_only_insufficient_symbol_samples")
        self.assertAlmostEqual(quality["spread_coverage"], 0.5)
        self.assertIn("spread 값이 2/4건뿐", warnings)
        self.assertIn("tight/normal/wide 절단값을 만들지 않고", warnings)

    def test_event_regime_context_reports_spread_telemetry_gaps_by_volatility(self) -> None:
        events = [
            {
                "event": "decision",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "action": "BUY",
                "reason": "LSR_BUY_ENTRY",
                "metadata": {"atr_regime_ratio": 0.82, "current_spread": 4.0},
            },
            {
                "event": "decision",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "action": "SELL",
                "reason": "LSR_SELL_ENTRY",
                "metadata": {"atr_regime_ratio": 1.31},
            },
            {
                "event": "order_skip",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "reason": "ENTRY_QUALITY_BLOCK",
                "details": {"atr_regime_ratio": 1.31},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            _write_jsonl(path, events)
            context = event_regime_context(path)

        by_vol = {row["key"]: row for row in context["entry_signal_by_volatility"]}
        blocked = {row["key"]: row for row in context["blocked_by_volatility"]}
        warnings = "\n".join(context["warnings"])

        self.assertTrue(context["available"])
        self.assertEqual(context["entry_signal_count"], 2)
        self.assertEqual(context["entry_signal_atr_value_count"], 2)
        self.assertEqual(context["entry_signal_spread_value_count"], 1)
        self.assertAlmostEqual(context["entry_signal_spread_coverage"], 0.5)
        self.assertEqual(by_vol["LOW_VOL"]["event_count"], 1)
        self.assertEqual(by_vol["HIGH_VOL"]["event_count"], 1)
        self.assertEqual(by_vol["HIGH_VOL"]["spread_value_count"], 0)
        self.assertEqual(blocked["HIGH_VOL"]["top_reasons"][0]["reason"], "ENTRY_QUALITY_BLOCK")
        self.assertIn("spread telemetry coverage가 낮다", warnings)

    def test_event_regime_context_inherits_entry_regime_for_following_block(self) -> None:
        events = [
            {
                "event": "decision",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "action": "SELL",
                "reason": "LSR_SELL_ENTRY",
                "metadata": {"atr_regime_ratio": 1.31, "current_spread": 18.0},
            },
            {
                "event": "order_skip",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "reason": "ENTRY_QUALITY_BLOCK",
                "details": {"allow": False, "score": 0.2, "threshold": 0.25},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            _write_jsonl(path, events)
            context = event_regime_context(path)

        blocked = {row["key"]: row for row in context["blocked_by_volatility"]}
        warnings = "\n".join(context["warnings"])

        self.assertEqual(context["blocked_signal_atr_inherited_count"], 1)
        self.assertEqual(context["blocked_signal_spread_inherited_count"], 1)
        self.assertEqual(blocked["HIGH_VOL"]["event_count"], 1)
        self.assertEqual(blocked["HIGH_VOL"]["atr_value_count"], 1)
        self.assertEqual(blocked["HIGH_VOL"]["spread_value_count"], 1)
        self.assertEqual(blocked["HIGH_VOL"]["top_reasons"][0]["reason"], "ENTRY_QUALITY_BLOCK")
        self.assertIn("직전 entry decision의 ATR 레짐을 상속", warnings)
        self.assertIn("직전 entry decision의 spread를 상속", warnings)

    def test_event_regime_context_crosses_volatility_and_spread_for_blockers(self) -> None:
        events = [
            {
                "event": "decision",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "action": "BUY",
                "reason": "LSR_BUY_ENTRY",
                "metadata": {"atr_regime_ratio": 0.82, "current_spread": 2.0},
            },
            {
                "event": "decision",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "action": "SELL",
                "reason": "LSR_SELL_ENTRY",
                "metadata": {"atr_regime_ratio": 1.05, "current_spread": 10.0},
            },
            {
                "event": "decision",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "action": "SELL",
                "reason": "LSR_SELL_ENTRY",
                "metadata": {"atr_regime_ratio": 1.31, "current_spread": 25.0},
            },
            {
                "event": "order_skip",
                "symbol": "BTCUSD",
                "strategy": "liquidity_sweep_reversal",
                "reason": "ENTRY_QUALITY_BLOCK",
                "details": {"allow": False},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            _write_jsonl(path, events)
            context = event_regime_context(path)

        entry_crossed = {row["key"]: row for row in context["entry_signal_by_volatility_spread"]}
        blocked_crossed = {row["key"]: row for row in context["blocked_by_volatility_spread"]}

        self.assertEqual(
            context["event_spread_thresholds_by_symbol"]["BTCUSD"],
            {"tight_cut": 2.0, "wide_cut": 10.0},
        )
        self.assertEqual(entry_crossed["LOW_VOL|TIGHT_SPREAD"]["event_count"], 1)
        self.assertEqual(entry_crossed["NORMAL_VOL|NORMAL_SPREAD"]["event_count"], 1)
        self.assertEqual(entry_crossed["HIGH_VOL|WIDE_SPREAD"]["event_count"], 1)
        self.assertEqual(blocked_crossed["HIGH_VOL|WIDE_SPREAD"]["event_count"], 1)
        self.assertEqual(blocked_crossed["HIGH_VOL|WIDE_SPREAD"]["top_reasons"][0]["reason"], "ENTRY_QUALITY_BLOCK")

    def test_execution_shortfall_separates_signal_edge_from_net_fill_quality(self) -> None:
        rows = [
            _sample(
                "signal-win-net-loss",
                label="loss",
                r=-0.2,
                price_r=0.4,
                entry_shortfall_r=0.15,
                execution_drag_r=0.6,
                spread_points=24.0,
            ),
            _sample(
                "clean-win",
                label="win",
                r=0.7,
                price_r=0.8,
                entry_shortfall_r=0.02,
                execution_drag_r=0.1,
                spread_points=4.0,
                chased=False,
                exhaustion=False,
                displacement_ratio=1.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        summary = execution_shortfall_summary(samples)
        overall = summary["overall"]

        self.assertTrue(summary["review_only"])
        self.assertEqual(overall["paired_r_value_count"], 2)
        self.assertAlmostEqual(overall["avg_signal_price_r"], 0.6)
        self.assertAlmostEqual(overall["avg_realized_net_r"], 0.25)
        self.assertAlmostEqual(overall["avg_execution_drag_r"], 0.35)
        self.assertAlmostEqual(overall["execution_drag_to_signal_ratio"], 0.35 / 0.6)
        self.assertAlmostEqual(overall["net_realization_ratio"], 0.25 / 0.6)
        self.assertAlmostEqual(overall["avg_entry_implementation_shortfall_r"], 0.085)
        self.assertEqual(overall["signal_positive_trade_count"], 2)
        self.assertEqual(overall["signal_positive_net_negative_count"], 1)
        self.assertEqual(overall["signal_positive_net_nonpositive_count"], 1)
        self.assertAlmostEqual(overall["signal_positive_net_nonpositive_rate"], 0.5)
        tuning_gate = summary["tuning_gate"]
        self.assertEqual(tuning_gate["status"], "blocked")
        self.assertTrue(tuning_gate["blocks_signal_threshold_tuning"])
        self.assertIn("insufficient_paired_samples", tuning_gate["reason_codes"])
        self.assertIn("frequent_positive_signal_net_nonpositive_trades", tuning_gate["reason_codes"])
        setup_rows = {row["key"]: row for row in summary["groups"]["by_symbol_strategy"]}
        self.assertIn("BTCUSD|liquidity_sweep_reversal", setup_rows)

    def test_execution_shortfall_blocks_tuning_when_drag_erodes_positive_edge(self) -> None:
        rows = [
            _sample(
                f"cost-eroded-{idx}",
                label="win",
                r=0.1,
                price_r=0.4,
                execution_drag_r=0.3,
                entry_shortfall_r=0.03,
                spread_points=18.0,
                chased=False,
                exhaustion=False,
                displacement_ratio=1.0,
            )
            for idx in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        summary = execution_shortfall_summary(samples)
        overall = summary["overall"]
        tuning_gate = summary["tuning_gate"]

        self.assertAlmostEqual(overall["avg_signal_price_r"], 0.4)
        self.assertAlmostEqual(overall["avg_realized_net_r"], 0.1)
        self.assertAlmostEqual(overall["execution_drag_to_signal_ratio"], 0.75)
        self.assertAlmostEqual(overall["net_realization_ratio"], 0.25)
        self.assertEqual(tuning_gate["status"], "blocked")
        self.assertIn("execution_drag_erodes_signal_edge", tuning_gate["reason_codes"])
        self.assertNotIn("insufficient_paired_samples", tuning_gate["reason_codes"])
        self.assertIn("50% 이상", " ".join(summary["warnings"]))

    def test_execution_shortfall_inversion_rate_uses_positive_signal_denominator(self) -> None:
        rows = [
            _sample(
                "only-positive-signal-eroded",
                label="flat",
                r=0.0,
                price_r=0.2,
                execution_drag_r=0.2,
                spread_points=20.0,
                chased=False,
                exhaustion=False,
                displacement_ratio=1.0,
            ),
            *[
                _sample(
                    f"negative-signal-{idx}",
                    label="loss",
                    r=-0.3,
                    price_r=-0.3,
                    execution_drag_r=0.0,
                    spread_points=8.0,
                    chased=False,
                    exhaustion=False,
                    displacement_ratio=1.0,
                )
                for idx in range(9)
            ],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        summary = execution_shortfall_summary(samples)
        overall = summary["overall"]
        tuning_gate = summary["tuning_gate"]

        self.assertEqual(overall["paired_r_value_count"], 10)
        self.assertEqual(overall["signal_positive_trade_count"], 1)
        self.assertEqual(overall["signal_positive_net_nonpositive_count"], 1)
        self.assertAlmostEqual(overall["signal_positive_net_nonpositive_rate"], 1.0)
        self.assertEqual(tuning_gate["status"], "blocked")
        self.assertIn("frequent_positive_signal_net_nonpositive_trades", tuning_gate["reason_codes"])

    def test_execution_shortfall_derives_cost_ratios_from_usd_fields(self) -> None:
        rows = [
            _sample(
                "cost-usd-only",
                label="win",
                r=0.2,
                price_r=0.6,
                execution_drag_r=0.4,
                estimated_cost_usd=0.30,
                explicit_cost_usd=0.10,
                expected_loss_usd=1.0,
                spread_points=16.0,
                chased=False,
                exhaustion=False,
                displacement_ratio=1.0,
            ),
            _sample(
                "cost-ratio-present",
                label="win",
                r=0.4,
                price_r=0.7,
                execution_drag_r=0.3,
                estimated_cost_r=0.20,
                explicit_cost_r=0.05,
                estimated_cost_usd=9.99,
                explicit_cost_usd=9.99,
                expected_loss_usd=1.0,
                spread_points=8.0,
                chased=False,
                exhaustion=False,
                displacement_ratio=1.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        overall = execution_shortfall_summary(samples)["overall"]

        self.assertAlmostEqual(overall["avg_estimated_cost_to_expected_loss_r"], 0.25)
        self.assertAlmostEqual(overall["avg_realized_explicit_cost_r"], 0.075)

    def test_naive_block_rejected_when_missed_winners_are_bigger(self) -> None:
        rows = [
            *[_sample(f"loss-{idx}", label="loss", r=-1.0) for idx in range(4)],
            *[_sample(f"win-{idx}", label="win", r=2.0) for idx in range(6)],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        shadow = shadow_evaluate(
            {
                "candidate_id": "naive-block",
                "candidate_action": "block",
                "condition": {"feature": "displacement_ratio", "op": "gt", "value": 2.0},
            },
            samples,
            max_trade_frequency_drop=1.0,
        )
        self.assertLess(shadow["net_r_delta"], 0)
        self.assertGreater(shadow["sacrificed_win_r"], shadow["avoided_loss_r"])
        self.assertFalse(shadow["live_review_ready"])
        self.assertIn("놓치는 수익", shadow["over_filtering_warning"])

    def test_combo_block_can_be_live_review_ready_with_small_missed_wins(self) -> None:
        rows = [
            *[_sample(f"loss-{idx}", label="loss", r=-1.0) for idx in range(20)],
            *[_sample(f"small-win-{idx}", label="win", r=0.2) for idx in range(2)],
            *[
                _sample(
                    f"clean-win-{idx}",
                    label="win",
                    r=0.8,
                    displacement_ratio=1.0,
                    adx_entry=20.0,
                    chased=False,
                    exhaustion=False,
                )
                for idx in range(30)
            ],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        candidate = {
            "candidate_id": "combo-risk",
            "candidate_action": "tighten_only_when_combo",
            "condition": {
                "all": [
                    {"feature": "adx_entry", "op": "gte", "value": 45.0},
                    {"feature": "entry_chased_extension", "op": "eq", "value": True},
                ]
            },
        }
        shadow = shadow_evaluate(candidate, samples)
        self.assertGreater(shadow["net_r_delta"], 0)
        self.assertLessEqual(shadow["false_block_rate"], 0.25)
        self.assertTrue(shadow["live_review_ready"])
        self.assertEqual(shadow["promotion_gate"]["status"], "pass")
        self.assertEqual(shadow["promotion_gate"]["block_reasons"], [])
        self.assertTrue(shadow["promotion_gate"]["metrics_present"]["false_block_rate"])
        self.assertTrue(shadow["promotion_gate"]["metrics_present"]["trade_frequency_delta"])

    def test_win_pattern_generates_relaxation_candidate(self) -> None:
        rows = [
            _sample(
                f"exception-win-{idx}",
                label="win",
                r=1.4,
                displacement_ratio=1.0,
                score=0.2,
                threshold=0.2,
                chased=False,
                exhaustion=False,
                clean_reclaim=True,
            )
            for idx in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)
            patterns = aggregate_patterns(samples)

        candidates = generate_candidates(patterns, min_samples_to_emit=3)
        actions = {candidate["candidate_action"] for candidate in candidates}
        self.assertIn("relax_threshold_candidate", actions)
        self.assertTrue(any("예외" in candidate["rationale_korean"] or candidate["feature"] == "low_score_clean_reclaim_exception" for candidate in candidates))

    def test_unconfirmed_lsr_reclaim_chase_generates_confirmation_candidate(self) -> None:
        rows = [
            _sample(
                f"unconfirmed-loss-{idx}",
                label="loss",
                r=-1.0,
                lsr_unconfirmed_reclaim=True,
                weak_reclaim_after_deep_sweep=True,
                lsr_unconfirmed_reclaim_chase=True,
            )
            for idx in range(4)
        ]
        rows.extend(
            _sample(
                f"confirmed-win-{idx}",
                label="win",
                r=1.0,
                displacement_ratio=1.0,
                chased=False,
                exhaustion=False,
                clean_reclaim=True,
            )
            for idx in range(4)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)
            patterns = aggregate_patterns(samples)

        candidates = generate_candidates(patterns, min_samples_to_emit=3)
        match = next(
            candidate
            for candidate in candidates
            if candidate["feature"] == "lsr_unconfirmed_reclaim_chase"
            and candidate["condition"] == {"feature": "lsr_unconfirmed_reclaim_chase", "op": "eq", "value": True}
        )
        self.assertEqual(match["candidate_action"], "require_confirmation")
        self.assertEqual(match["rule_type"], "required_confirm")
        self.assertIn("작은 표본으로 live gate를 바꾸면 안 된다", match["rationale_korean"])

    def test_lsr_confirmation_expectancy_splits_unconfirmed_from_retest(self) -> None:
        rows = [
            _sample(
                f"unconfirmed-loss-{idx}",
                label="loss",
                r=-1.0,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                lsr_unconfirmed_reclaim=True,
                shallow_reclaim_confirmation=True,
                weak_reclaim_after_deep_sweep=True,
                lsr_unconfirmed_reclaim_chase=True,
                reclaim_distance_atr=0.2,
                sweep_depth_atr=1.0,
                reclaim_to_sweep_depth_ratio=0.2,
                time_from_sweep_to_entry_sec=780.0,
                reclaim_window_sec=900.0,
            )
            for idx in range(3)
        ]
        rows.extend(
            _sample(
                f"retest-win-{idx}",
                label="win",
                r=1.5,
                displacement_ratio=1.0,
                chased=False,
                exhaustion=False,
                clean_reclaim=True,
                confirmation_path="retest",
                retest_confirmed=True,
                reclaim_distance_atr=0.45,
                sweep_depth_atr=0.9,
                reclaim_to_sweep_depth_ratio=0.5,
                time_from_sweep_to_entry_sec=120.0,
                reclaim_window_sec=900.0,
            )
            for idx in range(3)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        summary = lsr_confirmation_expectancy_summary(samples)
        by_path = {row["key"]: row for row in summary["dimensions"]["by_confirmation_path"]}

        self.assertTrue(summary["review_only"])
        self.assertEqual(summary["sample_count"], 6)
        self.assertEqual(summary["unconfirmed_reclaim_count"], 3)
        self.assertEqual(summary["retest_confirmed_count"], 3)
        self.assertEqual(summary["unknown_confirmation_count"], 0)
        self.assertEqual(summary["reclaim_metric_complete_count"], 6)
        self.assertEqual(summary["reclaim_timing_complete_count"], 6)
        self.assertEqual(summary["late_window_reclaim_count"], 3)
        self.assertEqual(summary["shallow_reclaim_confirmation_count"], 3)
        self.assertEqual(summary["confirmation_score_complete_count"], 6)
        self.assertAlmostEqual(by_path["reclaim_only"]["expectancy_r"], -1.0)
        self.assertAlmostEqual(by_path["retest"]["expectancy_r"], 1.5)
        self.assertEqual(by_path["reclaim_only"]["weak_reclaim_after_deep_sweep_count"], 3)
        self.assertEqual(by_path["reclaim_only"]["shallow_reclaim_confirmation_count"], 3)
        self.assertEqual(by_path["reclaim_only"]["late_window_reclaim_count"], 3)
        self.assertEqual(by_path["reclaim_only"]["unconfirmed_reclaim_chase_count"], 3)
        self.assertAlmostEqual(by_path["reclaim_only"]["avg_reclaim_window_elapsed_ratio"], 780 / 900)
        self.assertLess(by_path["reclaim_only"]["avg_lsr_confirmation_score"], by_path["retest"]["avg_lsr_confirmation_score"])
        self.assertIn("10건 미만", " ".join(summary["warnings"]))

    def test_lsr_confirmation_expectancy_does_not_treat_missing_metadata_as_unconfirmed(self) -> None:
        rows = [
            _sample("unknown-loss", label="loss", r=-1.0),
            _sample(
                "unconfirmed-loss",
                label="loss",
                r=-0.8,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                lsr_unconfirmed_reclaim=True,
                reclaim_distance_atr=0.25,
                sweep_depth_atr=0.8,
                reclaim_to_sweep_depth_ratio=0.3125,
            ),
            _sample(
                "retest-win",
                label="win",
                r=1.2,
                confirmation_path="retest",
                retest_confirmed=True,
                clean_reclaim=True,
                reclaim_distance_atr=0.5,
                sweep_depth_atr=0.7,
                reclaim_to_sweep_depth_ratio=0.714,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        summary = lsr_confirmation_expectancy_summary(samples)
        metadata_quality = summary["dimensions"]["metadata_quality"]

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["unconfirmed_reclaim_count"], 1)
        self.assertEqual(summary["retest_confirmed_count"], 1)
        self.assertEqual(summary["unknown_confirmation_count"], 1)
        self.assertEqual(metadata_quality["path_missing_count"], 1)
        self.assertEqual(metadata_quality["reclaim_metric_missing_count"], 1)
        self.assertEqual(metadata_quality["reclaim_timing_missing_count"], 3)
        self.assertIsNotNone(summary["dimensions"]["unknown_confirmation"])
        self.assertIn("unknown으로 분리", " ".join(summary["warnings"]))

    def test_lsr_confirmation_expectancy_derives_late_window_reclaim(self) -> None:
        rows = [
            _sample(
                "late-loss",
                label="loss",
                r=-1.0,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                reclaim_distance_atr=0.3,
                sweep_depth_atr=0.6,
                reclaim_to_sweep_depth_ratio=0.5,
                time_from_sweep_to_entry_sec=760.0,
                reclaim_window_sec=900.0,
            ),
            _sample(
                "quick-loss",
                label="loss",
                r=-0.5,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                reclaim_distance_atr=0.3,
                sweep_depth_atr=0.6,
                reclaim_to_sweep_depth_ratio=0.5,
                time_from_sweep_to_entry_sec=120.0,
                reclaim_window_sec=900.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        by_id = {sample.trade_id: sample for sample in samples}
        self.assertTrue(by_id["late-loss"].features["late_window_reclaim"])
        self.assertTrue(by_id["late-loss"].features["lsr_unconfirmed_reclaim_chase"])
        self.assertAlmostEqual(by_id["late-loss"].features["reclaim_window_elapsed_ratio"], 760 / 900)
        self.assertNotIn("late_window_reclaim", by_id["quick-loss"].features)

    def test_lsr_confirmation_expectancy_accepts_reclaim_time_alias(self) -> None:
        rows = [
            _sample(
                "reclaim-alias-loss",
                label="loss",
                r=-1.0,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                reclaim_distance_atr=0.3,
                sweep_depth_atr=0.6,
                reclaim_to_sweep_depth_ratio=0.5,
                time_from_sweep_to_reclaim_sec=760.0,
                reclaim_window_sec=900.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        sample = samples[0]
        summary = lsr_confirmation_expectancy_summary(samples)
        metadata_quality = summary["dimensions"]["metadata_quality"]

        self.assertTrue(sample.features["late_window_reclaim"])
        self.assertTrue(sample.features["lsr_unconfirmed_reclaim_chase"])
        self.assertAlmostEqual(sample.features["reclaim_window_elapsed_ratio"], 760 / 900)
        self.assertEqual(summary["reclaim_timing_complete_count"], 1)
        self.assertEqual(metadata_quality["reclaim_timing_complete_count"], 1)
        self.assertEqual(metadata_quality["reclaim_timing_missing_count"], 0)

    def test_lsr_confirmation_expectancy_flags_invalid_negative_sweep_timing(self) -> None:
        rows = [
            _sample(
                "bad-time-loss",
                label="loss",
                r=-1.0,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                reclaim_distance_atr=0.4,
                sweep_depth_atr=0.8,
                reclaim_to_sweep_depth_ratio=0.5,
                time_from_sweep_to_entry_sec=-60.0,
                reclaim_window_sec=900.0,
            ),
            _sample(
                "good-time-loss",
                label="loss",
                r=-0.5,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                reclaim_distance_atr=0.4,
                sweep_depth_atr=0.8,
                reclaim_to_sweep_depth_ratio=0.5,
                time_from_sweep_to_entry_sec=120.0,
                reclaim_window_sec=900.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        by_id = {sample.trade_id: sample for sample in samples}
        summary = lsr_confirmation_expectancy_summary(samples)
        by_path = {row["key"]: row for row in summary["dimensions"]["by_confirmation_path"]}
        metadata_quality = summary["dimensions"]["metadata_quality"]

        self.assertTrue(by_id["bad-time-loss"].features["invalid_reclaim_timing"])
        self.assertNotIn("reclaim_window_elapsed_ratio", by_id["bad-time-loss"].features)
        self.assertFalse(by_id["bad-time-loss"].features.get("late_window_reclaim", False))
        self.assertEqual(summary["invalid_reclaim_timing_count"], 1)
        self.assertEqual(metadata_quality["invalid_reclaim_timing_count"], 1)
        self.assertEqual(by_path["reclaim_only"]["invalid_reclaim_timing_count"], 1)
        self.assertIn("음수", " ".join(summary["warnings"]))

    def test_lsr_confirmation_expectancy_derives_shallow_reclaim(self) -> None:
        rows = [
            _sample(
                "shallow-loss",
                label="loss",
                r=-1.0,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                reclaim_distance_atr=0.1,
                sweep_depth_atr=0.2,
                reclaim_to_sweep_depth_ratio=0.5,
            ),
            _sample(
                "strong-loss",
                label="loss",
                r=-0.5,
                confirmation_path="reclaim_only",
                retest_confirmed=False,
                reclaim_distance_atr=0.5,
                sweep_depth_atr=0.2,
                reclaim_to_sweep_depth_ratio=2.5,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        by_id = {sample.trade_id: sample for sample in samples}
        summary = lsr_confirmation_expectancy_summary(samples)
        by_path = {row["key"]: row for row in summary["dimensions"]["by_confirmation_path"]}

        self.assertTrue(by_id["shallow-loss"].features["shallow_reclaim_confirmation"])
        self.assertTrue(by_id["shallow-loss"].features["lsr_unconfirmed_reclaim_chase"])
        self.assertEqual(by_id["shallow-loss"].features["shallow_reclaim_threshold_atr"], 0.25)
        self.assertNotIn("shallow_reclaim_confirmation", by_id["strong-loss"].features)
        self.assertEqual(summary["shallow_reclaim_confirmation_count"], 1)
        self.assertEqual(by_path["reclaim_only"]["shallow_reclaim_confirmation_count"], 1)

    def test_over_filtering_warning_when_blocking_many_winners(self) -> None:
        rows = [
            *[_sample(f"loss-{idx}", label="loss", r=-0.5) for idx in range(2)],
            *[_sample(f"win-{idx}", label="win", r=1.0) for idx in range(8)],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "learning_samples.jsonl"
            _write_jsonl(input_path, rows)
            samples = load_samples(input_path)

        shadow = shadow_evaluate(
            {
                "candidate_id": "too-strict",
                "candidate_action": "block",
                "condition": {"feature": "entry_chased_extension", "op": "eq", "value": True},
            },
            samples,
            max_trade_frequency_drop=1.0,
        )
        self.assertGreater(shadow["false_block_rate"], 0.25)
        self.assertIn("과보수", shadow["over_filtering_warning"])

    def test_cli_help_works_and_analysis_writes_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        help_result = subprocess.run(
            [sys.executable, "tools/analyze_postmortem_learning.py", "--help"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--min-samples", help_result.stdout)
        self.assertIn("--max-false-block-rate", help_result.stdout)
        self.assertIn("--max-trade-frequency-drop", help_result.stdout)
        self.assertIn("--events", help_result.stdout)

        rows = [
            _sample("loss-1", label="loss", r=-1.0),
            _sample("loss-2", label="loss", r=-0.5),
            _sample("loss-3", label="loss", r=-1.2),
            _sample("win-1", label="win", r=2.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "learning_samples.jsonl"
            output_dir = tmp / "reports"
            _write_jsonl(input_path, rows)
            result = analyze_learning(
                argparse.Namespace(
                    input=str(input_path),
                    postmortem_dir=None,
                    output_dir=str(output_dir),
                    min_samples=1,
                    shadow_candidates=None,
                    limit_candidates=10,
                )
            )
            self.assertEqual(result["sample_count"], 4)
            self.assertGreater(result["candidate_count"], 0)
            self.assertTrue((output_dir / "learning_aggregates.json").exists())
            self.assertTrue((output_dir / "rule_candidates.jsonl").exists())
            self.assertTrue((output_dir / "shadow_evaluations.jsonl").exists())
            self.assertTrue((output_dir / "learning_review.md").exists())
            aggregate = json.loads((output_dir / "learning_aggregates.json").read_text(encoding="utf-8"))
            review_text = (output_dir / "learning_review.md").read_text(encoding="utf-8")
            self.assertIn("execution_shortfall", aggregate)
            self.assertIn("tuning_gate", aggregate["execution_shortfall"])
            self.assertIn("lsr_confirmation_expectancy", aggregate)
            self.assertIn("TCA Execution Shortfall", review_text)
            self.assertIn("TCA tuning gate", review_text)
            self.assertIn("symbol별 레짐 데이터 품질", review_text)
            self.assertIn("LSR Confirmation Path Expectancy", review_text)
            first_candidate = json.loads((output_dir / "rule_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_candidate["status"], "review_only")
            self.assertIn("candidate_action", first_candidate)
            self.assertIn("opportunity_cost", first_candidate)
            self.assertIn("expected_benefit", first_candidate)
            self.assertFalse(first_candidate["live_review_ready"])
            self.assertFalse(first_candidate["safety"]["live_apply_allowed"])
            self.assertIn("false_block_rate", first_candidate["safety"]["live_review_requires_shadow_metrics"])
            self.assertIn("trade_frequency_delta", first_candidate["safety"]["live_review_requires_shadow_metrics"])
            self.assertIn("promotion_gate_block_reasons", first_candidate["safety"])


if __name__ == "__main__":
    unittest.main()
