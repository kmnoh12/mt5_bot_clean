from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.liquidity import classify_lsr_confirmation_quality


DEFAULT_INPUT = "reports/trade_postmortems/learning_samples.jsonl"
DEFAULT_OUTPUT_DIR = "reports/trade_postmortems"
CANDIDATES_FILE = "rule_candidates.jsonl"
REVIEW_FILE = "learning_review.md"
AGGREGATES_FILE = "learning_aggregates.json"
SHADOW_FILE = "shadow_evaluations.jsonl"

OBSERVATION_MIN = 1
SUSPICION_MIN = 3
CANDIDATE_MIN = 10
LIVE_REVIEW_MIN = 20
DEFAULT_MAX_FALSE_BLOCK_RATE = 0.25
DEFAULT_MAX_TRADE_FREQUENCY_DROP = 0.50
REQUIRED_SHADOW_METRICS = [
    "blocked_total",
    "net_r_delta",
    "false_block_rate",
    "trade_frequency_delta",
]
VOLATILITY_REGIME_FEATURES = ("atr_regime_ratio", "volatility_regime_ratio")
SPREAD_REGIME_FEATURES = ("spread_points", "current_spread", "current_spread_points", "spread")
COST_FEATURES = ("estimated_cost_usd", "current_cost", "round_trip_cost", "estimated_round_trip_cost")

RISK_REDUCTION_ACTIONS = {
    "block",
    "require_confirmation",
    "delay_entry",
    "reduce_size",
    "tighten_only_when_combo",
}
RELAXATION_ACTIONS = {
    "relax_threshold_candidate",
    "allow_exception_candidate",
}
ACTION_TYPES = sorted(RISK_REDUCTION_ACTIONS | RELAXATION_ACTIONS | {"no_change"})

NUMERIC_THRESHOLDS: Dict[str, List[Tuple[str, float]]] = {
    "displacement_ratio": [("gt", 2.0), ("gte", 1.5), ("gte", 3.0)],
    "adx_entry": [("gte", 25.0), ("gte", 45.0), ("gte", 60.0)],
    "entry_quality_margin": [("lte", 0.0), ("lte", 0.03), ("lte", 0.05)],
    "price_r_multiple": [("lte", -1.0), ("lt", 0.0), ("gte", 1.0)],
    "pnl_r_multiple": [("lte", -1.0), ("lt", 0.0), ("gte", 1.0)],
    "m5_align": [("lte", 0.0), ("gte", 1.0)],
    "entry_position_in_recent_range": [("lte", 0.2), ("gte", 0.8)],
    "time_from_sweep_to_entry_sec": [("lte", 0.0), ("gte", 300.0), ("gte", 900.0)],
    "reclaim_window_elapsed_ratio": [("gte", 0.75), ("gte", 1.0)],
    "reclaim_distance_atr": [("lte", 0.1), ("lte", 0.25), ("gte", 0.5), ("gte", 1.0), ("gte", 1.5)],
    "sweep_depth_atr": [("lte", 0.05), ("gte", 0.5), ("gte", 1.0)],
    "reclaim_to_sweep_depth_ratio": [("gte", 1.0), ("gte", 2.0), ("gte", 3.0)],
    "lsr_confirmation_score": [("lte", 0.35), ("lte", 0.5), ("gte", 0.7)],
    "entry_implementation_shortfall_r": [("gt", 0.0), ("gte", 0.1), ("gte", 0.25)],
    "net_execution_drag_r": [("gt", 0.0), ("gte", 0.05), ("gte", 0.1)],
    "estimated_cost_to_expected_loss_r": [("gte", 0.05), ("gte", 0.1), ("gte", 0.25)],
    "realized_explicit_cost_r": [("gte", 0.01), ("gte", 0.05), ("gte", 0.1)],
}

GENERIC_NUMERIC_BINS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("<0", None, 0.0),
    ("0..0.5", 0.0, 0.5),
    ("0.5..1", 0.5, 1.0),
    ("1..1.5", 1.0, 1.5),
    ("1.5..2", 1.5, 2.0),
    ("2..3", 2.0, 3.0),
    (">=3", 3.0, None),
]

BOOLEAN_FEATURES = [
    "entry_chased_extension",
    "entered_into_exhaustion",
    "entered_against_short_term_momentum",
    "bar_chase_flag",
    "bar_exhaustion_flag",
    "sl_tp_geometry_enough_after_costs",
    "clean_reclaim",
    "clean_reclaim_confirmed",
    "retest_confirmed",
    "lsr_unconfirmed_reclaim",
    "shallow_reclaim_confirmation",
    "weak_reclaim_after_deep_sweep",
    "late_window_reclaim",
    "invalid_reclaim_timing",
    "lsr_unconfirmed_reclaim_chase",
]

CATEGORICAL_FEATURES = [
    "symbol",
    "side",
    "strategy",
    "quality_grade",
    "bar_source",
    "exit_inferred_type",
    "last_swing_direction",
    "confirmation_path",
    "lsr_confirmation_band",
]

FEATURE_PRIORITY = {
    "high_adx_reversal_chase": 100,
    "chase_exhaustion_combo": 95,
    "displacement_ratio": 90,
    "entry_quality_margin": 85,
    "entry_quality_at_threshold": 82,
    "entry_chased_extension": 80,
    "entered_into_exhaustion": 78,
    "entered_against_short_term_momentum": 76,
    "lsr_unconfirmed_reclaim_chase": 75,
    "shallow_reclaim_confirmation": 74,
    "weak_reclaim_after_deep_sweep": 74,
    "lsr_unconfirmed_reclaim": 73,
    "late_window_reclaim": 73,
    "invalid_reclaim_timing": 72,
    "lsr_confirmation_score": 71,
    "adx_entry": 72,
    "bar_chase_flag": 68,
    "bar_exhaustion_flag": 66,
    "reclaim_to_sweep_depth_ratio": 64,
    "reclaim_distance_atr": 63,
    "sweep_depth_atr": 62,
    "clean_reclaim_confirmed": 61,
    "price_r_multiple": 60,
    "pnl_r_multiple": 60,
}


@dataclass
class LearningSample:
    trade_id: str
    label: str
    r: Optional[float]
    pnl: Optional[float]
    features: Dict[str, Any]
    quality_flags: Dict[str, Any]
    raw: Dict[str, Any]


@dataclass
class PatternStats:
    pattern_id: str
    feature: str
    condition: Dict[str, Any]
    rule_type: str
    description_korean: str
    sample_count: int = 0
    loss_count: int = 0
    win_count: int = 0
    flat_count: int = 0
    r_values: List[float] = field(default_factory=list)
    source_trade_ids: List[str] = field(default_factory=list)

    def add(self, sample: LearningSample) -> None:
        self.sample_count += 1
        if sample.label == "loss":
            self.loss_count += 1
        elif sample.label == "win":
            self.win_count += 1
        else:
            self.flat_count += 1
        if sample.r is not None:
            self.r_values.append(sample.r)
        if len(self.source_trade_ids) < 12:
            self.source_trade_ids.append(sample.trade_id)

    def summary(self) -> Dict[str, Any]:
        positive = sum(value for value in self.r_values if value > 0)
        negative = sum(value for value in self.r_values if value < 0)
        avg_r = sum(self.r_values) / len(self.r_values) if self.r_values else None
        profit_factor = None
        if negative < 0:
            profit_factor = positive / abs(negative)
        elif positive > 0:
            profit_factor = math.inf
        return {
            "pattern_id": self.pattern_id,
            "feature": self.feature,
            "condition": self.condition,
            "rule_type": self.rule_type,
            "description_korean": self.description_korean,
            "sample_count": self.sample_count,
            "loss_count": self.loss_count,
            "win_count": self.win_count,
            "flat_count": self.flat_count,
            "loss_rate": self.loss_count / self.sample_count if self.sample_count else None,
            "avg_r": avg_r,
            "expectancy_r": avg_r,
            "profit_factor": profit_factor,
            "source_trade_ids": self.source_trade_ids,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only learning review for post-close MT5 trade postmortems. "
            "It aggregates closed-trade learning samples, emits review-only rule "
            "candidates, and shadow-evaluates candidate blocks without touching live gates."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="learning_samples.jsonl, a postmortem JSON file, or a directory")
    parser.add_argument("--postmortem-dir", default=None, help="Optional directory of postmortem JSON reports to merge")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for learning reports")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=CANDIDATE_MIN,
        help=(
            "Minimum matching sample count required to emit a review-only candidate row. "
            "The candidate safety block still requires 10+ for actual candidate status."
        ),
    )
    parser.add_argument(
        "--shadow-candidates",
        default=None,
        help="Optional candidate JSON/JSONL file to shadow-evaluate instead of generated candidates",
    )
    parser.add_argument("--limit-candidates", type=int, default=25, help="Maximum generated candidate rows")
    parser.add_argument(
        "--max-false-block-rate",
        type=float,
        default=DEFAULT_MAX_FALSE_BLOCK_RATE,
        help="Maximum winner share a risk-reduction candidate may block before live-review is allowed",
    )
    parser.add_argument(
        "--max-trade-frequency-drop",
        type=float,
        default=DEFAULT_MAX_TRADE_FREQUENCY_DROP,
        help="Maximum estimated trade-frequency drop before over-filtering warnings block live-review",
    )
    parser.add_argument(
        "--min-live-review-samples",
        type=int,
        default=LIVE_REVIEW_MIN,
        help="Minimum shadow-matched samples required before a candidate can be marked live-review-ready",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Optional events JSONL path for signal/block/trade-frequency context in the starvation health section",
    )
    return parser.parse_args(argv)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _lsr_reclaim_age_sec(features: Dict[str, Any]) -> Optional[float]:
    age_sec = _safe_float(features.get("time_from_sweep_to_entry_sec"))
    if age_sec is None:
        age_sec = _safe_float(features.get("time_from_sweep_to_reclaim_sec"))
    return age_sec


def _derive_lsr_timing_features(features: Dict[str, Any]) -> None:
    strategy = str(features.get("strategy") or "").strip().lower()
    entry_style = str(features.get("entry_style") or "").strip().lower()
    is_lsr = strategy.startswith("liquidity_sweep_reversal") or entry_style.startswith("liquidity_sweep_reversal")
    if not is_lsr:
        return

    path = str(features.get("confirmation_path") or "").strip().lower()
    retest_confirmed = bool(features.get("retest_confirmed"))
    explicit_unconfirmed = bool(features.get("lsr_unconfirmed_reclaim"))
    if not path and not retest_confirmed and not explicit_unconfirmed:
        return
    unconfirmed_reclaim = bool(
        explicit_unconfirmed
        or (not retest_confirmed and path in {"reclaim_only", "tick_reclaim"})
    )
    if unconfirmed_reclaim:
        features.setdefault("lsr_unconfirmed_reclaim", True)
    reclaim_atr = _safe_float(features.get("reclaim_distance_atr"))
    if unconfirmed_reclaim and reclaim_atr is not None and reclaim_atr <= 0.25:
        features.setdefault("shallow_reclaim_confirmation", True)
        features.setdefault("shallow_reclaim_threshold_atr", 0.25)
        features.setdefault("lsr_unconfirmed_reclaim_chase", True)
    age_sec = _lsr_reclaim_age_sec(features)
    invalid_reclaim_timing = bool(age_sec is not None and age_sec < 0.0)
    if invalid_reclaim_timing:
        features.setdefault("invalid_reclaim_timing", True)
    window_sec = _safe_float(features.get("reclaim_window_sec"))
    elapsed_ratio = None
    if age_sec is not None and age_sec >= 0.0 and window_sec is not None and window_sec > 0.0:
        elapsed_ratio = age_sec / window_sec
    if elapsed_ratio is not None:
        features.setdefault("reclaim_window_elapsed_ratio", elapsed_ratio)
    late_window_reclaim = bool(unconfirmed_reclaim and elapsed_ratio is not None and elapsed_ratio >= 0.75)
    if late_window_reclaim:
        features.setdefault("late_window_reclaim", True)
        features.setdefault("lsr_unconfirmed_reclaim_chase", True)

    flags = classify_lsr_confirmation_quality(
        confirmation_path=path,
        retest_confirmed=retest_confirmed,
        reclaim_distance_atr=reclaim_atr,
        sweep_depth_atr=features.get("sweep_depth_atr"),
        reclaim_to_sweep_depth_ratio=features.get("reclaim_to_sweep_depth_ratio"),
        displacement_ratio=features.get("displacement_ratio"),
        time_from_sweep_to_reclaim_sec=age_sec,
        reclaim_window_sec=window_sec,
        entered_into_exhaustion=bool(features.get("entered_into_exhaustion")),
        is_lsr=True,
    )
    for key in ("confirmation_score", "confirmation_band", "confirmation_score_components"):
        value = flags.get(key)
        if value is not None:
            features.setdefault(f"lsr_{key}", value)
    for key in (
        "lsr_unconfirmed_reclaim",
        "shallow_reclaim_confirmation",
        "weak_reclaim_after_deep_sweep",
        "late_window_reclaim",
        "invalid_reclaim_timing",
        "lsr_unconfirmed_reclaim_chase",
    ):
        if flags.get(key) is True:
            features.setdefault(key, True)
    if flags.get("shallow_reclaim_confirmation") is True:
        features.setdefault("shallow_reclaim_threshold_atr", flags.get("shallow_reclaim_threshold_atr"))
    if flags.get("reclaim_window_elapsed_ratio") is not None:
        features.setdefault("reclaim_window_elapsed_ratio", flags.get("reclaim_window_elapsed_ratio"))


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _read_json_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        item = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(item, list):
        return [row for row in item if isinstance(row, dict)]
    if isinstance(item, dict):
        return [item]
    return []


def _looks_like_postmortem_json(path: Path) -> bool:
    if path.name in {CANDIDATES_FILE, AGGREGATES_FILE, SHADOW_FILE}:
        return False
    return path.suffix.lower() == ".json"


def load_raw_rows(input_path: Path, postmortem_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if input_path.is_dir():
        for path in sorted(input_path.glob("*.jsonl")):
            if path.name == CANDIDATES_FILE:
                continue
            rows.extend(_read_jsonl(path))
        for path in sorted(input_path.glob("*.json")):
            if _looks_like_postmortem_json(path):
                rows.extend(_read_json_file(path))
    elif input_path.suffix.lower() == ".jsonl":
        rows.extend(_read_jsonl(input_path))
    elif input_path.suffix.lower() == ".json":
        rows.extend(_read_json_file(input_path))

    if postmortem_dir and postmortem_dir.exists():
        for path in sorted(postmortem_dir.glob("*.json")):
            if _looks_like_postmortem_json(path):
                rows.extend(_read_json_file(path))
    return rows


def normalize_sample(row: Dict[str, Any]) -> Optional[LearningSample]:
    features = dict(row.get("features") or {})
    quality_flags = dict(row.get("quality_flags") or {})
    for key, value in quality_flags.items():
        features.setdefault(key, value)
    strategy_metadata = row.get("strategy_metadata")
    if isinstance(strategy_metadata, dict):
        for key, value in strategy_metadata.items():
            features.setdefault(key, value)

    trade_id = str(
        row.get("trade_key")
        or row.get("trade_id")
        or row.get("id")
        or row.get("ticket")
        or ""
    ).strip()
    if not trade_id:
        return None

    outcome = str(row.get("outcome") or row.get("label") or "").strip().lower()
    pnl = _safe_float(row.get("pnl"))
    if outcome in {"win", "winner", "profit"}:
        label = "win"
    elif outcome in {"loss", "loser"}:
        label = "loss"
    elif outcome in {"flat", "breakeven", "scratch"}:
        label = "flat"
    elif pnl is not None:
        label = "win" if pnl > 0 else "loss" if pnl < 0 else "flat"
    else:
        label = "unknown"

    score = _safe_float(features.get("entry_quality_score"))
    threshold = _safe_float(features.get("entry_quality_threshold"))
    if score is not None and threshold is not None:
        features["entry_quality_margin"] = score - threshold
        features["entry_quality_at_threshold"] = abs(score - threshold) <= 1e-9
    else:
        features.setdefault("entry_quality_margin", None)
        features.setdefault("entry_quality_at_threshold", None)

    r = _safe_float(features.get("pnl_r_multiple"))
    if r is None:
        r = _safe_float(features.get("price_r_multiple"))
    if r is None:
        r = _safe_float(row.get("r"))

    for key in (*VOLATILITY_REGIME_FEATURES, *SPREAD_REGIME_FEATURES, *COST_FEATURES, "tp_profile", "chop_score"):
        if key in row:
            features.setdefault(key, row.get(key))
    for key in ("symbol", "side", "strategy", "quality_grade", "bar_source"):
        if key in row:
            features.setdefault(key, row.get(key))
    _derive_lsr_timing_features(features)

    return LearningSample(
        trade_id=trade_id,
        label=label,
        r=r,
        pnl=pnl,
        features=features,
        quality_flags=quality_flags,
        raw=row,
    )


def load_samples(input_path: Path, postmortem_dir: Optional[Path] = None) -> List[LearningSample]:
    samples: List[LearningSample] = []
    by_id: Dict[str, LearningSample] = {}
    for row in load_raw_rows(input_path, postmortem_dir):
        sample = normalize_sample(row)
        if sample is None:
            continue
        existing = by_id.get(sample.trade_id)
        if existing is not None:
            for key, value in sample.features.items():
                if existing.features.get(key) is None and value is not None:
                    existing.features[key] = value
            for key, value in sample.quality_flags.items():
                if existing.quality_flags.get(key) is None and value is not None:
                    existing.quality_flags[key] = value
            if existing.r is None and sample.r is not None:
                existing.r = sample.r
            if existing.pnl is None and sample.pnl is not None:
                existing.pnl = sample.pnl
            continue
        by_id[sample.trade_id] = sample
        samples.append(sample)
    return samples


def sample_value(sample: LearningSample, feature: str) -> Any:
    if feature in sample.features:
        return sample.features.get(feature)
    if feature in sample.quality_flags:
        return sample.quality_flags.get(feature)
    return sample.raw.get(feature)


def _compare(value: Any, op: str, expected: Any) -> bool:
    if op in {"eq", "ne"}:
        result = value == expected
        return result if op == "eq" else not result
    value_f = _safe_float(value)
    expected_f = _safe_float(expected)
    if value_f is None or expected_f is None:
        return False
    if op == "gt":
        return value_f > expected_f
    if op == "gte":
        return value_f >= expected_f
    if op == "lt":
        return value_f < expected_f
    if op == "lte":
        return value_f <= expected_f
    return False


def condition_matches(sample: LearningSample, condition: Dict[str, Any]) -> bool:
    if "all" in condition:
        parts = condition.get("all")
        return isinstance(parts, list) and all(condition_matches(sample, part) for part in parts)
    if "any" in condition:
        parts = condition.get("any")
        return isinstance(parts, list) and any(condition_matches(sample, part) for part in parts)
    feature = str(condition.get("feature") or "")
    op = str(condition.get("op") or "eq")
    expected = condition.get("value")
    if not feature:
        return False
    return _compare(sample_value(sample, feature), op, expected)


def _condition_id(condition: Dict[str, Any]) -> str:
    return hashlib.sha1(_json_dumps(condition).encode("utf-8")).hexdigest()[:12]


def _make_stats(
    feature: str,
    condition: Dict[str, Any],
    rule_type: str,
    description_korean: str,
) -> PatternStats:
    pattern_id = f"{feature}:{_condition_id(condition)}"
    return PatternStats(
        pattern_id=pattern_id,
        feature=feature,
        condition=condition,
        rule_type=rule_type,
        description_korean=description_korean,
    )


def _condition_text(feature: str, op: str, value: Any) -> str:
    op_text = {
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "eq": "==",
        "ne": "!=",
    }.get(op, op)
    return f"{feature} {op_text} {value}"


def pattern_definitions(samples: Sequence[LearningSample]) -> List[PatternStats]:
    definitions: Dict[str, PatternStats] = {}

    observed_features = sorted({key for sample in samples for key in sample.features.keys()})

    def has_numeric_value(feature: str) -> bool:
        for sample in samples:
            value = sample_value(sample, feature)
            if isinstance(value, bool):
                continue
            if _safe_float(value) is not None:
                return True
        return False

    numeric_features = [
        feature
        for feature in observed_features
        if has_numeric_value(feature)
    ]
    for feature in numeric_features:
        thresholds = NUMERIC_THRESHOLDS.get(feature)
        if thresholds:
            for op, value in thresholds:
                condition = {"feature": feature, "op": op, "value": value}
                description = f"{_condition_text(feature, op, value)} 구간"
                definitions[_condition_id(condition)] = _make_stats(
                    feature,
                    condition,
                    _rule_type_for(feature, op),
                    description,
                )
        else:
            for label, lower, upper in GENERIC_NUMERIC_BINS:
                parts: List[Dict[str, Any]] = []
                if lower is not None:
                    parts.append({"feature": feature, "op": "gte", "value": lower})
                if upper is not None:
                    parts.append({"feature": feature, "op": "lt", "value": upper})
                if not parts:
                    continue
                condition = parts[0] if len(parts) == 1 else {"all": parts}
                description = f"{feature} bin {label}"
                definitions[_condition_id(condition)] = _make_stats(
                    feature,
                    condition,
                    "block_combo" if len(parts) > 1 else _rule_type_for(feature, parts[0]["op"]),
                    description,
                )

    for feature in BOOLEAN_FEATURES:
        if any(sample_value(sample, feature) is not None for sample in samples):
            for expected in (True, False):
                condition = {"feature": feature, "op": "eq", "value": expected}
                description = f"{feature} == {expected} 플래그"
                definitions[_condition_id(condition)] = _make_stats(
                    feature,
                    condition,
                    "required_confirm" if expected is True else "block_combo",
                    description,
                )

    for feature in CATEGORICAL_FEATURES:
        values = sorted(
            {
                str(sample_value(sample, feature))
                for sample in samples
                if sample_value(sample, feature) not in (None, "")
            }
        )
        for value in values[:20]:
            condition = {"feature": feature, "op": "eq", "value": value}
            description = f"{feature} == {value}"
            definitions[_condition_id(condition)] = _make_stats(
                feature,
                condition,
                "block_combo",
                description,
            )

    special_conditions = [
        (
            "displacement_ratio",
            {"feature": "displacement_ratio", "op": "gt", "value": 2.0},
            "max_threshold",
            "displacement_ratio > 2.0: 추격 진입 의심",
        ),
        (
            "entry_quality_margin",
            {"feature": "entry_quality_margin", "op": "lte", "value": 0.0},
            "min_margin",
            "entry_quality_score가 임계값에 붙거나 미달: 경계값 통과 금지 후보",
        ),
        (
            "high_adx_reversal_chase",
            {
                "all": [
                    {"feature": "adx_entry", "op": "gte", "value": 45.0},
                    {
                        "any": [
                            {"feature": "entry_chased_extension", "op": "eq", "value": True},
                            {"feature": "entered_into_exhaustion", "op": "eq", "value": True},
                        ]
                    },
                ]
            },
            "block_combo",
            "고 ADX + 추격/소진 조합: 반전 타점으로 보기 어렵고 늦은 진입 가능성",
        ),
        (
            "chase_exhaustion_combo",
            {
                "all": [
                    {"feature": "entry_chased_extension", "op": "eq", "value": True},
                    {"feature": "entered_into_exhaustion", "op": "eq", "value": True},
                ]
            },
            "required_confirm",
            "추격 플래그와 소진 플래그가 동시에 켜진 진입은 추가 확인 요구 후보",
        ),
        (
            "lsr_unconfirmed_reclaim_chase",
            {
                "all": [
                    {"feature": "lsr_unconfirmed_reclaim", "op": "eq", "value": True},
                    {
                        "any": [
                            {"feature": "entry_chased_extension", "op": "eq", "value": True},
                            {"feature": "entered_into_exhaustion", "op": "eq", "value": True},
                            {"feature": "weak_reclaim_after_deep_sweep", "op": "eq", "value": True},
                        ]
                    },
                ]
            },
            "required_confirm",
            "LSR reclaim_only/tick_reclaim + 추격/약한 되돌림 조합은 sweep 단독 진입이므로 확인봉/retest 요구 후보",
        ),
        (
            "low_score_clean_reclaim_exception",
            {
                "all": [
                    {"feature": "entry_quality_margin", "op": "lte", "value": 0.03},
                    {"feature": "m5_align", "op": "gte", "value": 1.0},
                    {"feature": "displacement_ratio", "op": "lte", "value": 1.5},
                    {
                        "any": [
                            {"feature": "fee_adjusted_rr", "op": "gte", "value": 2.0},
                            {"feature": "sl_tp_geometry_enough_after_costs", "op": "eq", "value": True},
                        ]
                    },
                    {
                        "any": [
                            {"feature": "clean_reclaim", "op": "eq", "value": True},
                            {"feature": "reclaim_clean", "op": "eq", "value": True},
                            {"feature": "clean_reclaim_confirmed", "op": "eq", "value": True},
                        ]
                    },
                ]
            },
            "protect_exception",
            "낮은 점수/경계값이라도 clean reclaim + M5 정렬 + 낮은 displacement + 좋은 RR이면 수익 예외 보호 후보",
        ),
    ]
    for feature, condition, rule_type, description in special_conditions:
        definitions[_condition_id(condition)] = _make_stats(feature, condition, rule_type, description)

    return list(definitions.values())


def _rule_type_for(feature: str, op: str) -> str:
    if feature == "entry_quality_margin":
        return "min_margin"
    if op in {"gt", "gte"}:
        return "max_threshold"
    if op in {"lt", "lte"}:
        return "min_threshold"
    return "block_combo"


def aggregate_patterns(samples: Sequence[LearningSample]) -> List[Dict[str, Any]]:
    stats = pattern_definitions(samples)
    for pattern in stats:
        for sample in samples:
            if condition_matches(sample, pattern.condition):
                pattern.add(sample)
    summaries = [pattern.summary() for pattern in stats if pattern.sample_count > 0]
    summaries.sort(
        key=lambda item: (
            FEATURE_PRIORITY.get(item["feature"], 0),
            item["loss_count"],
            item["sample_count"],
            -(item["avg_r"] if item["avg_r"] is not None else 999.0),
        ),
        reverse=True,
    )
    return summaries


def overall_stats(samples: Sequence[LearningSample]) -> Dict[str, Any]:
    r_values = [sample.r for sample in samples if sample.r is not None]
    wins = sum(1 for sample in samples if sample.label == "win")
    losses = sum(1 for sample in samples if sample.label == "loss")
    flats = sum(1 for sample in samples if sample.label == "flat")
    positive = sum(value for value in r_values if value > 0)
    negative = sum(value for value in r_values if value < 0)
    return {
        "sample_count": len(samples),
        "win_count": wins,
        "loss_count": losses,
        "flat_count": flats,
        "avg_r": sum(r_values) / len(r_values) if r_values else None,
        "expectancy_r": sum(r_values) / len(r_values) if r_values else None,
        "profit_factor": positive / abs(negative) if negative < 0 else math.inf if positive > 0 else None,
        "missing_r_count": len(samples) - len(r_values),
        "review_only": True,
    }


def _first_float(sample: LearningSample, features: Sequence[str]) -> Optional[float]:
    for feature in features:
        value = _safe_float(sample_value(sample, feature))
        if value is not None:
            return value
    return None


def _volatility_regime(sample: LearningSample) -> str:
    value = _first_float(sample, VOLATILITY_REGIME_FEATURES)
    if value is None:
        return "VOL_MISSING"
    return _volatility_regime_label(value)


def _volatility_regime_label(value: float) -> str:
    if value < 0.9:
        return "LOW_VOL"
    if value <= 1.2:
        return "NORMAL_VOL"
    return "HIGH_VOL"


def _spread_thresholds(samples: Sequence[LearningSample]) -> Dict[str, Tuple[float, float]]:
    by_symbol: Dict[str, List[float]] = {}
    for sample in samples:
        spread = _first_float(sample, SPREAD_REGIME_FEATURES)
        if spread is None:
            continue
        symbol = str(sample_value(sample, "symbol") or "UNKNOWN").upper()
        by_symbol.setdefault(symbol, []).append(spread)
    thresholds: Dict[str, Tuple[float, float]] = {}
    for symbol, values in by_symbol.items():
        ordered = sorted(values)
        if len(ordered) < 3:
            continue
        low_idx = max(0, min(len(ordered) - 1, int(math.floor((len(ordered) - 1) * 0.33))))
        high_idx = max(0, min(len(ordered) - 1, int(math.floor((len(ordered) - 1) * 0.66))))
        thresholds[symbol] = (ordered[low_idx], ordered[high_idx])
    return thresholds


def _spread_regime(sample: LearningSample, thresholds: Dict[str, Tuple[float, float]]) -> str:
    spread = _first_float(sample, SPREAD_REGIME_FEATURES)
    if spread is None:
        return "SPREAD_MISSING"
    symbol = str(sample_value(sample, "symbol") or "UNKNOWN").upper()
    cuts = thresholds.get(symbol)
    if cuts is None:
        return "SPREAD_OBSERVED"
    low_cut, high_cut = cuts
    if spread <= low_cut:
        return "TIGHT_SPREAD"
    if spread <= high_cut:
        return "NORMAL_SPREAD"
    return "WIDE_SPREAD"


def _regime_data_quality(
    samples: Sequence[LearningSample],
    spread_thresholds: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        symbol = str(sample_value(sample, "symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        row = by_symbol.setdefault(
            symbol,
            {
                "symbol": symbol,
                "sample_count": 0,
                "r_value_count": 0,
                "volatility_value_count": 0,
                "spread_value_count": 0,
                "spread_missing_count": 0,
                "spread_values": [],
            },
        )
        row["sample_count"] += 1
        if sample.r is not None:
            row["r_value_count"] += 1
        if _first_float(sample, VOLATILITY_REGIME_FEATURES) is not None:
            row["volatility_value_count"] += 1
        spread = _first_float(sample, SPREAD_REGIME_FEATURES)
        if spread is None:
            row["spread_missing_count"] += 1
        else:
            row["spread_value_count"] += 1
            row["spread_values"].append(spread)

    rows: List[Dict[str, Any]] = []
    for symbol, row in sorted(by_symbol.items()):
        sample_count = int(row["sample_count"])
        spread_count = int(row["spread_value_count"])
        vol_count = int(row["volatility_value_count"])
        spread_values = sorted(float(value) for value in row.pop("spread_values"))
        if spread_count == 0:
            threshold_status = "missing"
        elif symbol in spread_thresholds:
            threshold_status = "quantile_thresholds"
        else:
            threshold_status = "observed_only_insufficient_symbol_samples"
        row.update(
            {
                "spread_coverage": spread_count / sample_count if sample_count else None,
                "volatility_coverage": vol_count / sample_count if sample_count else None,
                "spread_min": spread_values[0] if spread_values else None,
                "spread_max": spread_values[-1] if spread_values else None,
                "spread_threshold_status": threshold_status,
            }
        )
        rows.append(row)

    return {
        "by_symbol": rows,
        "spread_threshold_min_values_per_symbol": 3,
        "interpretation": (
            "레짐 조건부 기대값은 관찰값이 충분한 symbol만 tight/normal/wide로 나뉜다. "
            "coverage가 낮거나 observed_only bucket이면 전역 spread 필터 튜닝 근거로 쓰지 않는다."
        ),
    }


def _summarize_regime_group(
    key: str,
    label: str,
    grouped_samples: Sequence[LearningSample],
) -> Dict[str, Any]:
    r_values = [sample.r for sample in grouped_samples if sample.r is not None]
    positive = sum(value for value in r_values if value > 0)
    negative = sum(value for value in r_values if value < 0)
    spreads = [
        value
        for value in (_first_float(sample, SPREAD_REGIME_FEATURES) for sample in grouped_samples)
        if value is not None
    ]
    costs = [
        value
        for value in (_first_float(sample, COST_FEATURES) for sample in grouped_samples)
        if value is not None
    ]
    fee_rr = [
        value
        for value in (_safe_float(sample_value(sample, "fee_adjusted_rr")) for sample in grouped_samples)
        if value is not None
    ]
    return {
        "key": key,
        "label": label,
        "sample_count": len(grouped_samples),
        "win_count": sum(1 for sample in grouped_samples if sample.label == "win"),
        "loss_count": sum(1 for sample in grouped_samples if sample.label == "loss"),
        "flat_count": sum(1 for sample in grouped_samples if sample.label == "flat"),
        "expectancy_r": sum(r_values) / len(r_values) if r_values else None,
        "profit_factor": positive / abs(negative) if negative < 0 else math.inf if positive > 0 else None,
        "avg_spread": sum(spreads) / len(spreads) if spreads else None,
        "avg_cost_usd": sum(costs) / len(costs) if costs else None,
        "avg_fee_adjusted_rr": sum(fee_rr) / len(fee_rr) if fee_rr else None,
        "source_trade_ids": [sample.trade_id for sample in grouped_samples[:8]],
    }


def _group_by_regime(samples: Sequence[LearningSample], labels: Dict[str, str]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[LearningSample]] = {}
    for sample in samples:
        key = labels.get(sample.trade_id)
        if not key:
            continue
        buckets.setdefault(key, []).append(sample)
    rows = [_summarize_regime_group(key, key, values) for key, values in buckets.items()]
    rows.sort(
        key=lambda item: (
            item["sample_count"],
            -(item["expectancy_r"] if item["expectancy_r"] is not None else 999.0),
            item["key"],
        ),
        reverse=True,
    )
    return rows


def _setup_key(sample: LearningSample) -> str:
    symbol = str(sample_value(sample, "symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
    strategy = str(sample_value(sample, "strategy") or sample_value(sample, "entry_style") or "UNKNOWN").strip()
    return f"{symbol}|{strategy or 'UNKNOWN'}"


def _setup_regime_contrasts(
    samples: Sequence[LearningSample],
    combined_labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    by_setup_regime: Dict[str, Dict[str, List[LearningSample]]] = {}
    for sample in samples:
        regime = combined_labels.get(sample.trade_id)
        if not regime:
            continue
        by_setup_regime.setdefault(_setup_key(sample), {}).setdefault(regime, []).append(sample)

    contrasts: List[Dict[str, Any]] = []
    for setup, buckets in by_setup_regime.items():
        rows = [
            _summarize_regime_group(regime, regime, values)
            for regime, values in buckets.items()
        ]
        rows = [row for row in rows if row.get("expectancy_r") is not None]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda item: float(item["expectancy_r"]))
        worst = rows[0]
        best = rows[-1]
        contrasts.append(
            {
                "setup": setup,
                "regime_bucket_count": len(rows),
                "total_sample_count": sum(int(row.get("sample_count", 0) or 0) for row in rows),
                "best_regime": best["key"],
                "best_expectancy_r": best.get("expectancy_r"),
                "best_sample_count": best.get("sample_count"),
                "worst_regime": worst["key"],
                "worst_expectancy_r": worst.get("expectancy_r"),
                "worst_sample_count": worst.get("sample_count"),
                "expectancy_gap_r": float(best["expectancy_r"]) - float(worst["expectancy_r"]),
                "review_warning": (
                    "표본이 10건 미만이면 같은 setup의 레짐 차이도 관찰용이다."
                    if sum(int(row.get("sample_count", 0) or 0) for row in rows) < CANDIDATE_MIN
                    else None
                ),
            }
        )
    contrasts.sort(
        key=lambda item: (
            float(item.get("expectancy_gap_r") or 0.0),
            int(item.get("total_sample_count") or 0),
            item.get("setup") or "",
        ),
        reverse=True,
    )
    return contrasts


def _is_lsr_sample(sample: LearningSample) -> bool:
    strategy = str(sample_value(sample, "strategy") or "").strip().lower()
    entry_style = str(sample_value(sample, "entry_style") or "").strip().lower()
    return strategy.startswith("liquidity_sweep_reversal") or entry_style.startswith("liquidity_sweep_reversal")


def _confirmation_path(sample: LearningSample) -> str:
    path = str(sample_value(sample, "confirmation_path") or "").strip().lower()
    if path:
        return path
    if bool(sample_value(sample, "retest_confirmed")):
        return "retest"
    if bool(sample_value(sample, "lsr_unconfirmed_reclaim")):
        return "reclaim_only"
    return "unknown"


def _summarize_lsr_confirmation_group(key: str, grouped_samples: Sequence[LearningSample]) -> Dict[str, Any]:
    r_values = [sample.r for sample in grouped_samples if sample.r is not None]
    positive = sum(value for value in r_values if value > 0)
    negative = sum(value for value in r_values if value < 0)
    reclaim_distance_atr = [
        value
        for value in (_safe_float(sample_value(sample, "reclaim_distance_atr")) for sample in grouped_samples)
        if value is not None
    ]
    sweep_depth_atr = [
        value
        for value in (_safe_float(sample_value(sample, "sweep_depth_atr")) for sample in grouped_samples)
        if value is not None
    ]
    reclaim_depth_ratio = [
        value
        for value in (_safe_float(sample_value(sample, "reclaim_to_sweep_depth_ratio")) for sample in grouped_samples)
        if value is not None
    ]
    reclaim_window_elapsed_ratio = [
        value
        for value in (_safe_float(sample_value(sample, "reclaim_window_elapsed_ratio")) for sample in grouped_samples)
        if value is not None
    ]
    confirmation_scores = [
        value
        for value in (_safe_float(sample_value(sample, "lsr_confirmation_score")) for sample in grouped_samples)
        if value is not None
    ]
    unconfirmed_count = sum(1 for sample in grouped_samples if bool(sample_value(sample, "lsr_unconfirmed_reclaim")))
    weak_count = sum(1 for sample in grouped_samples if bool(sample_value(sample, "weak_reclaim_after_deep_sweep")))
    late_count = sum(1 for sample in grouped_samples if bool(sample_value(sample, "late_window_reclaim")))
    invalid_timing_count = sum(1 for sample in grouped_samples if bool(sample_value(sample, "invalid_reclaim_timing")))
    shallow_count = sum(1 for sample in grouped_samples if bool(sample_value(sample, "shallow_reclaim_confirmation")))
    chase_count = sum(1 for sample in grouped_samples if bool(sample_value(sample, "lsr_unconfirmed_reclaim_chase")))
    retest_count = sum(1 for sample in grouped_samples if bool(sample_value(sample, "retest_confirmed")))
    return {
        "key": key,
        "sample_count": len(grouped_samples),
        "win_count": sum(1 for sample in grouped_samples if sample.label == "win"),
        "loss_count": sum(1 for sample in grouped_samples if sample.label == "loss"),
        "flat_count": sum(1 for sample in grouped_samples if sample.label == "flat"),
        "expectancy_r": sum(r_values) / len(r_values) if r_values else None,
        "profit_factor": positive / abs(negative) if negative < 0 else math.inf if positive > 0 else None,
        "unconfirmed_reclaim_count": unconfirmed_count,
        "weak_reclaim_after_deep_sweep_count": weak_count,
        "shallow_reclaim_confirmation_count": shallow_count,
        "late_window_reclaim_count": late_count,
        "invalid_reclaim_timing_count": invalid_timing_count,
        "unconfirmed_reclaim_chase_count": chase_count,
        "retest_confirmed_count": retest_count,
        "avg_reclaim_distance_atr": _mean(reclaim_distance_atr),
        "avg_sweep_depth_atr": _mean(sweep_depth_atr),
        "avg_reclaim_to_sweep_depth_ratio": _mean(reclaim_depth_ratio),
        "avg_reclaim_window_elapsed_ratio": _mean(reclaim_window_elapsed_ratio),
        "avg_lsr_confirmation_score": _mean(confirmation_scores),
        "source_trade_ids": [sample.trade_id for sample in grouped_samples[:8]],
    }


def lsr_confirmation_expectancy_summary(samples: Sequence[LearningSample]) -> Dict[str, Any]:
    lsr_samples = [sample for sample in samples if _is_lsr_sample(sample)]
    buckets: Dict[str, List[LearningSample]] = {}
    for sample in lsr_samples:
        buckets.setdefault(_confirmation_path(sample), []).append(sample)
    by_path = [_summarize_lsr_confirmation_group(key, values) for key, values in buckets.items()]
    by_path.sort(
        key=lambda item: (
            item["sample_count"],
            -(item["expectancy_r"] if item["expectancy_r"] is not None else 999.0),
            item["key"],
        ),
        reverse=True,
    )

    unconfirmed = [
        sample
        for sample in lsr_samples
        if _confirmation_path(sample) in {"reclaim_only", "tick_reclaim"}
        and not bool(sample_value(sample, "retest_confirmed"))
    ]
    clean = [sample for sample in lsr_samples if bool(sample_value(sample, "retest_confirmed"))]
    unknown_confirmation = [sample for sample in lsr_samples if _confirmation_path(sample) == "unknown"]
    reclaim_metric_complete = [
        sample
        for sample in lsr_samples
        if _safe_float(sample_value(sample, "reclaim_distance_atr")) is not None
        and _safe_float(sample_value(sample, "sweep_depth_atr")) is not None
        and _safe_float(sample_value(sample, "reclaim_to_sweep_depth_ratio")) is not None
    ]
    reclaim_timing_complete = [
        sample
        for sample in lsr_samples
        if _lsr_reclaim_age_sec(sample.features) is not None
        and _safe_float(sample_value(sample, "reclaim_window_sec")) is not None
        and _safe_float(sample_value(sample, "reclaim_window_elapsed_ratio")) is not None
    ]
    late_window = [sample for sample in lsr_samples if bool(sample_value(sample, "late_window_reclaim"))]
    invalid_timing = [sample for sample in lsr_samples if bool(sample_value(sample, "invalid_reclaim_timing"))]
    shallow_reclaim = [sample for sample in lsr_samples if bool(sample_value(sample, "shallow_reclaim_confirmation"))]
    confirmation_score_complete = [
        sample for sample in lsr_samples if _safe_float(sample_value(sample, "lsr_confirmation_score")) is not None
    ]
    warnings: List[str] = []
    if not lsr_samples:
        warnings.append("LSR 표본이 없어 confirmation path별 기대값을 계산하지 못했다.")
    if len(lsr_samples) < CANDIDATE_MIN:
        warnings.append("LSR 표본이 10건 미만이면 confirmation path 차이는 관찰용이다. live gate 근거로 쓰면 안 된다.")
    if unknown_confirmation:
        warnings.append(
            "confirmation_path/flag가 없는 LSR 표본은 unknown으로 분리했다. "
            "누락 데이터를 reclaim_only 근거로 섞어 튜닝하면 안 된다."
        )
    if invalid_timing:
        warnings.append(
            "sweep→entry 시간이 음수인 LSR 표본은 broker/server time 또는 sweep_event_key 파싱 문제로 보고 "
            "reclaim 속도/late-window 튜닝 근거에서 제외해야 한다."
        )
    if unconfirmed and not clean:
        warnings.append("retest/확인봉 표본이 없어 reclaim_only와 clean reclaim을 비교하지 못했다.")
    elif clean and not unconfirmed:
        warnings.append("unconfirmed reclaim 표본이 없어 단일 sweep 추격 손실 패턴을 비교하지 못했다.")

    return {
        "review_only": True,
        "sample_count": len(lsr_samples),
        "path_value_count": sum(1 for sample in lsr_samples if _confirmation_path(sample) != "unknown"),
        "unconfirmed_reclaim_count": len(unconfirmed),
        "retest_confirmed_count": len(clean),
        "unknown_confirmation_count": len(unknown_confirmation),
        "reclaim_metric_complete_count": len(reclaim_metric_complete),
        "reclaim_timing_complete_count": len(reclaim_timing_complete),
        "late_window_reclaim_count": len(late_window),
        "invalid_reclaim_timing_count": len(invalid_timing),
        "shallow_reclaim_confirmation_count": len(shallow_reclaim),
        "confirmation_score_complete_count": len(confirmation_score_complete),
        "dimensions": {
            "by_confirmation_path": by_path,
            "unconfirmed_reclaim": _summarize_lsr_confirmation_group("UNCONFIRMED_RECLAIM", unconfirmed)
            if unconfirmed
            else None,
            "retest_confirmed": _summarize_lsr_confirmation_group("RETEST_CONFIRMED", clean)
            if clean
            else None,
            "unknown_confirmation": _summarize_lsr_confirmation_group("UNKNOWN_CONFIRMATION", unknown_confirmation)
            if unknown_confirmation
            else None,
            "metadata_quality": {
                "path_missing_count": len(unknown_confirmation),
                "path_present_count": sum(1 for sample in lsr_samples if _confirmation_path(sample) != "unknown"),
                "reclaim_metric_complete_count": len(reclaim_metric_complete),
                "reclaim_metric_missing_count": len(lsr_samples) - len(reclaim_metric_complete),
                "reclaim_timing_complete_count": len(reclaim_timing_complete),
                "reclaim_timing_missing_count": len(lsr_samples) - len(reclaim_timing_complete),
                "invalid_reclaim_timing_count": len(invalid_timing),
                "confirmation_score_complete_count": len(confirmation_score_complete),
                "confirmation_score_missing_count": len(lsr_samples) - len(confirmation_score_complete),
            },
        },
        "warnings": warnings,
        "interpretation": (
            "LSR은 sweep 발생보다 reclaim 확인 품질이 핵심이다. "
            "reclaim_only/tick_reclaim bucket의 기대값, 약한 되돌림, late reclaim, 추격 플래그를 "
            "retest/확인봉 bucket과 비교한 뒤에만 확인 요구 후보를 검토한다."
        ),
    }


def regime_expectancy_summary(samples: Sequence[LearningSample]) -> Dict[str, Any]:
    spread_thresholds = _spread_thresholds(samples)
    data_quality = _regime_data_quality(samples, spread_thresholds)
    vol_labels = {sample.trade_id: _volatility_regime(sample) for sample in samples}
    spread_labels = {sample.trade_id: _spread_regime(sample, spread_thresholds) for sample in samples}
    combined_labels = {
        sample.trade_id: f"{vol_labels[sample.trade_id]}|{spread_labels[sample.trade_id]}"
        for sample in samples
    }
    setup_combined_labels = {
        sample.trade_id: f"{_setup_key(sample)}|{combined_labels[sample.trade_id]}"
        for sample in samples
    }
    warnings: List[str] = []
    if not any(label != "VOL_MISSING" for label in vol_labels.values()):
        warnings.append("atr_regime_ratio/volatility feature가 없어 변동성 레짐별 기대값을 계산하지 못했다.")
    if not any(label != "SPREAD_MISSING" for label in spread_labels.values()):
        warnings.append("spread_points/current_spread feature가 없어 스프레드 레짐별 기대값을 계산하지 못했다.")
    for row in data_quality["by_symbol"]:
        symbol = row["symbol"]
        sample_count = int(row["sample_count"])
        spread_count = int(row["spread_value_count"])
        vol_count = int(row["volatility_value_count"])
        if 0 < spread_count < sample_count:
            warnings.append(
                f"{symbol}: spread 값이 {spread_count}/{sample_count}건뿐이다. "
                "누락 표본이 섞인 레짐 기대값으로 spread 필터를 튜닝하지 않는다."
            )
        if row["spread_threshold_status"] == "observed_only_insufficient_symbol_samples":
            warnings.append(
                f"{symbol}: spread 관찰값이 3건 미만이라 tight/normal/wide 절단값을 만들지 않고 "
                "SPREAD_OBSERVED로만 집계한다."
            )
        if 0 < vol_count < sample_count:
            warnings.append(
                f"{symbol}: volatility 값이 {vol_count}/{sample_count}건뿐이다. "
                "누락 표본이 섞인 레짐 기대값으로 ATR 레짐 필터를 튜닝하지 않는다."
            )
    if len(samples) < CANDIDATE_MIN:
        warnings.append("표본이 10건 미만이면 레짐별 기대값 차이는 관찰용이다. 튜닝 근거로 쓰면 안 된다.")
    return {
        "review_only": True,
        "sample_count": len(samples),
        "spread_value_count": sum(1 for sample in samples if _first_float(sample, SPREAD_REGIME_FEATURES) is not None),
        "volatility_value_count": sum(1 for sample in samples if _first_float(sample, VOLATILITY_REGIME_FEATURES) is not None),
        "spread_thresholds_by_symbol": {
            symbol: {"tight_cut": cuts[0], "wide_cut": cuts[1]} for symbol, cuts in spread_thresholds.items()
        },
        "data_quality": data_quality,
        "dimensions": {
            "volatility_regime": _group_by_regime(samples, vol_labels),
            "spread_regime": _group_by_regime(samples, spread_labels),
            "volatility_x_spread": _group_by_regime(samples, combined_labels),
            "setup_x_volatility_spread": _group_by_regime(samples, setup_combined_labels),
        },
        "setup_regime_contrasts": _setup_regime_contrasts(samples, combined_labels),
        "warnings": warnings,
    }


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [value for value in values if value is not None]
    return sum(items) / len(items) if items else None


def _sum_values(values: Iterable[float]) -> float:
    return sum(value for value in values if value is not None)


def _execution_sample_metrics(sample: LearningSample) -> Dict[str, Optional[float]]:
    price_r = _safe_float(sample_value(sample, "price_r_multiple"))
    net_r = _safe_float(sample_value(sample, "pnl_r_multiple"))
    if net_r is None:
        net_r = sample.r
    drag_r = _safe_float(sample_value(sample, "net_execution_drag_r"))
    if drag_r is None and price_r is not None and net_r is not None:
        drag_r = price_r - net_r
    explicit_cost_r = _safe_float(sample_value(sample, "realized_explicit_cost_r"))
    estimated_cost_r = _safe_float(sample_value(sample, "estimated_cost_to_expected_loss_r"))
    expected_loss = _safe_float(sample_value(sample, "expected_net_loss_usd"))
    if expected_loss is None:
        expected_loss = _safe_float(sample_value(sample, "estimated_net_loss_usd"))
    if expected_loss is not None and expected_loss > 0:
        if estimated_cost_r is None:
            estimated_cost_usd = _safe_float(sample_value(sample, "estimated_cost_usd"))
            if estimated_cost_usd is not None:
                estimated_cost_r = estimated_cost_usd / expected_loss
        if explicit_cost_r is None:
            explicit_cost_usd = _safe_float(sample_value(sample, "realized_explicit_cost_usd"))
            if explicit_cost_usd is not None:
                explicit_cost_r = explicit_cost_usd / expected_loss
    return {
        "price_r": price_r,
        "net_r": net_r,
        "drag_r": drag_r,
        "entry_shortfall_r": _safe_float(sample_value(sample, "entry_implementation_shortfall_r")),
        "estimated_cost_r": estimated_cost_r,
        "explicit_cost_r": explicit_cost_r,
        "spread": _first_float(sample, SPREAD_REGIME_FEATURES),
    }


def _summarize_execution_group(key: str, grouped_samples: Sequence[LearningSample]) -> Dict[str, Any]:
    metrics = [_execution_sample_metrics(sample) for sample in grouped_samples]
    paired = [
        item
        for item in metrics
        if item["price_r"] is not None and item["net_r"] is not None
    ]
    drags = [item["drag_r"] for item in metrics if item["drag_r"] is not None]
    adverse_shortfalls = [
        item["entry_shortfall_r"]
        for item in metrics
        if item["entry_shortfall_r"] is not None and item["entry_shortfall_r"] > 0
    ]
    signal_positive_net_negative = [
        item
        for item in paired
        if item["price_r"] is not None
        and item["net_r"] is not None
        and item["price_r"] > 0
        and item["net_r"] < 0
    ]
    signal_positive_net_nonpositive = [
        item
        for item in paired
        if item["price_r"] is not None
        and item["net_r"] is not None
        and item["price_r"] > 0
        and item["net_r"] <= 0
    ]
    signal_positive = [
        item
        for item in paired
        if item["price_r"] is not None and item["price_r"] > 0
    ]
    drag_sum = _sum_values(drag for drag in drags if drag is not None)
    avg_signal_price_r = _mean(item["price_r"] for item in metrics if item["price_r"] is not None)
    avg_realized_net_r = _mean(item["net_r"] for item in metrics if item["net_r"] is not None)
    avg_execution_drag_r = _mean(drag for drag in drags if drag is not None)
    execution_drag_to_signal_ratio = None
    net_realization_ratio = None
    if (
        avg_signal_price_r is not None
        and avg_signal_price_r > 0
        and avg_execution_drag_r is not None
        and avg_execution_drag_r > 0
    ):
        execution_drag_to_signal_ratio = avg_execution_drag_r / avg_signal_price_r
    if avg_signal_price_r is not None and avg_signal_price_r > 0 and avg_realized_net_r is not None:
        net_realization_ratio = avg_realized_net_r / avg_signal_price_r
    return {
        "key": key,
        "sample_count": len(grouped_samples),
        "price_r_value_count": sum(1 for item in metrics if item["price_r"] is not None),
        "net_r_value_count": sum(1 for item in metrics if item["net_r"] is not None),
        "paired_r_value_count": len(paired),
        "avg_signal_price_r": avg_signal_price_r,
        "avg_realized_net_r": avg_realized_net_r,
        "avg_execution_drag_r": avg_execution_drag_r,
        "execution_drag_to_signal_ratio": execution_drag_to_signal_ratio,
        "net_realization_ratio": net_realization_ratio,
        "total_execution_drag_r": drag_sum,
        "avg_entry_implementation_shortfall_r": _mean(
            item["entry_shortfall_r"] for item in metrics if item["entry_shortfall_r"] is not None
        ),
        "adverse_entry_shortfall_count": len(adverse_shortfalls),
        "avg_estimated_cost_to_expected_loss_r": _mean(
            item["estimated_cost_r"] for item in metrics if item["estimated_cost_r"] is not None
        ),
        "avg_realized_explicit_cost_r": _mean(
            item["explicit_cost_r"] for item in metrics if item["explicit_cost_r"] is not None
        ),
        "avg_spread": _mean(item["spread"] for item in metrics if item["spread"] is not None),
        "signal_positive_trade_count": len(signal_positive),
        "signal_positive_net_negative_count": len(signal_positive_net_negative),
        "signal_positive_net_nonpositive_count": len(signal_positive_net_nonpositive),
        "signal_positive_net_nonpositive_rate": (
            len(signal_positive_net_nonpositive) / len(signal_positive) if signal_positive else None
        ),
        "source_trade_ids": [sample.trade_id for sample in grouped_samples[:8]],
    }


def _execution_tuning_gate(overall: Dict[str, Any]) -> Dict[str, Any]:
    paired_count = int(overall.get("paired_r_value_count") or 0)
    avg_signal_r = overall.get("avg_signal_price_r")
    avg_net_r = overall.get("avg_realized_net_r")
    drag_ratio = overall.get("execution_drag_to_signal_ratio")
    net_realization_ratio = overall.get("net_realization_ratio")
    inversion_rate = overall.get("signal_positive_net_nonpositive_rate")
    reasons: List[str] = []

    if paired_count == 0:
        reasons.append("missing_paired_signal_and_net_r")
    elif paired_count < CANDIDATE_MIN:
        reasons.append("insufficient_paired_samples")
    if (
        isinstance(avg_signal_r, (int, float))
        and isinstance(avg_net_r, (int, float))
        and avg_signal_r > 0
        and avg_net_r <= 0
    ):
        reasons.append("positive_signal_erased_after_costs")
    if isinstance(drag_ratio, (int, float)) and drag_ratio >= 0.50:
        reasons.append("execution_drag_erodes_signal_edge")
    if isinstance(inversion_rate, (int, float)) and inversion_rate >= 0.25:
        reasons.append("frequent_positive_signal_net_nonpositive_trades")

    blocked = bool(reasons)
    return {
        "review_only": True,
        "status": "blocked" if blocked else "pass",
        "blocks_signal_threshold_tuning": blocked,
        "reason_codes": reasons,
        "paired_r_value_count": paired_count,
        "min_paired_samples_for_tuning": CANDIDATE_MIN,
        "signal_positive_net_nonpositive_rate": inversion_rate,
        "execution_drag_to_signal_ratio": drag_ratio,
        "net_realization_ratio": net_realization_ratio,
        "recommendation_korean": (
            "전략 파라미터/entry threshold 튜닝을 보류하고 symbol/strategy별 spread, slippage, fee 원인부터 분해한다."
            if blocked
            else "순체결 기대값 기준의 최소 TCA 게이트는 통과했지만 live 변경은 별도 shadow/OOS 검증이 필요하다."
        ),
    }


def _group_execution_samples(
    samples: Sequence[LearningSample],
    label_for_sample: Callable[[LearningSample], Any],
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[LearningSample]] = {}
    for sample in samples:
        key = str(label_for_sample(sample) or "UNKNOWN")
        buckets.setdefault(key, []).append(sample)
    rows = [_summarize_execution_group(key, values) for key, values in buckets.items()]
    rows.sort(
        key=lambda item: (
            item["paired_r_value_count"],
            item["sample_count"],
            item["total_execution_drag_r"],
            item["key"],
        ),
        reverse=True,
    )
    return rows


def execution_shortfall_summary(samples: Sequence[LearningSample]) -> Dict[str, Any]:
    spread_thresholds = _spread_thresholds(samples)
    overall = _summarize_execution_group("ALL", samples)
    warnings: List[str] = []
    if overall["paired_r_value_count"] == 0:
        warnings.append("price_r_multiple와 pnl_r_multiple 쌍이 없어 신호 기대값과 순체결 기대값을 분리 비교하지 못했다.")
    if overall["price_r_value_count"] < len(samples):
        warnings.append("일부 표본에 price_r_multiple이 없어 전략 신호 자체의 가격 R을 온전히 집계하지 못했다.")
    if overall["net_r_value_count"] < len(samples):
        warnings.append("일부 표본에 pnl_r_multiple/net R이 없어 체결 후 순기대값 집계가 불완전하다.")
    if overall["sample_count"] < CANDIDATE_MIN:
        warnings.append("표본이 10건 미만이면 execution drag 결론은 관찰용이다. 튜닝 근거로 쓰면 안 된다.")
    if (
        isinstance(overall.get("avg_signal_price_r"), (int, float))
        and isinstance(overall.get("avg_realized_net_r"), (int, float))
        and overall["avg_signal_price_r"] > 0
        and overall["avg_realized_net_r"] <= 0
    ):
        warnings.append("신호 가격 R은 양수인데 순체결 R은 0 이하이다. 타점 튜닝보다 spread/slippage/fee 원인 분리가 먼저다.")
    if isinstance(overall.get("execution_drag_to_signal_ratio"), (int, float)) and overall["execution_drag_to_signal_ratio"] >= 0.50:
        warnings.append("execution drag가 신호 가격 R의 50% 이상을 잠식한다. net 기대값이 아직 양수여도 타점 튜닝보다 체결 비용 분해가 먼저다.")

    return {
        "review_only": True,
        "sample_count": len(samples),
        "overall": overall,
        "tuning_gate": _execution_tuning_gate(overall),
        "groups": {
            "by_symbol": _group_execution_samples(
                samples,
                lambda sample: str(sample_value(sample, "symbol") or "UNKNOWN").upper(),
            ),
            "by_strategy": _group_execution_samples(
                samples,
                lambda sample: sample_value(sample, "strategy") or "UNKNOWN",
            ),
            "by_symbol_strategy": _group_execution_samples(
                samples,
                lambda sample: (
                    f"{str(sample_value(sample, 'symbol') or 'UNKNOWN').upper()}|"
                    f"{sample_value(sample, 'strategy') or 'UNKNOWN'}"
                ),
            ),
            "by_spread_regime": _group_execution_samples(
                samples,
                lambda sample: _spread_regime(sample, spread_thresholds),
            ),
        },
        "warnings": warnings,
        "interpretation": (
            "avg_signal_price_r는 진입/청산 가격만 본 전략 신호 성과이고, "
            "avg_realized_net_r는 spread/slippage/fee 이후 순성과다. "
            "두 값의 차이와 entry_implementation_shortfall_r를 먼저 본 뒤 파라미터를 튜닝한다."
        ),
    }


def evidence_grade(sample_count: int, net_r_delta: Optional[float] = None) -> str:
    if sample_count < SUSPICION_MIN:
        return "observation_only"
    if sample_count < CANDIDATE_MIN:
        return "suspicion"
    if sample_count < LIVE_REVIEW_MIN:
        return "candidate_rule"
    if net_r_delta is not None and net_r_delta > 0:
        return "possible_live_review_candidate"
    return "candidate_rule_needs_positive_shadow"


def _action_for_pattern(pattern: Dict[str, Any]) -> str:
    feature = str(pattern.get("feature") or "")
    rule_type = str(pattern.get("rule_type") or "")
    avg_r = pattern.get("avg_r")
    win_count = int(pattern.get("win_count") or 0)
    loss_count = int(pattern.get("loss_count") or 0)

    if win_count > loss_count and isinstance(avg_r, (int, float)) and avg_r > 0:
        if feature in {"entry_quality_margin", "entry_quality_score", "low_score_clean_reclaim_exception"}:
            return "relax_threshold_candidate"
        return "allow_exception_candidate"
    if rule_type == "protect_exception":
        return "allow_exception_candidate"
    if rule_type == "required_confirm":
        return "require_confirmation"
    if feature in {"time_from_sweep_to_entry_sec", "late_window_reclaim"} or "chase" in feature:
        return "delay_entry"
    if feature in {"expected_net_loss_usd", "risk_per_unit", "adverse_excursion_price"}:
        return "reduce_size"
    if rule_type == "block_combo" or feature.endswith("_combo") or feature == "high_adx_reversal_chase":
        return "tighten_only_when_combo"
    if loss_count <= 0:
        return "no_change"
    return "block"


def _is_relaxation_action(action: str) -> bool:
    return action in RELAXATION_ACTIONS


def _is_risk_reduction_action(action: str) -> bool:
    return action in RISK_REDUCTION_ACTIONS


def _expected_benefit_for(pattern: Dict[str, Any], action: str) -> Dict[str, Any]:
    if _is_relaxation_action(action):
        return {
            "kind": "recover_or_protect_missed_winners",
            "win_count": pattern.get("win_count", 0),
            "avg_r": pattern.get("avg_r"),
            "description": "Profitable/near-blocked pattern worth protecting from over-strict filters.",
        }
    if action == "no_change":
        return {
            "kind": "observe_only",
            "description": "Evidence does not justify changing strictness.",
        }
    return {
        "kind": "risk_reduction",
        "loss_count": pattern.get("loss_count", 0),
        "avg_r": pattern.get("avg_r"),
        "description": "Reduce exposure to a repeated negative-expectancy pattern while measuring missed winners.",
    }


def _opportunity_cost_for(pattern: Dict[str, Any], action: str) -> Dict[str, Any]:
    if _is_relaxation_action(action):
        return {
            "possible_extra_losses_count": pattern.get("loss_count", 0),
            "risk_leak_rate": (
                pattern["loss_count"] / pattern["sample_count"] if pattern.get("sample_count") else None
            ),
            "requires_shadow_confirmation": True,
        }
    return {
        "blocked_wins_count": pattern.get("win_count", 0),
        "missed_profit_r": None,
        "false_block_rate": (
            pattern["win_count"] / pattern["sample_count"] if pattern.get("sample_count") else None
        ),
        "requires_shadow_confirmation": True,
    }


def _candidate_id(feature: str, condition: Dict[str, Any]) -> str:
    seed = f"{feature}|{_json_dumps(condition)}"
    return "rc_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:14]


def _rationale(pattern: Dict[str, Any], grade: str) -> str:
    loss_rate = pattern.get("loss_rate")
    avg_r = pattern.get("avg_r")
    loss_rate_text = f"{loss_rate:.0%}" if isinstance(loss_rate, (int, float)) else "N/A"
    avg_r_text = f"{avg_r:.2f}R" if isinstance(avg_r, (int, float)) else "N/A"
    prefix = {
        "observation_only": "아직 1건 수준 관찰이다. 자동 강화 금지.",
        "suspicion": "반복 손실 의심 구간이다. 더 모아야 하지만 냄새가 좋지 않다.",
        "candidate_rule": "10건 이상 반복되어 검토 가능한 후보 규칙이다.",
        "possible_live_review_candidate": "샘플과 shadow 결과가 충분해 라이브 검토 후보가 될 수 있다.",
        "candidate_rule_needs_positive_shadow": "샘플은 충분하지만 shadow/replay 이득 확인 전까지 보류다.",
    }.get(grade, "검토 전용 후보 규칙이다.")
    return (
        f"{prefix} 조건 `{pattern['description_korean']}`에서 "
        f"{pattern['sample_count']}건 중 손실 {pattern['loss_count']}건, "
        f"승률 반대 손실률 {loss_rate_text}, 평균 {avg_r_text}. "
        "작은 표본으로 live gate를 바꾸면 안 된다."
    )


def generate_candidates(
    patterns: Sequence[Dict[str, Any]],
    *,
    created_at: Optional[str] = None,
    min_samples_to_emit: int = CANDIDATE_MIN,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    selected: List[Dict[str, Any]] = []
    for pattern in patterns:
        if pattern["sample_count"] < max(OBSERVATION_MIN, min_samples_to_emit):
            continue
        avg_r = pattern.get("avg_r")
        loss_rate = pattern.get("loss_rate") or 0.0
        win_count = int(pattern.get("win_count") or 0)
        loss_count = int(pattern.get("loss_count") or 0)
        is_profitable_pattern = (
            win_count > 0
            and win_count >= max(1, loss_count)
            and isinstance(avg_r, (int, float))
            and avg_r > 0
        )
        is_loss_pattern = loss_count > 0 and not (avg_r is not None and avg_r > 0 and loss_rate < 0.5)
        if not is_loss_pattern and not is_profitable_pattern:
            continue
        action = _action_for_pattern(pattern)
        if action == "no_change":
            continue
        blocked = {
            "sample_count": pattern["sample_count"],
            "loss_count": pattern["loss_count"],
            "win_count": pattern["win_count"],
            "avg_r": avg_r,
            "avoided_loss_r": None,
            "sacrificed_win_r": None,
        }
        grade = evidence_grade(pattern["sample_count"])
        candidate = {
            "candidate_id": _candidate_id(pattern["feature"], pattern["condition"]),
            "created_at": created_at,
            "status": "review_only",
            "candidate_action": action,
            "rule_type": pattern["rule_type"],
            "feature": pattern["feature"],
            "condition": pattern["condition"],
            "rationale_korean": _rationale(pattern, grade),
            "evidence": blocked,
            "expected_benefit": _expected_benefit_for(pattern, action),
            "opportunity_cost": _opportunity_cost_for(pattern, action),
            "live_review_ready": False,
            "safety": {
                "evidence_grade": grade,
                "min_samples_required": CANDIDATE_MIN,
                "live_review_min_samples": LIVE_REVIEW_MIN,
                "preferred_live_review_samples": 50,
                "live_apply_allowed": False,
                "needs_shadow_validation": True,
                "never_auto_mutate_live_gates": True,
            },
            "source_trade_ids": pattern["source_trade_ids"],
            "examples": pattern["source_trade_ids"][:3],
        }
        selected.append(candidate)
    selected.sort(
        key=lambda item: (
            1 if _is_relaxation_action(item.get("candidate_action", "")) else 0,
            FEATURE_PRIORITY.get(item["feature"], 0),
            item["evidence"]["loss_count"],
            item["evidence"]["win_count"],
            item["evidence"]["sample_count"],
            -(item["evidence"]["avg_r"] if item["evidence"]["avg_r"] is not None else 999.0),
        ),
        reverse=True,
    )
    return selected[:limit]


def shadow_evaluate(
    candidate: Dict[str, Any],
    samples: Sequence[LearningSample],
    *,
    max_false_block_rate: float = DEFAULT_MAX_FALSE_BLOCK_RATE,
    max_trade_frequency_drop: float = DEFAULT_MAX_TRADE_FREQUENCY_DROP,
    min_live_review_samples: int = LIVE_REVIEW_MIN,
) -> Dict[str, Any]:
    condition = candidate.get("condition") or {}
    action = str(candidate.get("candidate_action") or "block")
    matched = [sample for sample in samples if condition_matches(sample, condition)]
    blocked_losses = 0
    blocked_wins = 0
    blocked_flats = 0
    missing_r = 0
    avoided_loss_r = 0.0
    sacrificed_win_r = 0.0
    blocked_r_values: List[float] = []
    for sample in matched:
        if sample.label == "loss":
            blocked_losses += 1
        elif sample.label == "win":
            blocked_wins += 1
        else:
            blocked_flats += 1
        if sample.r is None:
            missing_r += 1
            continue
        blocked_r_values.append(sample.r)
        if sample.r < 0:
            avoided_loss_r += abs(sample.r)
        elif sample.r > 0:
            sacrificed_win_r += sample.r
    action_is_relaxation = _is_relaxation_action(action)
    if not blocked_r_values:
        net_r_delta = None
    elif action_is_relaxation:
        net_r_delta = sum(blocked_r_values)
    else:
        net_r_delta = -sum(blocked_r_values)

    total_samples = len(samples)
    blocked_total = len(matched)
    false_block_rate = blocked_wins / blocked_total if blocked_total else 0.0
    missed_profit_r = sacrificed_win_r
    trade_frequency_delta = -(blocked_total / total_samples) if total_samples and not action_is_relaxation else 0.0
    acceptance_rate_before = 1.0 if total_samples else None
    acceptance_rate_after = (
        (total_samples - blocked_total) / total_samples if total_samples and not action_is_relaxation else acceptance_rate_before
    )
    risk_leak_rate = blocked_losses / blocked_total if blocked_total else 0.0
    warnings: List[str] = []
    if len(matched) < CANDIDATE_MIN:
        warnings.append("표본이 10건 미만이다. 이 shadow 결과는 관찰/의심 단계일 뿐 live gate 근거가 아니다.")
    if missing_r:
        warnings.append("일부 거래에 R 값이 없어 net_r_delta가 불완전하다.")
    if not action_is_relaxation and false_block_rate > max_false_block_rate:
        warnings.append("승자 차단 비율이 높다. 이 후보는 봇을 과보수화할 수 있다.")
    if not action_is_relaxation and total_samples and abs(trade_frequency_delta) > max_trade_frequency_drop:
        warnings.append("거래 빈도 감소 추정치가 너무 크다. trade starvation 위험이 있다.")
    if not action_is_relaxation and sacrificed_win_r > avoided_loss_r:
        warnings.append("놓치는 수익 R이 피하는 손실 R보다 크다. naive block은 기각해야 한다.")
    if action_is_relaxation and risk_leak_rate > max_false_block_rate:
        warnings.append("예외/완화 후보가 손실도 많이 포함한다. 더 좁은 확인 조건이 필요하다.")

    if action_is_relaxation:
        live_review_ready = (
            blocked_total >= min_live_review_samples
            and net_r_delta is not None
            and net_r_delta > 0
            and risk_leak_rate <= max_false_block_rate
        )
    else:
        live_review_ready = (
            blocked_total >= min_live_review_samples
            and net_r_delta is not None
            and net_r_delta > 0
            and false_block_rate <= max_false_block_rate
            and (not total_samples or abs(trade_frequency_delta) <= max_trade_frequency_drop)
        )
    over_filtering_warning = "; ".join(
        warning
        for warning in warnings
        if "과보수" in warning or "starvation" in warning or "놓치는 수익" in warning
    ) or None
    promotion_block_reasons: List[str] = []
    if blocked_total < min_live_review_samples:
        promotion_block_reasons.append("insufficient_shadow_samples")
    if net_r_delta is None:
        promotion_block_reasons.append("missing_net_r_delta")
    elif net_r_delta <= 0:
        promotion_block_reasons.append("non_positive_net_r_delta")
    if action_is_relaxation:
        if risk_leak_rate > max_false_block_rate:
            promotion_block_reasons.append("risk_leak_rate_above_limit")
    else:
        if false_block_rate > max_false_block_rate:
            promotion_block_reasons.append("false_block_rate_above_limit")
        if total_samples and abs(trade_frequency_delta) > max_trade_frequency_drop:
            promotion_block_reasons.append("trade_frequency_drop_above_limit")
    promotion_gate = {
        "status": "pass" if live_review_ready else "blocked",
        "required_shadow_metrics": REQUIRED_SHADOW_METRICS,
        "metrics_present": {
            "blocked_total": True,
            "net_r_delta": net_r_delta is not None,
            "false_block_rate": True,
            "trade_frequency_delta": True,
        },
        "block_reasons": promotion_block_reasons,
        "thresholds": {
            "min_live_review_samples": min_live_review_samples,
            "max_false_block_rate": max_false_block_rate,
            "max_trade_frequency_drop": max_trade_frequency_drop,
        },
    }
    return {
        "candidate_id": candidate.get("candidate_id"),
        "status": "review_only",
        "candidate_action": action,
        "evaluation_mode": "opportunity_recovery" if action_is_relaxation else "risk_reduction_shadow_block",
        "blocked_losses_count": blocked_losses,
        "blocked_wins_count": blocked_wins,
        "blocked_losses": blocked_losses,
        "blocked_wins": blocked_wins,
        "blocked_flats": blocked_flats,
        "blocked_total": blocked_total,
        "net_r_delta": net_r_delta,
        "trade_frequency_delta": trade_frequency_delta,
        "trade_count_delta": -blocked_total if not action_is_relaxation else 0,
        "avoided_loss_r": avoided_loss_r,
        "sacrificed_win_r": sacrificed_win_r,
        "missed_profit_r": missed_profit_r,
        "false_block_rate": false_block_rate,
        "acceptance_rate_before": acceptance_rate_before,
        "acceptance_rate_after": acceptance_rate_after,
        "risk_leak_rate": risk_leak_rate,
        "over_filtering_warning": over_filtering_warning,
        "live_review_ready": live_review_ready,
        "promotion_gate": promotion_gate,
        "missing_r_count": missing_r,
        "warning": "; ".join(warnings) if warnings else None,
        "limitations": (
            "Approximate shadow estimate: it assumes blocked trades disappear and does not "
            "model missed re-entries, changed sizing, slippage, spread, or market-state feedback."
        ),
    }


def enrich_candidates_with_shadow(
    candidates: Sequence[Dict[str, Any]],
    samples: Sequence[LearningSample],
    *,
    max_false_block_rate: float = DEFAULT_MAX_FALSE_BLOCK_RATE,
    max_trade_frequency_drop: float = DEFAULT_MAX_TRADE_FREQUENCY_DROP,
    min_live_review_samples: int = LIVE_REVIEW_MIN,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item.setdefault("candidate_action", _action_for_pattern(item) if "evidence" in item else "block")
        shadow = shadow_evaluate(
            item,
            samples,
            max_false_block_rate=max_false_block_rate,
            max_trade_frequency_drop=max_trade_frequency_drop,
            min_live_review_samples=min_live_review_samples,
        )
        item["shadow_evaluation"] = shadow
        item["evidence"] = dict(item.get("evidence") or {})
        item["evidence"]["avoided_loss_r"] = shadow["avoided_loss_r"]
        item["evidence"]["sacrificed_win_r"] = shadow["sacrificed_win_r"]
        item["opportunity_cost"] = dict(item.get("opportunity_cost") or {})
        item["opportunity_cost"].update(
            {
                "blocked_wins_count": shadow["blocked_wins_count"],
                "sacrificed_win_r": shadow["sacrificed_win_r"],
                "missed_profit_r": shadow["missed_profit_r"],
                "false_block_rate": shadow["false_block_rate"],
                "trade_frequency_delta": shadow["trade_frequency_delta"],
                "over_filtering_warning": shadow["over_filtering_warning"],
            }
        )
        item["expected_benefit"] = dict(item.get("expected_benefit") or {})
        item["expected_benefit"].update(
            {
                "avoided_loss_r": shadow["avoided_loss_r"],
                "net_r_delta": shadow["net_r_delta"],
            }
        )
        item["safety"] = dict(item.get("safety") or {})
        item["safety"]["evidence_grade"] = evidence_grade(
            int(item["evidence"].get("sample_count") or 0), shadow.get("net_r_delta")
        )
        item["safety"]["max_false_block_rate"] = max_false_block_rate
        item["safety"]["max_trade_frequency_drop"] = max_trade_frequency_drop
        item["safety"]["live_review_min_samples"] = min_live_review_samples
        item["safety"]["live_review_requires_positive_net_r_delta"] = True
        item["safety"]["live_review_requires_acceptable_false_block_rate"] = True
        item["safety"]["live_review_requires_shadow_metrics"] = REQUIRED_SHADOW_METRICS
        item["safety"]["promotion_gate_block_reasons"] = list(
            (shadow.get("promotion_gate") or {}).get("block_reasons") or []
        )
        item["live_review_ready"] = bool(shadow.get("live_review_ready"))
        enriched.append(item)
    return enriched


def repeated_loss_patterns(patterns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [
        pattern
        for pattern in patterns
        if pattern["loss_count"] >= SUSPICION_MIN
        and pattern["loss_count"] >= pattern["win_count"]
        and (pattern.get("avg_r") is None or pattern["avg_r"] <= 0)
    ]
    out.sort(key=lambda item: (item["loss_count"], item["sample_count"]), reverse=True)
    return out[:20]


def repeated_win_patterns(patterns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [
        pattern
        for pattern in patterns
        if pattern["win_count"] >= SUSPICION_MIN
        and pattern["win_count"] >= pattern["loss_count"]
        and isinstance(pattern.get("avg_r"), (int, float))
        and pattern["avg_r"] > 0
    ]
    out.sort(key=lambda item: (item["win_count"], item["sample_count"], item.get("avg_r") or 0.0), reverse=True)
    return out[:20]


def opportunity_cost_summary(shadows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    blocked_losses = sum(int(shadow.get("blocked_losses_count") or 0) for shadow in shadows)
    blocked_wins = sum(int(shadow.get("blocked_wins_count") or 0) for shadow in shadows)
    avoided_loss_r = sum(float(shadow.get("avoided_loss_r") or 0.0) for shadow in shadows)
    sacrificed_win_r = sum(float(shadow.get("sacrificed_win_r") or 0.0) for shadow in shadows)
    net_values = [shadow.get("net_r_delta") for shadow in shadows if isinstance(shadow.get("net_r_delta"), (int, float))]
    return {
        "candidate_count": len(shadows),
        "blocked_losses_count": blocked_losses,
        "blocked_wins_count": blocked_wins,
        "avoided_loss_r": avoided_loss_r,
        "sacrificed_win_r": sacrificed_win_r,
        "missed_profit_r": sacrificed_win_r,
        "net_r_delta_sum": sum(float(value) for value in net_values),
        "over_filtering_warning_count": sum(1 for shadow in shadows if shadow.get("over_filtering_warning")),
    }


def _event_name(event: Dict[str, Any]) -> str:
    parts = [
        event.get("event"),
        event.get("event_type"),
        event.get("type"),
        event.get("action"),
        event.get("decision"),
        event.get("reason"),
    ]
    return " ".join(str(part).lower() for part in parts if part is not None)


def event_frequency_context(events_path: Optional[Path]) -> Dict[str, Any]:
    if events_path is None:
        return {"events_path": None, "available": False}
    rows = _read_jsonl(events_path)
    signal_count = 0
    blocked_count = 0
    trade_count = 0
    for row in rows:
        name = _event_name(row)
        if any(token in name for token in ("signal", "setup", "candidate", "entry_decision")):
            signal_count += 1
        if any(token in name for token in ("block", "reject", "skip", "filtered")):
            blocked_count += 1
        if any(token in name for token in ("trade", "entry", "order", "deal", "position")):
            trade_count += 1
    block_rate = blocked_count / signal_count if signal_count else None
    return {
        "events_path": events_path.as_posix(),
        "available": events_path.exists(),
        "event_rows": len(rows),
        "signal_count": signal_count,
        "blocked_signal_count": blocked_count,
        "trade_or_entry_event_count": trade_count,
        "block_rate": block_rate,
    }


def _walk_first_float(value: Any, keys: Sequence[str], depth: int = 0) -> Optional[float]:
    if depth > 5:
        return None
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                found = _safe_float(value.get(key))
                if found is not None:
                    return found
        for nested in value.values():
            found = _walk_first_float(nested, keys, depth + 1)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _walk_first_float(nested, keys, depth + 1)
            if found is not None:
                return found
    return None


def _event_action(row: Dict[str, Any]) -> str:
    return str(row.get("action") or "").strip().upper()


def _is_entry_signal_event(row: Dict[str, Any]) -> bool:
    if str(row.get("event") or "").strip().lower() != "decision":
        return False
    action = _event_action(row)
    reason = str(row.get("reason") or "").upper()
    return action in {"BUY", "SELL"} or reason.endswith("_ENTRY")


def _is_block_event(row: Dict[str, Any]) -> bool:
    name = _event_name(row)
    return any(token in name for token in ("block", "reject", "skip", "filtered"))


def _event_regime_key(row: Dict[str, Any]) -> str:
    volatility = _walk_first_float(row, VOLATILITY_REGIME_FEATURES)
    if volatility is None:
        return "VOL_MISSING"
    return _volatility_regime_label(volatility)


def _event_spread_thresholds(rows: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    by_symbol: Dict[str, List[float]] = {}
    for row in rows:
        spread = _walk_first_float(row, SPREAD_REGIME_FEATURES)
        if spread is None:
            continue
        symbol = str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        by_symbol.setdefault(symbol, []).append(spread)

    thresholds: Dict[str, Tuple[float, float]] = {}
    for symbol, values in by_symbol.items():
        ordered = sorted(values)
        if len(ordered) < 3:
            continue
        low_idx = max(0, min(len(ordered) - 1, int(math.floor((len(ordered) - 1) * 0.33))))
        high_idx = max(0, min(len(ordered) - 1, int(math.floor((len(ordered) - 1) * 0.66))))
        thresholds[symbol] = (ordered[low_idx], ordered[high_idx])
    return thresholds


def _event_spread_regime_key(row: Dict[str, Any], thresholds: Dict[str, Tuple[float, float]]) -> str:
    spread = _walk_first_float(row, SPREAD_REGIME_FEATURES)
    if spread is None:
        return "SPREAD_MISSING"
    symbol = str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
    cuts = thresholds.get(symbol)
    if cuts is None:
        return "SPREAD_OBSERVED"
    low_cut, high_cut = cuts
    if spread <= low_cut:
        return "TIGHT_SPREAD"
    if spread <= high_cut:
        return "NORMAL_SPREAD"
    return "WIDE_SPREAD"


def _event_series_key(row: Dict[str, Any]) -> Tuple[str, str]:
    symbol = str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
    strategy = str(row.get("strategy") or "UNKNOWN").strip() or "UNKNOWN"
    return symbol, strategy


def _event_metric_context(row: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    volatility = _walk_first_float(row, VOLATILITY_REGIME_FEATURES)
    if volatility is not None:
        context["atr_regime_ratio"] = volatility
    spread = _walk_first_float(row, SPREAD_REGIME_FEATURES)
    if spread is not None:
        context["current_spread"] = spread
    return context


def _contextualize_block_events(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    last_entry_context: Dict[Tuple[str, str], Dict[str, Any]] = {}
    contextualized: List[Dict[str, Any]] = []
    inherited_atr_count = 0
    inherited_spread_count = 0

    for row in rows:
        key = _event_series_key(row)
        if _is_entry_signal_event(row):
            context = _event_metric_context(row)
            if context:
                last_entry_context[key] = context

        if not _is_block_event(row):
            continue

        item = dict(row)
        context = last_entry_context.get(key)
        if context:
            inherited: Dict[str, Any] = {}
            if _walk_first_float(item, VOLATILITY_REGIME_FEATURES) is None and "atr_regime_ratio" in context:
                inherited["atr_regime_ratio"] = context["atr_regime_ratio"]
                inherited_atr_count += 1
            if _walk_first_float(item, SPREAD_REGIME_FEATURES) is None and "current_spread" in context:
                inherited["current_spread"] = context["current_spread"]
                inherited_spread_count += 1
            if inherited:
                inherited["source"] = "previous_entry_decision_same_symbol_strategy"
                item["_entry_signal_context"] = inherited
        contextualized.append(item)

    return contextualized, inherited_atr_count, inherited_spread_count


def _summarize_event_bucket(key: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    atr_values = [
        value
        for value in (_walk_first_float(row, VOLATILITY_REGIME_FEATURES) for row in rows)
        if value is not None
    ]
    spread_values = [
        value
        for value in (_walk_first_float(row, SPREAD_REGIME_FEATURES) for row in rows)
        if value is not None
    ]
    reasons: Dict[str, int] = {}
    symbols: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "UNKNOWN")
        reasons[reason] = reasons.get(reason, 0) + 1
        symbol = str(row.get("symbol") or "UNKNOWN").upper()
        symbols[symbol] = symbols.get(symbol, 0) + 1
    return {
        "key": key,
        "event_count": len(rows),
        "atr_value_count": len(atr_values),
        "spread_value_count": len(spread_values),
        "spread_coverage": len(spread_values) / len(rows) if rows else None,
        "avg_atr_regime_ratio": _mean(atr_values),
        "avg_spread": _mean(spread_values),
        "top_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:6]
        ],
        "symbols": [
            {"symbol": symbol, "count": count}
            for symbol, count in sorted(symbols.items(), key=lambda item: (-item[1], item[0]))[:6]
        ],
    }


def event_regime_context(events_path: Optional[Path]) -> Dict[str, Any]:
    if events_path is None:
        return {"events_path": None, "available": False}
    rows = _read_jsonl(events_path)
    entry_rows = [row for row in rows if _is_entry_signal_event(row)]
    block_rows, block_inherited_atr_count, block_inherited_spread_count = _contextualize_block_events(rows)
    event_spread_thresholds = _event_spread_thresholds(entry_rows or block_rows)

    def group(items: Sequence[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], str]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in items:
            buckets.setdefault(key_fn(row), []).append(row)
        out = [_summarize_event_bucket(key, values) for key, values in buckets.items()]
        out.sort(key=lambda item: (item["event_count"], item["key"]), reverse=True)
        return out

    def volatility_spread_key(row: Dict[str, Any]) -> str:
        return f"{_event_regime_key(row)}|{_event_spread_regime_key(row, event_spread_thresholds)}"

    entry_spread_count = sum(1 for row in entry_rows if _walk_first_float(row, SPREAD_REGIME_FEATURES) is not None)
    entry_atr_count = sum(1 for row in entry_rows if _walk_first_float(row, VOLATILITY_REGIME_FEATURES) is not None)
    warnings: List[str] = []
    if entry_rows and entry_spread_count == 0:
        warnings.append("entry decision 이벤트에 spread telemetry가 없어 spread 레짐별 신호 품질을 계산하지 못한다.")
    elif entry_rows and entry_spread_count / len(entry_rows) < 0.8:
        warnings.append("entry decision 이벤트의 spread telemetry coverage가 낮다. 전역 spread 필터 튜닝 근거로 쓰지 않는다.")
    if entry_rows and entry_atr_count == 0:
        warnings.append("entry decision 이벤트에 atr_regime_ratio가 없어 변동성 레짐별 신호 품질을 계산하지 못한다.")
    if block_rows and block_inherited_atr_count:
        warnings.append(
            "block/skip 이벤트 일부는 직전 entry decision의 ATR 레짐을 상속해 집계했다. "
            "skip payload 자체에도 regime telemetry를 넣으면 더 안전하다."
        )
    if block_rows and block_inherited_spread_count:
        warnings.append(
            "block/skip 이벤트 일부는 직전 entry decision의 spread를 상속해 집계했다. "
            "skip payload 자체에도 spread telemetry를 넣으면 더 안전하다."
        )
    if block_rows and len(entry_rows) < CANDIDATE_MIN:
        warnings.append("차단 이벤트는 있지만 closed/live entry 표본이 적다. block reason 분포만 보고 보수화하지 않는다.")

    return {
        "events_path": events_path.as_posix(),
        "available": events_path.exists(),
        "event_rows": len(rows),
        "entry_signal_count": len(entry_rows),
        "entry_signal_atr_value_count": entry_atr_count,
        "entry_signal_spread_value_count": entry_spread_count,
        "entry_signal_spread_coverage": entry_spread_count / len(entry_rows) if entry_rows else None,
        "blocked_signal_atr_inherited_count": block_inherited_atr_count,
        "blocked_signal_spread_inherited_count": block_inherited_spread_count,
        "event_spread_thresholds_by_symbol": {
            symbol: {"tight_cut": cuts[0], "wide_cut": cuts[1]} for symbol, cuts in event_spread_thresholds.items()
        },
        "entry_signal_by_volatility": group(entry_rows, _event_regime_key),
        "blocked_by_volatility": group(block_rows, _event_regime_key),
        "entry_signal_by_volatility_spread": group(entry_rows, volatility_spread_key),
        "blocked_by_volatility_spread": group(block_rows, volatility_spread_key),
        "warnings": warnings,
        "interpretation": (
            "event 로그는 closed-trade 기대값을 대체하지 않는다. "
            "다만 spread/ATR telemetry coverage와 block reason의 레짐 편향을 보여 주어 "
            "전역 필터 튜닝 전에 어떤 데이터가 비어 있는지 확인하게 한다."
        ),
    }


def trade_starvation_health(
    samples: Sequence[LearningSample],
    candidates: Sequence[Dict[str, Any]],
    events_path: Optional[Path],
    *,
    max_trade_frequency_drop: float = DEFAULT_MAX_TRADE_FREQUENCY_DROP,
) -> Dict[str, Any]:
    event_context = event_frequency_context(events_path)
    event_regimes = event_regime_context(events_path)
    blocking_candidates = [
        candidate
        for candidate in candidates
        if _is_risk_reduction_action(str(candidate.get("candidate_action") or ""))
    ]
    over_filtering_candidates = [
        candidate
        for candidate in blocking_candidates
        if (candidate.get("shadow_evaluation") or {}).get("over_filtering_warning")
    ]
    actual_trades = len(samples)
    candidate_block_count = sum(
        int((candidate.get("shadow_evaluation") or {}).get("blocked_total") or 0)
        for candidate in blocking_candidates
    )
    warnings: List[str] = []
    if not event_context.get("available"):
        warnings.append("decision/event 로그가 없어 실제 signal 대비 block 비율은 추정하지 못했다.")
    if actual_trades and candidate_block_count / actual_trades > max_trade_frequency_drop:
        warnings.append("후보들이 현재 표본의 큰 비중을 막을 수 있어 trade starvation 점검이 필요하다.")
    block_rate = event_context.get("block_rate")
    if isinstance(block_rate, (int, float)) and block_rate > max_trade_frequency_drop and actual_trades < CANDIDATE_MIN:
        warnings.append("대부분 신호가 막히는데 closed-trade 증거는 부족하다. 보수화 결론 금지.")
    return {
        "actual_trade_samples": actual_trades,
        "blocking_candidate_count": len(blocking_candidates),
        "candidate_matched_trade_count_sum": candidate_block_count,
        "over_filtering_candidate_count": len(over_filtering_candidates),
        "max_trade_frequency_drop": max_trade_frequency_drop,
        "events": event_context,
        "event_regime_context": event_regimes,
        "warnings": warnings,
    }


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if value == math.inf:
        return "inf"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown_report(
    overall: Dict[str, Any],
    patterns: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    shadows: Sequence[Dict[str, Any]],
    health: Optional[Dict[str, Any]] = None,
    regime_expectancy: Optional[Dict[str, Any]] = None,
    execution_shortfall: Optional[Dict[str, Any]] = None,
    lsr_confirmation_expectancy: Optional[Dict[str, Any]] = None,
) -> str:
    repeated = repeated_loss_patterns(patterns)
    wins = repeated_win_patterns(patterns)
    opportunity = opportunity_cost_summary(shadows)
    lines: List[str] = [
        "# Postmortem Learning Review",
        "",
        "> REVIEW ONLY. 이 보고서는 live config, risk gate, entry gate를 자동 변경하지 않는다.",
        "> 작은 표본으로 문을 잠그는 것은 금지다. 후보 규칙은 shadow/replay 검증 전까지 제안일 뿐이다.",
        "> 미래 차트는 예측할 수 없다. 여기서 하는 일은 예언이 아니라 기대값 필터링이며, 봇을 겁쟁이로 만드는 학습을 막는 것이다.",
        "",
        "## 전체 성과",
        f"- 샘플: {overall['sample_count']}건 / 승 {overall['win_count']} / 패 {overall['loss_count']} / 본전 {overall['flat_count']}",
        f"- 평균 R / 기대값: {_format_number(overall.get('avg_r'))}",
        f"- profit factor: {_format_number(overall.get('profit_factor'))}",
        f"- R 누락: {overall['missing_r_count']}건",
        "",
        "## 증거 게이트",
        "- 1건: 관찰만. 결론 금지.",
        "- 3-5건 반복: 의심. 더 모으고 shadow로 본다.",
        "- 10건 이상: 후보 규칙으로 검토 가능.",
        "- 20-50건 이상 + positive netR + 허용 가능한 false block rate: live-review 후보. 그래도 자동 적용 금지.",
        "",
        "## 반복 손실 패턴",
    ]
    if not repeated:
        lines.append("- 3건 이상 반복 손실 패턴은 아직 없다. 지금은 성급한 자동 강화가 더 위험하다.")
    else:
        for pattern in repeated[:10]:
            lines.append(
                "- "
                f"{pattern['description_korean']} | 샘플 {pattern['sample_count']} / "
                f"손실 {pattern['loss_count']} / 승 {pattern['win_count']} / "
                f"평균R {_format_number(pattern.get('avg_r'))} / PF {_format_number(pattern.get('profit_factor'))}"
            )

    lines.extend(["", "## 반복 수익/예외 보호 패턴"])
    if not wins:
        lines.append("- 3건 이상 반복 수익 패턴은 아직 없다. 승자를 막는 규칙을 만들 근거도 아직 부족하다.")
    else:
        for pattern in wins[:10]:
            lines.append(
                "- "
                f"{pattern['description_korean']} | 샘플 {pattern['sample_count']} / "
                f"승 {pattern['win_count']} / 손실 {pattern['loss_count']} / "
                f"평균R {_format_number(pattern.get('avg_r'))} / PF {_format_number(pattern.get('profit_factor'))}"
            )

    lines.extend(["", "## 기회비용 요약"])
    lines.append(
        "- "
        f"후보 {opportunity['candidate_count']}개가 shadow에서 막거나 보호한 거래: "
        f"손실 {opportunity['blocked_losses_count']} / 승자 {opportunity['blocked_wins_count']}, "
        f"회피 손실 {_format_number(opportunity['avoided_loss_r'])}R, "
        f"희생/미스 수익 {_format_number(opportunity['missed_profit_r'])}R, "
        f"netR 합계 {_format_number(opportunity['net_r_delta_sum'])}R."
    )
    if opportunity["over_filtering_warning_count"]:
        lines.append(f"- 과필터링 경고 후보: {opportunity['over_filtering_warning_count']}개. 이 후보들은 live-review 전에 기각/축소해야 한다.")

    lines.extend(["", "## TCA Execution Shortfall"])
    if not execution_shortfall:
        lines.append("- execution shortfall 데이터를 만들지 못했다.")
    else:
        tca_overall = execution_shortfall.get("overall") or {}
        lines.append(
            "- "
            f"paired R {tca_overall.get('paired_r_value_count')}건: "
            f"signal price 기대값 {_format_number(tca_overall.get('avg_signal_price_r'))}R, "
            f"실현 net 기대값 {_format_number(tca_overall.get('avg_realized_net_r'))}R, "
            f"execution drag {_format_number(tca_overall.get('avg_execution_drag_r'))}R, "
            f"entry shortfall {_format_number(tca_overall.get('avg_entry_implementation_shortfall_r'))}R, "
            f"drag/signal {_format_number(tca_overall.get('execution_drag_to_signal_ratio'))}, "
            f"net/signal {_format_number(tca_overall.get('net_realization_ratio'))}."
        )
        lines.append(
            "- "
            f"signal은 양수였지만 net이 음수인 거래: "
            f"{tca_overall.get('signal_positive_net_negative_count')}건 "
            f"(positive signal {tca_overall.get('signal_positive_trade_count')}건 기준, "
            f"net<=0 rate {_format_number(tca_overall.get('signal_positive_net_nonpositive_rate'))}). "
            "이 케이스는 전략 타점과 실제 체결 품질을 분리해서 봐야 한다."
        )
        tuning_gate = execution_shortfall.get("tuning_gate") or {}
        lines.append(
            "- "
            f"TCA tuning gate: {tuning_gate.get('status', 'unknown')} / "
            f"blocks_signal_threshold_tuning={tuning_gate.get('blocks_signal_threshold_tuning')}. "
            f"reason_codes={tuning_gate.get('reason_codes') or []}"
        )
        if tuning_gate.get("recommendation_korean"):
            lines.append(f"- 권고: {tuning_gate['recommendation_korean']}")
        for warning in execution_shortfall.get("warnings") or []:
            lines.append(f"- 경고: {warning}")
        symbol_strategy = ((execution_shortfall.get("groups") or {}).get("by_symbol_strategy") or [])[:8]
        if symbol_strategy:
            lines.append("- setup별 순체결 드래그:")
            for row in symbol_strategy:
                lines.append(
                    "  - "
                    f"{row['key']} | 샘플 {row['sample_count']} / paired {row['paired_r_value_count']} / "
                    f"signal {_format_number(row.get('avg_signal_price_r'))}R -> "
                    f"net {_format_number(row.get('avg_realized_net_r'))}R / "
                    f"drag {_format_number(row.get('avg_execution_drag_r'))}R / "
                    f"drag/signal {_format_number(row.get('execution_drag_to_signal_ratio'))} / "
                    f"shortfall {_format_number(row.get('avg_entry_implementation_shortfall_r'))}R / "
                    f"spread {_format_number(row.get('avg_spread'))}"
                )

    lines.extend(["", "## Volatility / Spread Regime Expectancy"])
    if not regime_expectancy:
        lines.append("- 레짐별 기대값 데이터를 만들지 못했다.")
    else:
        lines.append(
            "- "
            f"변동성 값 {regime_expectancy.get('volatility_value_count')}건, "
            f"스프레드 값 {regime_expectancy.get('spread_value_count')}건. "
            "전역 필터 후보는 아래 레짐 차이를 확인하기 전 live gate로 올리면 안 된다."
        )
        quality_rows = ((regime_expectancy.get("data_quality") or {}).get("by_symbol") or [])[:8]
        if quality_rows:
            lines.append("- symbol별 레짐 데이터 품질:")
            for row in quality_rows:
                lines.append(
                    "  - "
                    f"{row.get('symbol')} | samples {row.get('sample_count')} / "
                    f"vol {row.get('volatility_value_count')} "
                    f"({_format_number(row.get('volatility_coverage'))}) / "
                    f"spread {row.get('spread_value_count')} "
                    f"({_format_number(row.get('spread_coverage'))}) / "
                    f"spread_threshold {row.get('spread_threshold_status')}"
                )
        for warning in regime_expectancy.get("warnings") or []:
            lines.append(f"- 경고: {warning}")
        combined = ((regime_expectancy.get("dimensions") or {}).get("volatility_x_spread") or [])[:12]
        if not combined:
            lines.append("- 결합 레짐 bucket이 비어 있다.")
        else:
            for row in combined:
                lines.append(
                    "- "
                    f"{row['key']} | 샘플 {row['sample_count']} / 승 {row['win_count']} / 패 {row['loss_count']} / "
                    f"기대값R {_format_number(row.get('expectancy_r'))} / PF {_format_number(row.get('profit_factor'))} / "
                    f"avg_spread {_format_number(row.get('avg_spread'))} / avg_feeRR {_format_number(row.get('avg_fee_adjusted_rr'))}"
                )
        setup_contrasts = (regime_expectancy.get("setup_regime_contrasts") or [])[:8]
        if setup_contrasts:
            lines.append("- setup별 레짐 대비:")
            for row in setup_contrasts:
                lines.append(
                    "  - "
                    f"{row['setup']} | gap {_format_number(row.get('expectancy_gap_r'))}R / "
                    f"best {row.get('best_regime')} {_format_number(row.get('best_expectancy_r'))}R "
                    f"({row.get('best_sample_count')}건) vs "
                    f"worst {row.get('worst_regime')} {_format_number(row.get('worst_expectancy_r'))}R "
                    f"({row.get('worst_sample_count')}건)"
                )

    lines.extend(["", "## LSR Confirmation Path Expectancy"])
    if not lsr_confirmation_expectancy:
        lines.append("- LSR confirmation path 데이터를 만들지 못했다.")
    else:
        lines.append(
            "- "
            f"LSR 표본 {lsr_confirmation_expectancy.get('sample_count')}건: "
            f"unconfirmed reclaim {lsr_confirmation_expectancy.get('unconfirmed_reclaim_count')}건, "
            f"retest/확인봉 {lsr_confirmation_expectancy.get('retest_confirmed_count')}건, "
            f"unknown {lsr_confirmation_expectancy.get('unknown_confirmation_count')}건. "
            "sweep 단독 신호를 좋은 reclaim과 섞어서 튜닝하면 안 된다."
        )
        metadata_quality = (lsr_confirmation_expectancy.get("dimensions") or {}).get("metadata_quality") or {}
        if metadata_quality:
            lines.append(
                "- "
                f"metadata completeness: path present {metadata_quality.get('path_present_count')} / "
                f"missing {metadata_quality.get('path_missing_count')}, "
                f"reclaim metrics complete {metadata_quality.get('reclaim_metric_complete_count')} / "
                f"missing {metadata_quality.get('reclaim_metric_missing_count')}, "
                f"timing complete {metadata_quality.get('reclaim_timing_complete_count')} / "
                f"missing {metadata_quality.get('reclaim_timing_missing_count')}"
            )
        for warning in lsr_confirmation_expectancy.get("warnings") or []:
            lines.append(f"- 경고: {warning}")
        by_path = ((lsr_confirmation_expectancy.get("dimensions") or {}).get("by_confirmation_path") or [])[:8]
        if not by_path:
            lines.append("- confirmation path bucket이 비어 있다.")
        else:
            for row in by_path:
                lines.append(
                    "- "
                    f"{row['key']} | 샘플 {row['sample_count']} / 승 {row['win_count']} / 패 {row['loss_count']} / "
                    f"기대값R {_format_number(row.get('expectancy_r'))} / PF {_format_number(row.get('profit_factor'))} / "
                    f"weak {row.get('weak_reclaim_after_deep_sweep_count')} / "
                    f"shallow {row.get('shallow_reclaim_confirmation_count')} / "
                    f"late {row.get('late_window_reclaim_count')} / "
                    f"bad_time {row.get('invalid_reclaim_timing_count')} / "
                    f"chase {row.get('unconfirmed_reclaim_chase_count')} / "
                    f"reclaim/sweep {_format_number(row.get('avg_reclaim_to_sweep_depth_ratio'))} / "
                    f"window_used {_format_number(row.get('avg_reclaim_window_elapsed_ratio'))}"
                )

    lines.extend(["", "## 후보 규칙"])
    if not candidates:
        lines.append("- 현재 min-samples 기준을 통과한 후보가 없다. 억지로 live gate를 만지지 않는다.")
    else:
        for candidate in candidates[:12]:
            shadow = candidate.get("shadow_evaluation") or {}
            lines.append(
                "- "
                f"`{candidate['candidate_id']}` {candidate.get('candidate_action')} / {candidate['rule_type']} / {candidate['feature']} / "
                f"{candidate['safety'].get('evidence_grade')} / "
                f"blocked L/W {shadow.get('blocked_losses', 0)}/{shadow.get('blocked_wins', 0)} / "
                f"false_block {_format_number(shadow.get('false_block_rate'))} / "
                f"freq_delta {_format_number(shadow.get('trade_frequency_delta'))} / "
                f"netR {_format_number(shadow.get('net_r_delta'))} / "
                f"live_review_ready={candidate.get('live_review_ready', False)}"
            )
            lines.append(f"  - {candidate['rationale_korean']}")

    lines.extend(["", "## Shadow 평가 한계"])
    if not shadows:
        lines.append("- 평가할 후보가 없다.")
    else:
        for shadow in shadows[:8]:
            warning = shadow.get("warning") or "경고 없음"
            lines.append(
                "- "
                f"`{shadow.get('candidate_id')}` blocked {shadow.get('blocked_total')}건, "
                f"false_block {_format_number(shadow.get('false_block_rate'))}, "
                f"freq_delta {_format_number(shadow.get('trade_frequency_delta'))}, "
                f"netR {_format_number(shadow.get('net_r_delta'))}, trade delta {shadow.get('trade_count_delta')}. "
                f"{warning}"
            )
    lines.extend(["", "## Trade Starvation / Over-filtering Health"])
    if not health:
        lines.append("- health 데이터를 만들지 못했다.")
    else:
        lines.append(
            "- "
            f"실거래 표본 {health.get('actual_trade_samples')}건, 차단형 후보 {health.get('blocking_candidate_count')}개, "
            f"후보 매칭 합계 {health.get('candidate_matched_trade_count_sum')}건, "
            f"과필터링 후보 {health.get('over_filtering_candidate_count')}개."
        )
        events = health.get("events") or {}
        if events.get("available"):
            lines.append(
                "- "
                f"events: signals {events.get('signal_count')} / blocked {events.get('blocked_signal_count')} / "
                f"block_rate {_format_number(events.get('block_rate'))}."
            )
        event_regimes = health.get("event_regime_context") or {}
        if event_regimes.get("available"):
            lines.append(
                "- "
                f"event regimes: entry signals {event_regimes.get('entry_signal_count')} / "
                f"ATR coverage {event_regimes.get('entry_signal_atr_value_count')} / "
                f"spread coverage {_format_number(event_regimes.get('entry_signal_spread_coverage'))} / "
                f"blocked ATR inherited {event_regimes.get('blocked_signal_atr_inherited_count', 0)} / "
                f"blocked spread inherited {event_regimes.get('blocked_signal_spread_inherited_count', 0)}."
            )
            for row in (event_regimes.get("entry_signal_by_volatility") or [])[:6]:
                lines.append(
                    "  - "
                    f"{row['key']} | entry events {row['event_count']} / "
                    f"avg ATR ratio {_format_number(row.get('avg_atr_regime_ratio'))} / "
                    f"spread coverage {_format_number(row.get('spread_coverage'))} / "
                    f"avg spread {_format_number(row.get('avg_spread'))}"
                )
            for row in (event_regimes.get("blocked_by_volatility") or [])[:6]:
                top_reason = ((row.get("top_reasons") or [{}])[0]).get("reason", "UNKNOWN")
                lines.append(
                    "  - "
                    f"{row['key']} | blocked events {row['event_count']} / "
                    f"top reason {top_reason} / "
                    f"ATR values {row.get('atr_value_count')} / "
                    f"spread coverage {_format_number(row.get('spread_coverage'))}"
                )
            crossed_rows = (event_regimes.get("blocked_by_volatility_spread") or [])[:6]
            if crossed_rows:
                lines.append("  - blocked volatility x spread:")
                for row in crossed_rows:
                    top_reason = ((row.get("top_reasons") or [{}])[0]).get("reason", "UNKNOWN")
                    lines.append(
                        "    - "
                        f"{row['key']} | events {row['event_count']} / "
                        f"top reason {top_reason} / "
                        f"avg spread {_format_number(row.get('avg_spread'))}"
                    )
            for warning in event_regimes.get("warnings") or []:
                lines.append(f"- 경고: {warning}")
        for warning in health.get("warnings") or []:
            lines.append(f"- 경고: {warning}")
    lines.append("")
    lines.append("이 보고서는 사후 학습용이다. 주문, 청산, live 설정 변경 기능은 없다.")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_candidates(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return _read_json_file(path)


def analyze_learning(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    postmortem_dir = Path(args.postmortem_dir) if args.postmortem_dir else None
    max_false_block_rate = float(getattr(args, "max_false_block_rate", DEFAULT_MAX_FALSE_BLOCK_RATE))
    max_trade_frequency_drop = float(getattr(args, "max_trade_frequency_drop", DEFAULT_MAX_TRADE_FREQUENCY_DROP))
    min_live_review_samples = int(getattr(args, "min_live_review_samples", LIVE_REVIEW_MIN))
    events_path = Path(args.events) if getattr(args, "events", None) else None
    samples = load_samples(input_path, postmortem_dir)
    patterns = aggregate_patterns(samples)
    overall = overall_stats(samples)
    regime_expectancy = regime_expectancy_summary(samples)
    execution_shortfall = execution_shortfall_summary(samples)
    lsr_confirmation_expectancy = lsr_confirmation_expectancy_summary(samples)

    if args.shadow_candidates:
        raw_candidates = read_candidates(Path(args.shadow_candidates))
    else:
        raw_candidates = generate_candidates(
            patterns,
            min_samples_to_emit=max(OBSERVATION_MIN, int(args.min_samples)),
            limit=max(0, int(args.limit_candidates)),
        )
    candidates = enrich_candidates_with_shadow(
        raw_candidates,
        samples,
        max_false_block_rate=max_false_block_rate,
        max_trade_frequency_drop=max_trade_frequency_drop,
        min_live_review_samples=min_live_review_samples,
    )
    shadows = [candidate["shadow_evaluation"] for candidate in candidates]
    health = trade_starvation_health(
        samples,
        candidates,
        events_path,
        max_trade_frequency_drop=max_trade_frequency_drop,
    )

    aggregate_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": input_path.as_posix(),
        "postmortem_dir": postmortem_dir.as_posix() if postmortem_dir else None,
        "overall": overall,
        "patterns": patterns,
        "repeated_loss_patterns": repeated_loss_patterns(patterns),
        "repeated_win_patterns": repeated_win_patterns(patterns),
        "opportunity_cost": opportunity_cost_summary(shadows),
        "regime_expectancy": regime_expectancy,
        "execution_shortfall": execution_shortfall,
        "lsr_confirmation_expectancy": lsr_confirmation_expectancy,
        "trade_starvation_health": health,
        "safety": {
            "review_only": True,
            "live_apply_allowed": False,
            "never_auto_mutate_live_gates": True,
            "allowed_candidate_actions": ACTION_TYPES,
            "live_review_policy": {
                "min_live_review_samples": min_live_review_samples,
                "max_false_block_rate": max_false_block_rate,
                "max_trade_frequency_drop": max_trade_frequency_drop,
                "requires_positive_net_r_delta": True,
                "requires_shadow_metrics": REQUIRED_SHADOW_METRICS,
                "never_auto_mutates_live_config": True,
            },
            "evidence_gates": {
                "observation_only": "1 trade",
                "suspicion": "3-5 repeated",
                "candidate_rule": "10+",
                "possible_live_review_candidate": "20-50+ plus positive net_r_delta and acceptable false_block_rate",
            },
        },
    }
    _write_json(output_dir / AGGREGATES_FILE, aggregate_payload)
    _write_jsonl(output_dir / CANDIDATES_FILE, candidates)
    _write_jsonl(output_dir / SHADOW_FILE, shadows)
    (output_dir / REVIEW_FILE).write_text(
        build_markdown_report(
            overall,
            patterns,
            candidates,
            shadows,
            health,
            regime_expectancy,
            execution_shortfall,
            lsr_confirmation_expectancy,
        ),
        encoding="utf-8",
    )
    return {
        "sample_count": len(samples),
        "pattern_count": len(patterns),
        "candidate_count": len(candidates),
        "output_dir": output_dir.as_posix(),
        "aggregate_path": (output_dir / AGGREGATES_FILE).as_posix(),
        "candidate_path": (output_dir / CANDIDATES_FILE).as_posix(),
        "shadow_path": (output_dir / SHADOW_FILE).as_posix(),
        "review_path": (output_dir / REVIEW_FILE).as_posix(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = analyze_learning(args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
