from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional for callers.
    pd = None  # type: ignore[assignment]


Direction = Literal["long", "short"]


@dataclass(frozen=True)
class TradeOpportunity:
    symbol: str
    timeframe: str
    direction: Direction
    entry_price: float
    invalidation_price: float
    target_reference_price: float
    signal_score: float
    components: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    detected_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    late_entry: bool = False


class TradeOpportunityScanner:
    """
    MT5-free liquidity sweep/reclaim scanner for execution-first workflows.

    The scanner evaluates the latest closed bar against prior OHLC context and
    returns scored long/short reversal candidates when the score passes the
    configured threshold.
    """

    def __init__(
        self,
        *,
        lookback_bars: int = 20,
        atr_period: int = 14,
        min_signal_score: float = 70.0,
        sweep_buffer_atr: float = 0.05,
        stop_buffer_atr: float = 0.05,
        min_atr: float = 0.0,
        late_entry_atr_mult: float = 0.75,
        late_entry_min_rr: float = 1.0,
        round_turn_cost: float = 0.0,
    ) -> None:
        self.lookback_bars = max(3, int(lookback_bars))
        self.atr_period = max(2, int(atr_period))
        self.min_signal_score = float(min_signal_score)
        self.sweep_buffer_atr = max(0.0, float(sweep_buffer_atr))
        self.stop_buffer_atr = max(0.0, float(stop_buffer_atr))
        self.min_atr = max(0.0, float(min_atr))
        self.late_entry_atr_mult = max(0.0, float(late_entry_atr_mult))
        self.late_entry_min_rr = max(0.0, float(late_entry_min_rr))
        self.round_turn_cost = max(0.0, float(round_turn_cost))

    def scan(
        self,
        *,
        symbol: str,
        timeframe: str,
        bars: Any,
        spread: Optional[float] = None,
        round_turn_cost: Optional[float] = None,
        min_signal_score: Optional[float] = None,
        detected_at_utc: Optional[datetime] = None,
    ) -> List[TradeOpportunity]:
        rows = self._coerce_rows(bars)
        if len(rows) < self.lookback_bars + 1:
            return []

        work = rows[-(self.lookback_bars + 1) :]
        prior = work[:-1]
        signal = work[-1]
        atr = self._atr(work, self.atr_period)
        if atr is None:
            atr = self._average_range(work)
        if atr is None or atr <= 0.0 or not math.isfinite(atr):
            return []

        spread_value = self._positive_float(spread, default=0.0)
        cost_value = self._positive_float(round_turn_cost, default=self.round_turn_cost)
        threshold = self.min_signal_score if min_signal_score is None else float(min_signal_score)
        detected_at = self._coerce_datetime(detected_at_utc) or self._bar_time(signal) or datetime.now(timezone.utc)

        prior_low = min(float(row["low"]) for row in prior)
        prior_high = max(float(row["high"]) for row in prior)
        last_open = float(signal["open"])
        last_high = float(signal["high"])
        last_low = float(signal["low"])
        last_close = float(signal["close"])
        buffer = atr * self.sweep_buffer_atr
        stop_buffer = atr * self.stop_buffer_atr

        candidates: List[TradeOpportunity] = []
        if last_low < prior_low - buffer and last_close > prior_low:
            opportunity = self._build_candidate(
                symbol=symbol,
                timeframe=timeframe,
                direction="long",
                entry=last_close,
                invalidation=last_low - stop_buffer,
                target=prior_high,
                swept_level=prior_low,
                sweep_extreme=last_low,
                signal_open=last_open,
                signal_high=last_high,
                signal_low=last_low,
                signal_close=last_close,
                atr=atr,
                spread=spread_value,
                round_turn_cost=cost_value,
                detected_at_utc=detected_at,
            )
            if opportunity.signal_score >= threshold:
                candidates.append(opportunity)

        if last_high > prior_high + buffer and last_close < prior_high:
            opportunity = self._build_candidate(
                symbol=symbol,
                timeframe=timeframe,
                direction="short",
                entry=last_close,
                invalidation=last_high + stop_buffer,
                target=prior_low,
                swept_level=prior_high,
                sweep_extreme=last_high,
                signal_open=last_open,
                signal_high=last_high,
                signal_low=last_low,
                signal_close=last_close,
                atr=atr,
                spread=spread_value,
                round_turn_cost=cost_value,
                detected_at_utc=detected_at,
            )
            if opportunity.signal_score >= threshold:
                candidates.append(opportunity)

        candidates.sort(key=lambda item: item.signal_score, reverse=True)
        return candidates

    def _build_candidate(
        self,
        *,
        symbol: str,
        timeframe: str,
        direction: Direction,
        entry: float,
        invalidation: float,
        target: float,
        swept_level: float,
        sweep_extreme: float,
        signal_open: float,
        signal_high: float,
        signal_low: float,
        signal_close: float,
        atr: float,
        spread: float,
        round_turn_cost: float,
        detected_at_utc: datetime,
    ) -> TradeOpportunity:
        long_side = direction == "long"
        fallback_reward_multiple = 2.0
        risk = entry - invalidation if long_side else invalidation - entry
        if risk <= 0.0:
            risk = max(atr, abs(entry) * 0.001)
            invalidation = entry - risk if long_side else entry + risk

        raw_reward = target - entry if long_side else entry - target
        if raw_reward <= 0.0:
            raw_reward = risk * fallback_reward_multiple
            target = entry + raw_reward if long_side else entry - raw_reward

        gross_rr = raw_reward / risk if risk > 0.0 else 0.0
        fee_adjusted_reward = max(0.0, raw_reward - round_turn_cost - spread)
        fee_adjusted_rr = fee_adjusted_reward / risk if risk > 0.0 else 0.0
        reclaim_distance = abs(entry - swept_level)
        sweep_depth = abs(swept_level - sweep_extreme)

        component_scores = {
            "liquidity_sweep_quality": self._score_sweep_quality(sweep_depth, atr),
            "reclaim_or_rejection_strength": self._score_reclaim_strength(
                direction=direction,
                open_price=signal_open,
                high_price=signal_high,
                low_price=signal_low,
                close_price=signal_close,
                swept_level=swept_level,
                atr=atr,
            ),
            "invalidation_distance_efficiency": self._score_invalidation_efficiency(risk, atr),
            "fee_adjusted_rr": self._score_rr(fee_adjusted_rr),
            "spread_quality": self._score_spread(spread, risk),
            "volatility_sufficiency": self._score_volatility(atr),
        }
        signal_score = round(sum(component_scores.values()), 4)
        late_reasons = self._late_entry_reasons(reclaim_distance, atr, gross_rr)
        late_entry = bool(late_reasons)

        components: Dict[str, Any] = {
            **component_scores,
            "invalidation_distance": risk,
            "invalidation_distance_atr": risk / atr if atr > 0.0 else None,
            "target_distance": raw_reward,
            "gross_rr": gross_rr,
            "fee_adjusted_rr_value": fee_adjusted_rr,
            "round_turn_cost": round_turn_cost,
            "spread": spread,
            "atr": atr,
            "swept_level": swept_level,
            "sweep_extreme": sweep_extreme,
            "sweep_depth": sweep_depth,
            "reclaim_distance": reclaim_distance,
            "reclaim_distance_atr": reclaim_distance / atr if atr > 0.0 else None,
            "late_entry_reasons": late_reasons,
        }
        reason = f"LIQUIDITY_SWEEP_RECLAIM_{direction.upper()}"

        return TradeOpportunity(
            symbol=str(symbol),
            timeframe=str(timeframe),
            direction=direction,
            entry_price=float(entry),
            invalidation_price=float(invalidation),
            target_reference_price=float(target),
            signal_score=signal_score,
            components=components,
            reason=reason,
            detected_at_utc=detected_at_utc,
            late_entry=late_entry,
        )

    def _score_sweep_quality(self, sweep_depth: float, atr: float) -> float:
        depth_atr = sweep_depth / atr if atr > 0.0 else 0.0
        return round(min(25.0, 10.0 + (15.0 * min(depth_atr / 0.5, 1.0))), 4)

    def _score_reclaim_strength(
        self,
        *,
        direction: Direction,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        swept_level: float,
        atr: float,
    ) -> float:
        candle_range = max(high_price - low_price, 0.0)
        body = abs(close_price - open_price)
        body_ratio = body / candle_range if candle_range > 0.0 else 0.0
        reclaim_atr = abs(close_price - swept_level) / atr if atr > 0.0 else 0.0
        directional_close = close_price > open_price if direction == "long" else close_price < open_price
        directional_bonus = 4.0 if directional_close else 0.0
        body_score = 8.0 * min(body_ratio / 0.6, 1.0)
        reclaim_score = 8.0 * min(reclaim_atr / 0.5, 1.0)
        return round(min(20.0, directional_bonus + body_score + reclaim_score), 4)

    def _score_invalidation_efficiency(self, risk: float, atr: float) -> float:
        risk_atr = risk / atr if atr > 0.0 else float("inf")
        if risk_atr <= 0.75:
            return 20.0
        if risk_atr >= 2.5:
            return 0.0
        return round(20.0 * ((2.5 - risk_atr) / 1.75), 4)

    def _score_rr(self, fee_adjusted_rr: float) -> float:
        if fee_adjusted_rr <= 0.0:
            return 0.0
        return round(min(20.0, 20.0 * min(fee_adjusted_rr / 2.0, 1.0)), 4)

    def _score_spread(self, spread: float, risk: float) -> float:
        if spread <= 0.0:
            return 10.0
        if risk <= 0.0:
            return 0.0
        ratio = spread / risk
        if ratio <= 0.02:
            return 10.0
        if ratio >= 0.12:
            return 0.0
        return round(10.0 * ((0.12 - ratio) / 0.10), 4)

    def _score_volatility(self, atr: float) -> float:
        if self.min_atr <= 0.0:
            return 5.0
        return 5.0 if atr >= self.min_atr else round(5.0 * max(0.0, atr / self.min_atr), 4)

    def _late_entry_reasons(self, reclaim_distance: float, atr: float, gross_rr: float) -> List[str]:
        reasons: List[str] = []
        if atr > 0.0 and self.late_entry_atr_mult > 0.0 and reclaim_distance > atr * self.late_entry_atr_mult:
            reasons.append("reclaim_extended_from_swept_level")
        if gross_rr < self.late_entry_min_rr:
            reasons.append("gross_rr_below_late_entry_floor")
        return reasons

    def _coerce_rows(self, bars: Any) -> List[Dict[str, Any]]:
        if bars is None:
            return []
        if pd is not None and isinstance(bars, pd.DataFrame):
            source = bars.to_dict("records")
        elif isinstance(bars, Iterable) and not isinstance(bars, (str, bytes, Mapping)):
            source = list(bars)
        else:
            return []

        rows: List[Dict[str, Any]] = []
        for row in source:
            if not isinstance(row, Mapping):
                continue
            parsed = self._parse_ohlc(row)
            if parsed is not None:
                rows.append(parsed)
        return rows

    def _parse_ohlc(self, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        parsed: Dict[str, Any] = {}
        for key in ("open", "high", "low", "close"):
            value = self._float_or_none(row.get(key))
            if value is None:
                return None
            parsed[key] = value
        for time_key in ("time", "timestamp", "datetime"):
            if time_key in row:
                parsed["time"] = row.get(time_key)
                break
        return parsed

    def _atr(self, rows: List[Dict[str, Any]], period: int) -> Optional[float]:
        if len(rows) < period + 1:
            return None
        true_ranges: List[float] = []
        for idx in range(1, len(rows)):
            high = float(rows[idx]["high"])
            low = float(rows[idx]["low"])
            prev_close = float(rows[idx - 1]["close"])
            true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        recent = true_ranges[-period:]
        if not recent:
            return None
        atr = sum(recent) / float(len(recent))
        return atr if math.isfinite(atr) and atr > 0.0 else None

    def _average_range(self, rows: List[Dict[str, Any]]) -> Optional[float]:
        ranges = [float(row["high"]) - float(row["low"]) for row in rows]
        ranges = [value for value in ranges if value > 0.0 and math.isfinite(value)]
        if not ranges:
            return None
        return sum(ranges) / float(len(ranges))

    def _bar_time(self, row: Mapping[str, Any]) -> Optional[datetime]:
        if "time" not in row:
            return None
        return self._coerce_datetime(row.get("time"))

    def _coerce_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if pd is not None:
            try:
                parsed = pd.to_datetime(value, utc=True, errors="coerce")
            except Exception:
                return None
            if parsed is not None and not pd.isna(parsed):
                return parsed.to_pydatetime().astimezone(timezone.utc)
        return None

    def _float_or_none(self, value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    def _positive_float(self, value: Any, *, default: float) -> float:
        parsed = self._float_or_none(value)
        if parsed is None:
            return max(0.0, float(default))
        return max(0.0, parsed)
