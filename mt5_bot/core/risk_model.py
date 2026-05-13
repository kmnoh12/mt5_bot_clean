from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeeAwareRiskInput:
    symbol: str
    entry_price: float
    stop_price: float
    direction: str
    lot: float
    spread: float
    commission_per_lot: float
    expected_slippage_points: float
    tick_size: float
    tick_value: float
    contract_size: float
    take_profit_price: Optional[float] = None
    target_net_loss_usd: Optional[float] = None
    hard_max_net_loss_usd: Optional[float] = None


@dataclass(frozen=True)
class FeeAwareRiskResult:
    symbol: str
    estimated_gross_loss_usd: float
    estimated_cost_usd: float
    estimated_net_loss_usd: float
    estimated_net_profit_at_tp_usd: Optional[float]
    fee_adjusted_rr: Optional[float]
    hard_max_loss_pass: bool


class FeeAwareRiskModel:
    """Pure-Python fee-aware risk estimator.

    ``spread`` is treated as a price distance. ``expected_slippage_points`` is
    converted to price distance with ``tick_size``. ``tick_value`` is the USD
    value of one tick for one lot.
    """

    VALID_DIRECTIONS = {"long", "short", "buy", "sell"}

    def estimate(self, risk_input: FeeAwareRiskInput) -> FeeAwareRiskResult:
        direction = self._normalize_direction(risk_input.direction)
        self._validate_positive("lot", risk_input.lot)
        self._validate_positive("tick_size", risk_input.tick_size)
        self._validate_positive("tick_value", risk_input.tick_value)
        self._validate_non_negative("spread", risk_input.spread)
        self._validate_non_negative("commission_per_lot", risk_input.commission_per_lot)
        self._validate_non_negative("expected_slippage_points", risk_input.expected_slippage_points)

        gross_loss = self._gross_loss_usd(
            entry_price=risk_input.entry_price,
            stop_price=risk_input.stop_price,
            direction=direction,
            lot=risk_input.lot,
            tick_size=risk_input.tick_size,
            tick_value=risk_input.tick_value,
        )
        costs = self._cost_usd(
            lot=risk_input.lot,
            spread=risk_input.spread,
            commission_per_lot=risk_input.commission_per_lot,
            expected_slippage_points=risk_input.expected_slippage_points,
            tick_size=risk_input.tick_size,
            tick_value=risk_input.tick_value,
        )
        net_loss = gross_loss + costs

        net_profit_at_tp: Optional[float] = None
        fee_adjusted_rr: Optional[float] = None
        if risk_input.take_profit_price is not None:
            gross_profit = self._gross_profit_usd(
                entry_price=risk_input.entry_price,
                take_profit_price=risk_input.take_profit_price,
                direction=direction,
                lot=risk_input.lot,
                tick_size=risk_input.tick_size,
                tick_value=risk_input.tick_value,
            )
            net_profit_at_tp = gross_profit - costs
            fee_adjusted_rr = net_profit_at_tp / net_loss if net_loss > 0 else None

        hard_max = risk_input.hard_max_net_loss_usd
        hard_max_pass = True if hard_max is None else net_loss <= float(hard_max) + 1e-12

        return FeeAwareRiskResult(
            symbol=str(risk_input.symbol),
            estimated_gross_loss_usd=gross_loss,
            estimated_cost_usd=costs,
            estimated_net_loss_usd=net_loss,
            estimated_net_profit_at_tp_usd=net_profit_at_tp,
            fee_adjusted_rr=fee_adjusted_rr,
            hard_max_loss_pass=hard_max_pass,
        )

    @classmethod
    def _normalize_direction(cls, direction: str) -> str:
        normalized = str(direction or "").strip().lower()
        if normalized not in cls.VALID_DIRECTIONS:
            raise ValueError(f"unsupported direction: {direction!r}")
        return "long" if normalized in {"long", "buy"} else "short"

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def _price_distance_to_usd(distance: float, lot: float, tick_size: float, tick_value: float) -> float:
        return (float(distance) / float(tick_size)) * float(tick_value) * float(lot)

    def _gross_loss_usd(
        self,
        *,
        entry_price: float,
        stop_price: float,
        direction: str,
        lot: float,
        tick_size: float,
        tick_value: float,
    ) -> float:
        distance = float(entry_price) - float(stop_price) if direction == "long" else float(stop_price) - float(entry_price)
        if distance <= 0.0:
            raise ValueError("stop_price must be adverse to entry_price for direction")
        return self._price_distance_to_usd(distance, lot, tick_size, tick_value)

    def _gross_profit_usd(
        self,
        *,
        entry_price: float,
        take_profit_price: float,
        direction: str,
        lot: float,
        tick_size: float,
        tick_value: float,
    ) -> float:
        distance = (
            float(take_profit_price) - float(entry_price)
            if direction == "long"
            else float(entry_price) - float(take_profit_price)
        )
        if distance <= 0.0:
            raise ValueError("take_profit_price must be favorable to entry_price for direction")
        return self._price_distance_to_usd(distance, lot, tick_size, tick_value)

    def _cost_usd(
        self,
        *,
        lot: float,
        spread: float,
        commission_per_lot: float,
        expected_slippage_points: float,
        tick_size: float,
        tick_value: float,
    ) -> float:
        spread_cost = self._price_distance_to_usd(spread, lot, tick_size, tick_value)
        slippage_distance = float(expected_slippage_points) * float(tick_size)
        slippage_cost = self._price_distance_to_usd(slippage_distance, lot, tick_size, tick_value)
        commission_cost = float(commission_per_lot) * float(lot) * 2.0
        return spread_cost + slippage_cost + commission_cost
