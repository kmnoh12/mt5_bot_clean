from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.risk_model import FeeAwareRiskInput, FeeAwareRiskModel


@dataclass(frozen=True)
class SymbolVolumeSpec:
    volume_min: float
    volume_step: float
    volume_max: float
    tick_size: float
    tick_value: float
    contract_size: float


@dataclass(frozen=True)
class NetRiskPositionSizeInput:
    symbol: str
    target_net_loss_usd: float
    hard_max_net_loss_usd: float
    entry_price: float
    stop_price: float
    direction: str
    symbol_spec: SymbolVolumeSpec
    spread: float
    commission_per_lot: float
    expected_slippage_points: float


@dataclass(frozen=True)
class NetRiskPositionSizeResult:
    recommended_lot: Optional[float]
    estimated_net_loss: Optional[float]
    passed: bool
    failure_reason: str


class NetRiskPositionSizer:
    MIN_LOT_RISK_EXCEEDS_HARD_MAX = "min_lot_risk_exceeds_hard_max"

    def __init__(self, risk_model: Optional[FeeAwareRiskModel] = None) -> None:
        self._risk_model = risk_model or FeeAwareRiskModel()

    def size(self, size_input: NetRiskPositionSizeInput) -> NetRiskPositionSizeResult:
        spec = size_input.symbol_spec
        self._validate_spec(spec)
        self._validate_positive("target_net_loss_usd", size_input.target_net_loss_usd)
        self._validate_positive("hard_max_net_loss_usd", size_input.hard_max_net_loss_usd)

        min_loss = self._net_loss_for_lot(size_input, spec.volume_min)
        if min_loss > float(size_input.hard_max_net_loss_usd) + 1e-12:
            return NetRiskPositionSizeResult(
                recommended_lot=None,
                estimated_net_loss=min_loss,
                passed=False,
                failure_reason=self.MIN_LOT_RISK_EXCEEDS_HARD_MAX,
            )

        per_lot_loss = self._net_loss_for_lot(size_input, 1.0)
        if per_lot_loss <= 0.0:
            return NetRiskPositionSizeResult(
                recommended_lot=None,
                estimated_net_loss=None,
                passed=False,
                failure_reason="non_positive_per_lot_risk",
            )

        target_lot = float(size_input.target_net_loss_usd) / per_lot_loss
        hard_max_lot = float(size_input.hard_max_net_loss_usd) / per_lot_loss
        candidate = self._nearest_step_lot(target_lot, spec)

        if candidate is None:
            return NetRiskPositionSizeResult(
                recommended_lot=None,
                estimated_net_loss=None,
                passed=False,
                failure_reason="no_valid_volume",
            )

        candidate_loss = self._net_loss_for_lot(size_input, candidate)
        if candidate_loss > float(size_input.hard_max_net_loss_usd) + 1e-12:
            candidate = self._floor_step_lot(min(target_lot, hard_max_lot), spec)
            if candidate is None:
                return NetRiskPositionSizeResult(
                    recommended_lot=None,
                    estimated_net_loss=min_loss,
                    passed=False,
                    failure_reason=self.MIN_LOT_RISK_EXCEEDS_HARD_MAX,
                )
            candidate_loss = self._net_loss_for_lot(size_input, candidate)

        if candidate_loss > float(size_input.hard_max_net_loss_usd) + 1e-12:
            return NetRiskPositionSizeResult(
                recommended_lot=None,
                estimated_net_loss=candidate_loss,
                passed=False,
                failure_reason="rounded_lot_risk_exceeds_hard_max",
            )

        return NetRiskPositionSizeResult(
            recommended_lot=candidate,
            estimated_net_loss=candidate_loss,
            passed=True,
            failure_reason="",
        )

    def _net_loss_for_lot(self, size_input: NetRiskPositionSizeInput, lot: float) -> float:
        result = self._risk_model.estimate(
            FeeAwareRiskInput(
                symbol=size_input.symbol,
                entry_price=size_input.entry_price,
                stop_price=size_input.stop_price,
                direction=size_input.direction,
                lot=lot,
                spread=size_input.spread,
                commission_per_lot=size_input.commission_per_lot,
                expected_slippage_points=size_input.expected_slippage_points,
                tick_size=size_input.symbol_spec.tick_size,
                tick_value=size_input.symbol_spec.tick_value,
                contract_size=size_input.symbol_spec.contract_size,
                target_net_loss_usd=size_input.target_net_loss_usd,
                hard_max_net_loss_usd=size_input.hard_max_net_loss_usd,
            )
        )
        return result.estimated_net_loss_usd

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")

    def _validate_spec(self, spec: SymbolVolumeSpec) -> None:
        self._validate_positive("volume_min", spec.volume_min)
        self._validate_positive("volume_step", spec.volume_step)
        self._validate_positive("volume_max", spec.volume_max)
        self._validate_positive("tick_size", spec.tick_size)
        self._validate_positive("tick_value", spec.tick_value)
        if float(spec.volume_max) < float(spec.volume_min):
            raise ValueError("volume_max must be greater than or equal to volume_min")

    def _nearest_step_lot(self, raw_lot: float, spec: SymbolVolumeSpec) -> Optional[float]:
        floor_lot = self._floor_step_lot(raw_lot, spec)
        ceil_lot = self._ceil_step_lot(raw_lot, spec)
        candidates = [lot for lot in (floor_lot, ceil_lot) if lot is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda lot: (abs(lot - raw_lot), lot))

    @staticmethod
    def _floor_step_lot(raw_lot: float, spec: SymbolVolumeSpec) -> Optional[float]:
        raw = min(float(raw_lot), float(spec.volume_max))
        steps = math.floor((raw - float(spec.volume_min)) / float(spec.volume_step) + 1e-12)
        if steps < 0:
            lot = float(spec.volume_min)
        else:
            lot = float(spec.volume_min) + steps * float(spec.volume_step)
        if lot > float(spec.volume_max) + 1e-12:
            return None
        return round(lot, 10)

    @staticmethod
    def _ceil_step_lot(raw_lot: float, spec: SymbolVolumeSpec) -> Optional[float]:
        raw = max(float(raw_lot), float(spec.volume_min))
        steps = math.ceil((raw - float(spec.volume_min)) / float(spec.volume_step) - 1e-12)
        lot = float(spec.volume_min) + steps * float(spec.volume_step)
        if lot > float(spec.volume_max) + 1e-12:
            return None
        return round(lot, 10)
