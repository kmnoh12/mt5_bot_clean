from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class InitialExitPlan:
    sl_price: Optional[float]
    tp_price: Optional[float]
    expected_net_loss_at_sl: float
    expected_net_profit_at_tp: float
    fee_adjusted_rr: float
    passed: bool
    reason: str


class InitialExitPlanner:
    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _get(source: Any, key: str, default: Any = None) -> Any:
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    @staticmethod
    def _direction(value: Any) -> Optional[str]:
        value = getattr(value, "value", value)
        text = str(value or "").strip().upper()
        if text in {"BUY", "LONG"}:
            return "LONG"
        if text in {"SELL", "SHORT"}:
            return "SHORT"
        return None

    @staticmethod
    def _costs_from_model(
        risk_model: Optional[Callable[..., Any]],
        *,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        position_size: float,
        contract_size: float,
    ) -> Dict[str, float]:
        if risk_model is None:
            return {}
        try:
            raw = risk_model(
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                position_size=position_size,
                contract_size=contract_size,
            )
        except TypeError:
            raw = risk_model(entry_price, sl_price, tp_price, position_size)
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, float] = {}
        for key in (
            "entry_cost",
            "entry_cost_usd",
            "exit_cost",
            "exit_cost_usd",
            "round_trip_cost",
            "round_trip_cost_usd",
        ):
            value = InitialExitPlanner._finite_float(raw.get(key))
            if value is not None:
                out[key] = max(0.0, value)
        return out

    @staticmethod
    def _gross_pnl(direction: str, entry_price: float, exit_price: float, units: float) -> float:
        if direction == "LONG":
            return (exit_price - entry_price) * units
        return (entry_price - exit_price) * units

    @staticmethod
    def _net_pnl(direction: str, entry_price: float, exit_price: float, units: float, costs: float) -> float:
        return InitialExitPlanner._gross_pnl(direction, entry_price, exit_price, units) - costs

    @staticmethod
    def _price_for_net_profit(
        direction: str,
        entry_price: float,
        desired_net_profit: float,
        units: float,
        costs: float,
    ) -> float:
        delta = (desired_net_profit + costs) / units
        if direction == "LONG":
            return entry_price + delta
        return entry_price - delta

    @staticmethod
    def _min_stop_distance(symbol_spec: Any) -> float:
        if symbol_spec is None:
            return 0.0
        point = InitialExitPlanner._finite_float(InitialExitPlanner._get(symbol_spec, "point"))
        if point is None or point <= 0:
            return 0.0
        stops = InitialExitPlanner._finite_float(InitialExitPlanner._get(symbol_spec, "trade_stops_level")) or 0.0
        freeze = InitialExitPlanner._finite_float(InitialExitPlanner._get(symbol_spec, "trade_freeze_level")) or 0.0
        return max(0.0, stops, freeze) * point

    def plan(
        self,
        *,
        opportunity: Any = None,
        direction: Any = None,
        entry_price: Any = None,
        invalidation_price: Any = None,
        target_reference_price: Any = None,
        position_size: Any = None,
        lot: Any = None,
        contract_size: Any = 1.0,
        estimated_entry_cost: Any = 0.0,
        estimated_exit_cost: Any = 0.0,
        estimated_round_trip_cost: Any = None,
        risk_model: Optional[Callable[..., Any]] = None,
        min_reward_to_net_risk_ratio: Any = 3.0,
        target_max_loss: Any = None,
        hard_max_loss: Any = None,
        min_tp_net_profit: Any = 0.0,
        preferred_tp_net_profit: Any = None,
        symbol_spec: Any = None,
    ) -> InitialExitPlan:
        raw_direction = direction if direction is not None else self._get(opportunity, "direction")
        side = self._direction(raw_direction)
        entry = self._finite_float(entry_price if entry_price is not None else self._get(opportunity, "entry_price"))
        sl = self._finite_float(
            invalidation_price if invalidation_price is not None else self._get(opportunity, "invalidation_price")
        )
        tp_reference = self._finite_float(
            target_reference_price
            if target_reference_price is not None
            else self._get(opportunity, "target_reference_price")
        )
        size = self._finite_float(position_size if position_size is not None else lot)
        cs = self._finite_float(contract_size)

        if side is None:
            return InitialExitPlan(None, None, 0.0, 0.0, 0.0, False, "INVALID_DIRECTION")
        if entry is None or entry <= 0 or sl is None or sl <= 0:
            return InitialExitPlan(None, None, 0.0, 0.0, 0.0, False, "INVALID_PRICE")
        if size is None or size <= 0 or cs is None or cs <= 0:
            return InitialExitPlan(None, None, 0.0, 0.0, 0.0, False, "INVALID_POSITION_SIZE")
        if side == "LONG" and sl >= entry:
            return InitialExitPlan(sl, None, 0.0, 0.0, 0.0, False, "INVALID_SL_SIDE")
        if side == "SHORT" and sl <= entry:
            return InitialExitPlan(sl, None, 0.0, 0.0, 0.0, False, "INVALID_SL_SIDE")

        units = size * cs
        entry_cost = max(0.0, self._finite_float(estimated_entry_cost) or 0.0)
        exit_cost = max(0.0, self._finite_float(estimated_exit_cost) or 0.0)
        round_trip_cost = self._finite_float(estimated_round_trip_cost)
        base_costs = max(0.0, round_trip_cost) if round_trip_cost is not None else entry_cost + exit_cost

        min_rr = max(0.0, self._finite_float(min_reward_to_net_risk_ratio) or 0.0)
        net_loss_preview = -self._net_pnl(side, entry, sl, units, base_costs)
        desired_profit = max(0.0, self._finite_float(min_tp_net_profit) or 0.0, net_loss_preview * min_rr)
        preferred_profit = self._finite_float(preferred_tp_net_profit)
        if preferred_profit is not None:
            desired_profit = max(desired_profit, preferred_profit)

        if tp_reference is None:
            tp = self._price_for_net_profit(side, entry, desired_profit, units, base_costs)
        else:
            tp = tp_reference

        model_costs = self._costs_from_model(
            risk_model,
            direction=side,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            position_size=size,
            contract_size=cs,
        )
        if model_costs:
            model_round_trip = model_costs.get("round_trip_cost", model_costs.get("round_trip_cost_usd"))
            if model_round_trip is not None:
                base_costs = model_round_trip
            else:
                base_costs = (
                    model_costs.get("entry_cost", model_costs.get("entry_cost_usd", entry_cost))
                    + model_costs.get("exit_cost", model_costs.get("exit_cost_usd", exit_cost))
                )
            if tp_reference is None:
                tp = self._price_for_net_profit(side, entry, desired_profit, units, base_costs)

        if side == "LONG" and tp <= entry:
            return InitialExitPlan(sl, tp, 0.0, 0.0, 0.0, False, "INVALID_TP_SIDE")
        if side == "SHORT" and tp >= entry:
            return InitialExitPlan(sl, tp, 0.0, 0.0, 0.0, False, "INVALID_TP_SIDE")

        min_distance = self._min_stop_distance(symbol_spec)
        if min_distance > 0:
            if abs(entry - sl) + 1e-12 < min_distance:
                return InitialExitPlan(sl, tp, 0.0, 0.0, 0.0, False, "BROKER_STOP_LEVEL_SL")
            if abs(tp - entry) + 1e-12 < min_distance:
                return InitialExitPlan(sl, tp, 0.0, 0.0, 0.0, False, "BROKER_STOP_LEVEL_TP")

        net_loss = -self._net_pnl(side, entry, sl, units, base_costs)
        net_profit = self._net_pnl(side, entry, tp, units, base_costs)
        rr = net_profit / net_loss if net_loss > 0 else float("inf")

        hard_loss = self._finite_float(hard_max_loss if hard_max_loss is not None else target_max_loss)
        min_profit = max(0.0, self._finite_float(min_tp_net_profit) or 0.0)

        if hard_loss is not None and net_loss - 1e-12 > hard_loss:
            return InitialExitPlan(sl, tp, net_loss, net_profit, rr, False, "HARD_MAX_LOSS_EXCEEDED")
        if net_profit + 1e-12 < min_profit:
            return InitialExitPlan(sl, tp, net_loss, net_profit, rr, False, "TP_NET_PROFIT_TOO_LOW")
        if rr + 1e-12 < min_rr:
            return InitialExitPlan(sl, tp, net_loss, net_profit, rr, False, "RR_TOO_LOW")

        return InitialExitPlan(sl, tp, net_loss, net_profit, rr, True, "OK")
