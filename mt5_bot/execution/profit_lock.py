from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProfitLockDecision:
    should_modify: bool
    reason: str
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    lock_net_profit: Optional[float] = None
    target_net_profit: Optional[float] = None
    trigger_net_profit: Optional[float] = None
    net_unrealized_pnl: Optional[float] = None


class ProfitLockTrailingManager:
    DEFAULT_STAGES = (
        (30.0, 15.0, 45.0),
        (20.0, 10.0, 30.0),
        (10.0, 5.0, 20.0),
        (5.0, 2.0, None),
        (3.0, 1.0, None),
        (2.0, 0.0, None),
    )

    def __init__(
        self,
        *,
        min_seconds_between_sltp_updates: float = 10.0,
        stages: Any = None,
    ) -> None:
        self.min_seconds_between_sltp_updates = max(0.0, float(min_seconds_between_sltp_updates))
        self.stages = tuple(stages or self.DEFAULT_STAGES)
        self._last_update_by_ticket: Dict[str, float] = {}

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
    def _gross_pnl(direction: str, entry_price: float, current_price: float, units: float) -> float:
        if direction == "LONG":
            return (current_price - entry_price) * units
        return (entry_price - current_price) * units

    @staticmethod
    def _price_for_net_profit(
        direction: str,
        entry_price: float,
        net_profit: float,
        units: float,
        exit_cost: float,
    ) -> float:
        delta = (net_profit + exit_cost) / units
        if direction == "LONG":
            return entry_price + delta
        return entry_price - delta

    @classmethod
    def _min_stop_distance(cls, symbol_spec: Any) -> float:
        if symbol_spec is None:
            return 0.0
        point = cls._finite_float(cls._get(symbol_spec, "point"))
        if point is None or point <= 0:
            return 0.0
        stops = cls._finite_float(cls._get(symbol_spec, "trade_stops_level")) or 0.0
        freeze = cls._finite_float(cls._get(symbol_spec, "trade_freeze_level")) or 0.0
        return max(0.0, stops, freeze) * point

    @staticmethod
    def _ticket(position: Any) -> str:
        ticket = ProfitLockTrailingManager._get(position, "ticket")
        return str(ticket if ticket is not None else id(position))

    def mark_updated(self, ticket: Any, now: Any) -> None:
        timestamp = self._finite_float(now)
        if timestamp is not None:
            self._last_update_by_ticket[str(ticket)] = timestamp

    def _stage_for_pnl(self, net_unrealized_pnl: float) -> Optional[tuple[float, float, Optional[float]]]:
        for trigger, lock, target in self.stages:
            if net_unrealized_pnl + 1e-12 >= float(trigger):
                return float(trigger), float(lock), None if target is None else float(target)
        return None

    def evaluate(
        self,
        *,
        position: Any,
        current_price: Any,
        now: Any = None,
        symbol_spec: Any = None,
        contract_size: Any = 1.0,
        estimated_exit_cost: Any = 0.0,
        net_unrealized_pnl: Any = None,
        record_update: bool = True,
    ) -> ProfitLockDecision:
        direction = self._direction(
            self._get(position, "side", self._get(position, "direction", self._get(position, "type")))
        )
        entry = self._finite_float(self._get(position, "price_open", self._get(position, "entry_price")))
        price = self._finite_float(current_price)
        volume = self._finite_float(self._get(position, "volume", self._get(position, "lot", self._get(position, "position_size"))))
        cs = self._finite_float(contract_size)
        exit_cost = max(0.0, self._finite_float(estimated_exit_cost) or 0.0)

        if direction is None:
            return ProfitLockDecision(False, "INVALID_DIRECTION")
        if entry is None or entry <= 0 or price is None or price <= 0:
            return ProfitLockDecision(False, "INVALID_PRICE")
        if volume is None or volume <= 0 or cs is None or cs <= 0:
            return ProfitLockDecision(False, "INVALID_POSITION_SIZE")

        units = volume * cs
        net_pnl = self._finite_float(net_unrealized_pnl)
        if net_pnl is None:
            net_pnl = self._gross_pnl(direction, entry, price, units) - exit_cost

        stage = self._stage_for_pnl(net_pnl)
        if stage is None:
            return ProfitLockDecision(False, "NO_THRESHOLD", net_unrealized_pnl=net_pnl)
        trigger, lock_profit, target_profit = stage

        ticket = self._ticket(position)
        now_ts = self._finite_float(now)
        last_update = self._last_update_by_ticket.get(ticket)
        if now_ts is not None and last_update is not None:
            elapsed = now_ts - last_update
            if elapsed + 1e-12 < self.min_seconds_between_sltp_updates:
                return ProfitLockDecision(
                    False,
                    "MIN_UPDATE_INTERVAL",
                    trigger_net_profit=trigger,
                    net_unrealized_pnl=net_pnl,
                )

        desired_sl = self._price_for_net_profit(direction, entry, lock_profit, units, exit_cost)
        desired_tp = (
            None
            if target_profit is None
            else self._price_for_net_profit(direction, entry, target_profit, units, exit_cost)
        )

        existing_sl = self._finite_float(self._get(position, "sl"))
        existing_tp = self._finite_float(self._get(position, "tp"))

        sl_price = desired_sl
        if existing_sl is not None:
            if direction == "LONG":
                sl_price = max(existing_sl, desired_sl)
            else:
                sl_price = min(existing_sl, desired_sl)

        tp_price = desired_tp
        if desired_tp is not None and existing_tp is not None:
            if abs(desired_tp - existing_tp) <= 1e-9:
                tp_price = existing_tp

        if direction == "LONG":
            if sl_price >= price:
                return ProfitLockDecision(False, "SL_NOT_BELOW_MARKET", trigger_net_profit=trigger, net_unrealized_pnl=net_pnl)
            if tp_price is not None and tp_price <= price:
                return ProfitLockDecision(False, "TP_NOT_ABOVE_MARKET", trigger_net_profit=trigger, net_unrealized_pnl=net_pnl)
        else:
            if sl_price <= price:
                return ProfitLockDecision(False, "SL_NOT_ABOVE_MARKET", trigger_net_profit=trigger, net_unrealized_pnl=net_pnl)
            if tp_price is not None and tp_price >= price:
                return ProfitLockDecision(False, "TP_NOT_BELOW_MARKET", trigger_net_profit=trigger, net_unrealized_pnl=net_pnl)

        min_distance = self._min_stop_distance(symbol_spec)
        if min_distance > 0:
            if abs(price - sl_price) + 1e-12 < min_distance:
                return ProfitLockDecision(False, "BROKER_STOP_LEVEL_SL", trigger_net_profit=trigger, net_unrealized_pnl=net_pnl)
            if tp_price is not None and abs(tp_price - price) + 1e-12 < min_distance:
                return ProfitLockDecision(False, "BROKER_STOP_LEVEL_TP", trigger_net_profit=trigger, net_unrealized_pnl=net_pnl)

        sl_changed = existing_sl is None or abs(sl_price - existing_sl) > 1e-9
        tp_changed = desired_tp is not None and (existing_tp is None or abs(desired_tp - existing_tp) > 1e-9)
        if not sl_changed and not tp_changed:
            return ProfitLockDecision(
                False,
                "NO_FORWARD_PROGRESS",
                sl_price=sl_price,
                tp_price=tp_price,
                lock_net_profit=lock_profit,
                target_net_profit=target_profit,
                trigger_net_profit=trigger,
                net_unrealized_pnl=net_pnl,
            )

        if record_update and now_ts is not None:
            self._last_update_by_ticket[ticket] = now_ts

        return ProfitLockDecision(
            True,
            "OK",
            sl_price=sl_price,
            tp_price=tp_price,
            lock_net_profit=lock_profit,
            target_net_profit=target_profit,
            trigger_net_profit=trigger,
            net_unrealized_pnl=net_pnl,
        )
