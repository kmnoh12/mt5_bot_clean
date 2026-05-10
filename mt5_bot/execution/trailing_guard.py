from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from core.models import Position, Side


@dataclass(frozen=True)
class TrailingGuardSignal:
    ticket: int
    symbol: str
    peak_pnl_usd: float
    current_pnl_usd: float
    trigger_pnl_usd: float
    drawdown_usd: float
    threshold_usd: float
    reason: str


@dataclass(frozen=True)
class BreakEvenSlSignal:
    ticket: int
    symbol: str
    current_pnl_usd: float
    peak_pnl_usd: float
    activation_pnl_usd: float
    lock_pnl_usd: float
    desired_sl: float
    existing_sl: Optional[float]
    stage: str
    reason: str


class DynamicTrailingProfitGuard:
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = True
        self.retain_ratio = 0.5
        self.min_activation_profit_usd = 3.0
        self.stage_a_activation_usd = 3.0
        self.stage_b_activation_usd = 15.0
        self.stage_b_retain_ratio = 0.10
        self.stage_c_activation_usd = 35.0
        self.stage_c_retain_ratio = 0.35
        self.break_even_enabled = True
        self.break_even_activation_profit_usd = 3.0
        self.break_even_lock_profit_usd = 0.2
        self.break_even_sync_sl = True
        self.min_hold_seconds_for_exit = 300.0
        self.min_drawdown_usd_for_exit = 2.5
        self.min_breach_count_for_exit = 2
        self._peak_by_ticket: Dict[str, float] = {}
        self._breach_count_by_ticket: Dict[str, int] = {}

        self.update_config(config)
        self._restore_snapshot(snapshot or {})

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        peaks = snapshot.get("peak_by_ticket")
        restored: Dict[str, float] = {}
        if isinstance(peaks, dict):
            for key, value in peaks.items():
                val = self._as_float(value)
                if val is None:
                    continue
                restored[str(key)] = val
        self._peak_by_ticket = restored
        breaches = snapshot.get("breach_count_by_ticket")
        restored_breaches: Dict[str, int] = {}
        if isinstance(breaches, dict):
            for key, value in breaches.items():
                try:
                    restored_breaches[str(key)] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
        self._breach_count_by_ticket = restored_breaches

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        retain_ratio = self._as_float(cfg.get("retain_ratio", 0.5))
        min_activation = self._as_float(cfg.get("min_activation_profit_usd", cfg.get("min_peak_profit_usd", 3.0)))
        if retain_ratio is None:
            retain_ratio = 0.5
        self.retain_ratio = max(0.0, min(1.0, float(retain_ratio)))
        self.min_activation_profit_usd = max(0.0, float(min_activation or 0.0))
        stage_a = self._as_float(cfg.get("stage_a_activation_usd", self.min_activation_profit_usd or 3.0))
        stage_b = self._as_float(cfg.get("stage_b_activation_usd", cfg.get("min_activation_profit_usd", 15.0)))
        stage_c = self._as_float(cfg.get("stage_c_activation_usd", 35.0))
        stage_b_ratio = self._as_float(cfg.get("stage_b_retain_ratio", 0.10))
        stage_c_ratio = self._as_float(cfg.get("stage_c_retain_ratio", 0.35))
        self.stage_a_activation_usd = max(0.0, float(stage_a or 3.0))
        self.stage_b_activation_usd = max(self.stage_a_activation_usd, float(stage_b or self.stage_a_activation_usd))
        self.stage_c_activation_usd = max(self.stage_b_activation_usd, float(stage_c or self.stage_b_activation_usd))
        self.stage_b_retain_ratio = max(0.0, min(1.0, float(stage_b_ratio if stage_b_ratio is not None else 0.10)))
        self.stage_c_retain_ratio = max(0.0, min(1.0, float(stage_c_ratio if stage_c_ratio is not None else 0.35)))
        self.break_even_enabled = bool(cfg.get("break_even_enabled", True))
        be_activation = self._as_float(cfg.get("break_even_activation_profit_usd", self.stage_a_activation_usd))
        be_lock = self._as_float(cfg.get("break_even_lock_profit_usd", 0.2))
        self.break_even_activation_profit_usd = max(0.0, float(be_activation or 0.0))
        self.break_even_lock_profit_usd = max(0.0, float(be_lock or 0.0))
        self.break_even_sync_sl = bool(cfg.get("break_even_sync_sl", True))
        min_hold_seconds = self._as_float(cfg.get("min_hold_seconds_for_exit", 300.0))
        min_drawdown_usd = self._as_float(cfg.get("min_drawdown_usd_for_exit", 2.5))
        breach_count = cfg.get("min_breach_count_for_exit", 2)
        self.min_hold_seconds_for_exit = max(0.0, float(min_hold_seconds or 0.0))
        self.min_drawdown_usd_for_exit = max(0.0, float(min_drawdown_usd or 0.0))
        try:
            self.min_breach_count_for_exit = max(1, int(breach_count))
        except (TypeError, ValueError):
            self.min_breach_count_for_exit = 2

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "retain_ratio": float(self.retain_ratio),
            "min_activation_profit_usd": float(self.min_activation_profit_usd),
            "stage_a_activation_usd": float(self.stage_a_activation_usd),
            "stage_b_activation_usd": float(self.stage_b_activation_usd),
            "stage_b_retain_ratio": float(self.stage_b_retain_ratio),
            "stage_c_activation_usd": float(self.stage_c_activation_usd),
            "stage_c_retain_ratio": float(self.stage_c_retain_ratio),
            "break_even_enabled": bool(self.break_even_enabled),
            "break_even_activation_profit_usd": float(self.break_even_activation_profit_usd),
            "break_even_lock_profit_usd": float(self.break_even_lock_profit_usd),
            "break_even_sync_sl": bool(self.break_even_sync_sl),
            "min_hold_seconds_for_exit": float(self.min_hold_seconds_for_exit),
            "min_drawdown_usd_for_exit": float(self.min_drawdown_usd_for_exit),
            "min_breach_count_for_exit": int(self.min_breach_count_for_exit),
            "peak_by_ticket": dict(self._peak_by_ticket),
            "breach_count_by_ticket": dict(self._breach_count_by_ticket),
        }

    @staticmethod
    def _ticket_key(position: Position) -> str:
        return str(int(position.ticket))

    def _extract_floating_pnl_usd(self, position: Position) -> Optional[float]:
        metadata = position.metadata if isinstance(position.metadata, dict) else {}
        for key in ("floating_pnl", "pnl", "profit", "unrealized_pnl"):
            if key not in metadata:
                continue
            value = self._as_float(metadata.get(key))
            if value is not None:
                return value
        return None

    def _extract_net_floating_pnl_usd(self, position: Position) -> Optional[float]:
        current_pnl = self._extract_floating_pnl_usd(position)
        if current_pnl is None:
            return None
        metadata = position.metadata if isinstance(position.metadata, dict) else {}
        swap = self._as_float(metadata.get("swap"))
        commission = self._as_float(metadata.get("commission"))
        return float(current_pnl + (swap or 0.0) + (commission or 0.0))

    def _track_and_get_pnl_state(self, position: Position) -> Optional[tuple[float, float]]:
        current_pnl = self._extract_net_floating_pnl_usd(position)
        if current_pnl is None:
            return None
        ticket_key = self._ticket_key(position)
        peak_pnl = self._peak_by_ticket.get(ticket_key, current_pnl)
        if current_pnl > peak_pnl:
            peak_pnl = current_pnl
        self._peak_by_ticket[ticket_key] = peak_pnl
        return current_pnl, peak_pnl

    @staticmethod
    def _compute_lock_price(position: Position, lock_pnl: float, contract_size: float) -> Optional[float]:
        try:
            volume = float(position.volume)
            entry = float(position.price_open)
            cs = max(1e-9, float(contract_size))
        except (TypeError, ValueError):
            return None
        denom = volume * cs
        if denom <= 0:
            return None
        delta = float(lock_pnl) / denom
        if position.side == Side.BUY:
            return entry + delta
        return entry - delta

    def evaluate_break_even_sl(self, position: Position, contract_size: float) -> Optional[BreakEvenSlSignal]:
        tracked = self._track_and_get_pnl_state(position)
        if tracked is None:
            return None
        _, peak_pnl = tracked
        if not self.break_even_enabled:
            return None
        if peak_pnl < self.break_even_activation_profit_usd:
            return None
        lock_price = self._compute_lock_price(
            position=position,
            lock_pnl=self.break_even_lock_profit_usd,
            contract_size=contract_size,
        )
        if lock_price is None:
            return None
        existing_sl = float(position.sl) if position.sl is not None else None
        if position.side == Side.BUY:
            desired_sl = max(existing_sl if existing_sl is not None else lock_price, lock_price)
        else:
            desired_sl = min(existing_sl if existing_sl is not None else lock_price, lock_price)
        if existing_sl is not None and abs(desired_sl - existing_sl) <= 1e-9:
            return None
        stage = "A_BE"
        if peak_pnl >= self.stage_c_activation_usd:
            stage = "C_TRAIL"
        elif peak_pnl >= self.stage_b_activation_usd:
            stage = "B_LOCK"
        return BreakEvenSlSignal(
            ticket=int(position.ticket),
            symbol=str(position.symbol),
            current_pnl_usd=float(tracked[0]),
            peak_pnl_usd=float(peak_pnl),
            activation_pnl_usd=float(self.break_even_activation_profit_usd),
            lock_pnl_usd=float(self.break_even_lock_profit_usd),
            desired_sl=float(desired_sl),
            existing_sl=existing_sl,
            stage=stage,
            reason=f"profit_lock_stage_{stage.lower()}",
        )

    def evaluate_profit_lock_sl(self, position: Position, contract_size: float) -> Optional[BreakEvenSlSignal]:
        tracked = self._track_and_get_pnl_state(position)
        if tracked is None:
            return None
        current_pnl, peak_pnl = tracked
        if not self.enabled:
            return None
        if peak_pnl < self.stage_a_activation_usd:
            return None

        lock_pnl = self.break_even_lock_profit_usd
        stage = "A_BE"
        if peak_pnl >= self.stage_c_activation_usd:
            stage = "C_TRAIL"
            lock_pnl = peak_pnl * self.stage_c_retain_ratio
        elif peak_pnl >= self.stage_b_activation_usd:
            stage = "B_LOCK"
            lock_pnl = peak_pnl * self.stage_b_retain_ratio

        lock_price = self._compute_lock_price(position=position, lock_pnl=lock_pnl, contract_size=contract_size)
        if lock_price is None:
            return None
        existing_sl = float(position.sl) if position.sl is not None else None
        if position.side == Side.BUY:
            desired_sl = max(existing_sl if existing_sl is not None else lock_price, lock_price)
        else:
            desired_sl = min(existing_sl if existing_sl is not None else lock_price, lock_price)
        if existing_sl is not None and abs(desired_sl - existing_sl) <= 1e-9:
            return None
        return BreakEvenSlSignal(
            ticket=int(position.ticket),
            symbol=str(position.symbol),
            current_pnl_usd=float(current_pnl),
            peak_pnl_usd=float(peak_pnl),
            activation_pnl_usd=float(self.stage_a_activation_usd),
            lock_pnl_usd=float(lock_pnl),
            desired_sl=float(desired_sl),
            existing_sl=existing_sl,
            stage=stage,
            reason=f"profit_lock_stage_{stage.lower()}",
        )

    def evaluate_position(self, position: Position) -> Optional[TrailingGuardSignal]:
        tracked = self._track_and_get_pnl_state(position)
        if tracked is None:
            return None
        current_pnl, peak_pnl = tracked
        ticket_key = self._ticket_key(position)

        if not self.enabled:
            return None
        if peak_pnl < self.stage_b_activation_usd:
            self._breach_count_by_ticket[ticket_key] = 0
            return None

        retain_ratio = self.stage_c_retain_ratio if peak_pnl >= self.stage_c_activation_usd else self.stage_b_retain_ratio
        trigger_pnl = peak_pnl * retain_ratio
        if current_pnl >= trigger_pnl:
            self._breach_count_by_ticket[ticket_key] = 0
            return None

        threshold_usd = peak_pnl - trigger_pnl
        drawdown = peak_pnl - current_pnl
        if drawdown <= 0:
            self._breach_count_by_ticket[ticket_key] = 0
            return None
        if drawdown < self.min_drawdown_usd_for_exit:
            self._breach_count_by_ticket[ticket_key] = 0
            return None

        if position.time_open_utc is not None and self.min_hold_seconds_for_exit > 0:
            hold_seconds = (datetime.now(timezone.utc) - position.time_open_utc.astimezone(timezone.utc)).total_seconds()
            if hold_seconds < self.min_hold_seconds_for_exit:
                self._breach_count_by_ticket[ticket_key] = 0
                return None

        breach_count = int(self._breach_count_by_ticket.get(ticket_key, 0) or 0) + 1
        self._breach_count_by_ticket[ticket_key] = breach_count
        if breach_count < self.min_breach_count_for_exit:
            return None

        return TrailingGuardSignal(
            ticket=int(position.ticket),
            symbol=str(position.symbol),
            peak_pnl_usd=float(peak_pnl),
            current_pnl_usd=float(current_pnl),
            trigger_pnl_usd=float(trigger_pnl),
            drawdown_usd=float(drawdown),
            threshold_usd=float(threshold_usd),
            reason="profit_lock_drawdown_exit",
        )

    def drop_closed_positions(self, open_positions: Iterable[Position]) -> None:
        open_keys = {self._ticket_key(pos) for pos in open_positions}
        self._peak_by_ticket = {k: v for k, v in self._peak_by_ticket.items() if k in open_keys}
        self._breach_count_by_ticket = {k: v for k, v in self._breach_count_by_ticket.items() if k in open_keys}

    def clear_ticket(self, ticket: int) -> None:
        key = str(int(ticket))
        self._peak_by_ticket.pop(key, None)
        self._breach_count_by_ticket.pop(key, None)
