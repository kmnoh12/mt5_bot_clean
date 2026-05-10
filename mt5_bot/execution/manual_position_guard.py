from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from brokers.base import BrokerGateway
from core.models import Position, Side


@dataclass(frozen=True)
class ManualGuardEvent:
    event: str
    payload: Dict[str, Any]


class ManualPositionGuard:
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = True
        self.symbols: Set[str] = {"GOLD"}
        self.manual_magic_values: Set[int] = {0}
        self.retain_ratio = 0.8
        self.min_activation_profit_usd = 5.0
        self.breach_close = True
        self.sync_sl_to_lock_line = True
        self.close_retry_cooldown_seconds = 30.0
        self.block_strategy_for_protected_symbols = True

        self._peak_by_ticket: Dict[str, float] = {}
        self._last_close_attempt_by_ticket: Dict[str, float] = {}

        self.update_config(config)
        self._restore_snapshot(snapshot or {})

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    @staticmethod
    def _ticket_key(ticket: Any) -> str:
        try:
            return str(int(ticket))
        except Exception:
            return str(ticket)

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        peak = snapshot.get("peak_by_ticket")
        last_attempt = snapshot.get("last_close_attempt_by_ticket")

        if isinstance(peak, dict):
            for key, value in peak.items():
                parsed = self._as_float(value)
                if parsed is not None:
                    self._peak_by_ticket[str(key)] = parsed

        if isinstance(last_attempt, dict):
            for key, value in last_attempt.items():
                parsed = self._as_float(value)
                if parsed is not None:
                    self._last_close_attempt_by_ticket[str(key)] = parsed

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))

        symbols = cfg.get("symbols", ["GOLD"])
        normalized = []
        if isinstance(symbols, (list, tuple, set)):
            for item in symbols:
                text = str(item or "").strip().upper()
                if text and text not in normalized:
                    normalized.append(text)
        self.symbols = set(normalized) if normalized else {"GOLD"}

        raw_magic = cfg.get("manual_magic_values", [0])
        parsed_magic: Set[int] = set()
        if isinstance(raw_magic, (list, tuple, set)):
            for item in raw_magic:
                try:
                    parsed_magic.add(int(item))
                except (TypeError, ValueError):
                    continue
        self.manual_magic_values = parsed_magic if parsed_magic else {0}

        ratio = self._as_float(cfg.get("retain_ratio", 0.8))
        self.retain_ratio = max(0.0, min(1.0, ratio if ratio is not None else 0.8))
        min_activation = self._as_float(cfg.get("min_activation_profit_usd", 5.0))
        self.min_activation_profit_usd = max(0.0, min_activation if min_activation is not None else 5.0)
        self.breach_close = bool(cfg.get("breach_close", True))
        self.sync_sl_to_lock_line = bool(cfg.get("sync_sl_to_lock_line", True))
        retry = self._as_float(cfg.get("close_retry_cooldown_seconds", 30.0))
        self.close_retry_cooldown_seconds = max(0.0, retry if retry is not None else 30.0)
        self.block_strategy_for_protected_symbols = bool(cfg.get("block_strategy_for_protected_symbols", True))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "symbols": sorted(self.symbols),
            "manual_magic_values": sorted(self.manual_magic_values),
            "retain_ratio": float(self.retain_ratio),
            "min_activation_profit_usd": float(self.min_activation_profit_usd),
            "breach_close": bool(self.breach_close),
            "sync_sl_to_lock_line": bool(self.sync_sl_to_lock_line),
            "close_retry_cooldown_seconds": float(self.close_retry_cooldown_seconds),
            "block_strategy_for_protected_symbols": bool(self.block_strategy_for_protected_symbols),
            "peak_by_ticket": dict(self._peak_by_ticket),
            "last_close_attempt_by_ticket": dict(self._last_close_attempt_by_ticket),
        }

    def _is_protected_position(self, position: Position) -> bool:
        symbol = str(position.symbol or "").strip().upper()
        if symbol not in self.symbols:
            return False
        try:
            magic = int(position.magic or 0)
        except (TypeError, ValueError):
            magic = 0
        return magic in self.manual_magic_values

    def _position_net_floating_pnl(self, position: Position) -> Optional[float]:
        metadata = position.metadata if isinstance(position.metadata, dict) else {}
        floating = self._as_float(metadata.get("floating_pnl"))
        if floating is None:
            floating = self._as_float(metadata.get("profit"))
        if floating is None:
            return None

        swap = self._as_float(metadata.get("swap"))
        commission = self._as_float(metadata.get("commission"))
        return float(floating + (swap or 0.0) + (commission or 0.0))

    @staticmethod
    def _compute_lock_price(position: Position, lock_pnl: float, contract_size: float) -> Optional[float]:
        try:
            volume = float(position.volume)
            entry = float(position.price_open)
        except (TypeError, ValueError):
            return None
        denom = volume * max(1e-9, float(contract_size))
        if denom <= 0:
            return None
        delta = float(lock_pnl) / denom
        if position.side == Side.BUY:
            return entry + delta
        return entry - delta

    def _cleanup_stale_tickets(self, positions: Iterable[Position]) -> None:
        live = {self._ticket_key(pos.ticket) for pos in positions}
        self._peak_by_ticket = {k: v for k, v in self._peak_by_ticket.items() if k in live}
        self._last_close_attempt_by_ticket = {k: v for k, v in self._last_close_attempt_by_ticket.items() if k in live}

    def run_cycle(self, positions: List[Position], broker: BrokerGateway, now_ts: float) -> Dict[str, Any]:
        events: List[ManualGuardEvent] = []
        protected_symbols: Set[str] = set()

        if not self.enabled:
            self._cleanup_stale_tickets(positions)
            return {"events": events, "protected_symbols": protected_symbols}

        for position in positions:
            if not self._is_protected_position(position):
                continue

            symbol = str(position.symbol or "").strip().upper()
            protected_symbols.add(symbol)
            ticket_key = self._ticket_key(position.ticket)
            net_pnl = self._position_net_floating_pnl(position)
            if net_pnl is None:
                continue

            prev_peak = self._peak_by_ticket.get(ticket_key, net_pnl)
            peak = max(prev_peak, net_pnl)
            self._peak_by_ticket[ticket_key] = peak
            if peak > prev_peak:
                events.append(
                    ManualGuardEvent(
                        event="manual_guard_peak_update",
                        payload={
                            "symbol": symbol,
                            "ticket": int(position.ticket),
                            "peak_pnl_usd": float(peak),
                            "current_pnl_usd": float(net_pnl),
                        },
                    )
                )

            if peak < self.min_activation_profit_usd:
                continue

            lock_pnl = peak * self.retain_ratio
            constraints = broker.get_symbol_constraints(symbol)
            contract_size = float(constraints.contract_size) if constraints is not None else 1.0
            lock_price = self._compute_lock_price(position=position, lock_pnl=lock_pnl, contract_size=contract_size)

            if self.sync_sl_to_lock_line and lock_price is not None:
                existing_sl = float(position.sl) if position.sl is not None else None
                if position.side == Side.BUY:
                    desired_sl = max(existing_sl if existing_sl is not None else lock_price, lock_price)
                else:
                    desired_sl = min(existing_sl if existing_sl is not None else lock_price, lock_price)
                if existing_sl is None or abs(desired_sl - existing_sl) > 1e-9:
                    modify = broker.modify_position_sl_tp(position=position, sl=float(desired_sl), tp=position.tp, reason="manual_guard_lock_line")
                    events.append(
                        ManualGuardEvent(
                            event="manual_guard_sl_sync",
                            payload={
                                "symbol": symbol,
                                "ticket": int(position.ticket),
                                "existing_sl": existing_sl,
                                "desired_sl": float(desired_sl),
                                "lock_price": float(lock_price),
                                "lock_pnl_usd": float(lock_pnl),
                                "result": modify.__dict__,
                            },
                        )
                    )

            if not self.breach_close:
                continue
            if net_pnl >= lock_pnl:
                continue

            last_attempt = self._last_close_attempt_by_ticket.get(ticket_key, 0.0)
            if (now_ts - last_attempt) < self.close_retry_cooldown_seconds:
                continue

            self._last_close_attempt_by_ticket[ticket_key] = float(now_ts)
            events.append(
                ManualGuardEvent(
                    event="manual_guard_trigger",
                    payload={
                        "symbol": symbol,
                        "ticket": int(position.ticket),
                        "peak_pnl_usd": float(peak),
                        "current_pnl_usd": float(net_pnl),
                        "lock_pnl_usd": float(lock_pnl),
                    },
                )
            )
            result = broker.close_position(position, reason="manual_guard_retain_breach")
            events.append(
                ManualGuardEvent(
                    event="manual_guard_close_result",
                    payload={
                        "symbol": symbol,
                        "ticket": int(position.ticket),
                        "result": result.__dict__,
                    },
                )
            )

        self._cleanup_stale_tickets(positions)
        return {"events": events, "protected_symbols": protected_symbols}
