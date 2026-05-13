from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from execution.order_permissions import OrderPermissionState


@dataclass(frozen=True)
class DailyBleedBlock:
    reason: str
    detail: str
    cooldown_until_ts: Optional[float] = None


class DailyBleedGuard:
    def __init__(self, config: Optional[Dict[str, Any]] = None, snapshot: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        nested_cfg = cfg.get("daily_bleed_guard")
        if isinstance(nested_cfg, dict):
            cfg = dict(nested_cfg)
        self.enabled = _bool_cfg(cfg.get("enabled"), True)
        self.max_daily_net_loss_usd = _float_cfg(cfg.get("max_daily_net_loss_usd"), 3.0, minimum=0.0)
        self.stop_after_consecutive_losses = _int_cfg(cfg.get("stop_after_consecutive_losses"), 3, minimum=1)
        self.cooldown_after_loss_minutes = _float_cfg(cfg.get("cooldown_after_loss_minutes"), 30.0, minimum=0.0)
        self.cooldown_after_same_setup_loss_minutes = _float_cfg(
            cfg.get("cooldown_after_same_setup_loss_minutes"),
            60.0,
            minimum=0.0,
        )
        self.same_direction_loss_limit_per_day = _int_cfg(
            cfg.get("same_direction_loss_limit_per_day"),
            2,
            minimum=1,
        )
        self.same_symbol_loss_limit_per_day = _int_cfg(cfg.get("same_symbol_loss_limit_per_day"), 3, minimum=1)

        self._day_key: Optional[str] = None
        self._daily_net_pnl_usd = 0.0
        self._consecutive_losses = 0
        self._loss_cooldown_until_by_symbol: Dict[str, float] = {}
        self._setup_cooldown_until: Dict[str, float] = {}
        self._same_setup_losses_by_day: Dict[str, int] = {}
        self._same_direction_losses_by_day: Dict[str, int] = {}
        self._same_symbol_losses_by_day: Dict[str, int] = {}

        if snapshot:
            self._restore_snapshot(snapshot)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "max_daily_net_loss_usd": float(self.max_daily_net_loss_usd),
            "stop_after_consecutive_losses": int(self.stop_after_consecutive_losses),
            "cooldown_after_loss_minutes": float(self.cooldown_after_loss_minutes),
            "cooldown_after_same_setup_loss_minutes": float(self.cooldown_after_same_setup_loss_minutes),
            "same_direction_loss_limit_per_day": int(self.same_direction_loss_limit_per_day),
            "same_symbol_loss_limit_per_day": int(self.same_symbol_loss_limit_per_day),
            "day_key": self._day_key,
            "daily_net_pnl_usd": float(self._daily_net_pnl_usd),
            "consecutive_losses": int(self._consecutive_losses),
            "loss_cooldown_until_by_symbol": dict(self._loss_cooldown_until_by_symbol),
            "setup_cooldown_until": dict(self._setup_cooldown_until),
            "same_setup_losses_by_day": dict(self._same_setup_losses_by_day),
            "same_direction_losses_by_day": dict(self._same_direction_losses_by_day),
            "same_symbol_losses_by_day": dict(self._same_symbol_losses_by_day),
        }

    def should_block_entry(
        self,
        symbol: str,
        now_ts: Optional[float] = None,
        direction: Optional[str] = None,
        setup_key: Optional[str] = None,
    ) -> Optional[str]:
        block = self.entry_block(symbol=symbol, now_ts=now_ts, direction=direction, setup_key=setup_key)
        return None if block is None else block.reason

    def entry_block(
        self,
        symbol: str,
        now_ts: Optional[float] = None,
        direction: Optional[str] = None,
        setup_key: Optional[str] = None,
    ) -> Optional[DailyBleedBlock]:
        if not self.enabled:
            return None
        ts = _now_ts(now_ts)
        self._reset_if_new_day(ts)
        self._prune_expired_cooldowns(ts)

        if self._daily_net_pnl_usd <= -abs(self.max_daily_net_loss_usd):
            return DailyBleedBlock("DAILY_BLEED_NET_LOSS_LIMIT", "daily realized net loss limit reached")
        if self._consecutive_losses >= self.stop_after_consecutive_losses:
            return DailyBleedBlock("DAILY_BLEED_CONSECUTIVE_LOSSES", "consecutive loss limit reached")

        symbol_key = _symbol_key(symbol)
        cooldown_until = self._loss_cooldown_until_by_symbol.get(symbol_key, 0.0)
        if ts < cooldown_until:
            return DailyBleedBlock("DAILY_BLEED_LOSS_COOLDOWN", "symbol loss cooldown active", cooldown_until)

        setup = _setup_key(setup_key)
        if setup:
            setup_cooldown_until = self._setup_cooldown_until.get(setup, 0.0)
            if ts < setup_cooldown_until:
                return DailyBleedBlock(
                    "DAILY_BLEED_SAME_SETUP_COOLDOWN",
                    "same setup loss cooldown active",
                    setup_cooldown_until,
                )

        if direction is not None:
            direction_key = self._symbol_direction_key(symbol_key, direction)
            if self._same_direction_losses_by_day.get(direction_key, 0) >= self.same_direction_loss_limit_per_day:
                return DailyBleedBlock("DAILY_BLEED_SAME_DIRECTION_LIMIT", "same symbol/direction loss limit reached")

        if self._same_symbol_losses_by_day.get(symbol_key, 0) >= self.same_symbol_loss_limit_per_day:
            return DailyBleedBlock("DAILY_BLEED_SAME_SYMBOL_LIMIT", "same symbol loss limit reached")
        return None

    def permission_state(
        self,
        symbol: str,
        now_ts: Optional[float] = None,
        direction: Optional[str] = None,
        setup_key: Optional[str] = None,
    ) -> OrderPermissionState:
        return OrderPermissionState.from_daily_bleed_guard(
            self.should_block_entry(symbol=symbol, now_ts=now_ts, direction=direction, setup_key=setup_key)
        )

    def order_permission_state(
        self,
        symbol: str,
        now_ts: Optional[float] = None,
        direction: Optional[str] = None,
        setup_key: Optional[str] = None,
    ) -> OrderPermissionState:
        return self.permission_state(symbol=symbol, now_ts=now_ts, direction=direction, setup_key=setup_key)

    def record_trade_close(
        self,
        symbol: str,
        realized_pnl: float,
        now_ts: Optional[float] = None,
        direction: Optional[str] = None,
        setup_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        ts = _now_ts(now_ts)
        self._reset_if_new_day(ts)
        pnl = float(realized_pnl)
        self._daily_net_pnl_usd += pnl

        symbol_key = _symbol_key(symbol)
        setup = _setup_key(setup_key)
        direction_key = self._symbol_direction_key(symbol_key, direction) if direction is not None else None

        if pnl < 0.0:
            self._consecutive_losses += 1
            if self.cooldown_after_loss_minutes > 0:
                self._loss_cooldown_until_by_symbol[symbol_key] = max(
                    self._loss_cooldown_until_by_symbol.get(symbol_key, 0.0),
                    ts + self.cooldown_after_loss_minutes * 60.0,
                )
            if setup and self.cooldown_after_same_setup_loss_minutes > 0:
                self._setup_cooldown_until[setup] = max(
                    self._setup_cooldown_until.get(setup, 0.0),
                    ts + self.cooldown_after_same_setup_loss_minutes * 60.0,
                )
                self._same_setup_losses_by_day[setup] = self._same_setup_losses_by_day.get(setup, 0) + 1
            if direction_key:
                self._same_direction_losses_by_day[direction_key] = (
                    self._same_direction_losses_by_day.get(direction_key, 0) + 1
                )
            self._same_symbol_losses_by_day[symbol_key] = self._same_symbol_losses_by_day.get(symbol_key, 0) + 1
        elif pnl > 0.0:
            self._consecutive_losses = 0

        self._prune_expired_cooldowns(ts)
        return {
            "day_key": self._day_key,
            "symbol": symbol_key,
            "realized_pnl": pnl,
            "daily_net_pnl_usd": float(self._daily_net_pnl_usd),
            "consecutive_losses": int(self._consecutive_losses),
        }

    def record_close(
        self,
        symbol: str,
        now_ts: float,
        realized_pnl: float,
        direction: Optional[str] = None,
        setup_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.record_trade_close(
            symbol=symbol,
            realized_pnl=realized_pnl,
            now_ts=now_ts,
            direction=direction,
            setup_key=setup_key,
        )

    def reset_for_day(self, now_ts: Optional[float] = None) -> None:
        self._day_key = _date_key(_now_ts(now_ts))
        self._daily_net_pnl_usd = 0.0
        self._consecutive_losses = 0
        self._loss_cooldown_until_by_symbol = {}
        self._setup_cooldown_until = {}
        self._same_setup_losses_by_day = {}
        self._same_direction_losses_by_day = {}
        self._same_symbol_losses_by_day = {}

    @property
    def daily_net_pnl_usd(self) -> float:
        return float(self._daily_net_pnl_usd)

    @property
    def consecutive_losses(self) -> int:
        return int(self._consecutive_losses)

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.enabled = _bool_cfg(snapshot.get("enabled"), self.enabled)
        self.max_daily_net_loss_usd = _float_cfg(
            snapshot.get("max_daily_net_loss_usd"),
            self.max_daily_net_loss_usd,
            minimum=0.0,
        )
        self.stop_after_consecutive_losses = _int_cfg(
            snapshot.get("stop_after_consecutive_losses"),
            self.stop_after_consecutive_losses,
            minimum=1,
        )
        self.cooldown_after_loss_minutes = _float_cfg(
            snapshot.get("cooldown_after_loss_minutes"),
            self.cooldown_after_loss_minutes,
            minimum=0.0,
        )
        self.cooldown_after_same_setup_loss_minutes = _float_cfg(
            snapshot.get("cooldown_after_same_setup_loss_minutes"),
            self.cooldown_after_same_setup_loss_minutes,
            minimum=0.0,
        )
        self.same_direction_loss_limit_per_day = _int_cfg(
            snapshot.get("same_direction_loss_limit_per_day"),
            self.same_direction_loss_limit_per_day,
            minimum=1,
        )
        self.same_symbol_loss_limit_per_day = _int_cfg(
            snapshot.get("same_symbol_loss_limit_per_day"),
            self.same_symbol_loss_limit_per_day,
            minimum=1,
        )
        self._day_key = str(snapshot.get("day_key") or "") or None
        self._daily_net_pnl_usd = _float_cfg(snapshot.get("daily_net_pnl_usd"), 0.0)
        self._consecutive_losses = _int_cfg(snapshot.get("consecutive_losses"), 0, minimum=0)
        self._loss_cooldown_until_by_symbol = _float_map(snapshot.get("loss_cooldown_until_by_symbol"))
        self._setup_cooldown_until = _float_map(snapshot.get("setup_cooldown_until"))
        self._same_setup_losses_by_day = _int_map(snapshot.get("same_setup_losses_by_day"))
        self._same_direction_losses_by_day = _int_map(snapshot.get("same_direction_losses_by_day"))
        self._same_symbol_losses_by_day = _int_map(snapshot.get("same_symbol_losses_by_day"))

    def _reset_if_new_day(self, now_ts: float) -> None:
        day_key = _date_key(now_ts)
        if self._day_key is None:
            self._day_key = day_key
            return
        if self._day_key != day_key:
            self.reset_for_day(now_ts)

    def _prune_expired_cooldowns(self, now_ts: float) -> None:
        self._loss_cooldown_until_by_symbol = {
            key: value for key, value in self._loss_cooldown_until_by_symbol.items() if float(value) > now_ts
        }
        self._setup_cooldown_until = {
            key: value for key, value in self._setup_cooldown_until.items() if float(value) > now_ts
        }

    @staticmethod
    def _symbol_direction_key(symbol_key: str, direction: Optional[str]) -> str:
        return f"{symbol_key}:{str(direction or '').strip().upper()}"


def _now_ts(now_ts: Optional[float]) -> float:
    if now_ts is None:
        return datetime.now(timezone.utc).timestamp()
    return float(now_ts)


def _date_key(now_ts: float) -> str:
    return datetime.fromtimestamp(float(now_ts), tz=timezone.utc).date().isoformat()


def _symbol_key(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _setup_key(setup_key: Optional[str]) -> str:
    return str(setup_key or "").strip().upper()


def _bool_cfg(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _float_cfg(value: Any, default: float, minimum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if minimum is not None:
        parsed = max(float(minimum), parsed)
    return parsed


def _int_cfg(value: Any, default: int, minimum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    return parsed


def _float_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _int_map(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return out


def utc_ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> float:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()
