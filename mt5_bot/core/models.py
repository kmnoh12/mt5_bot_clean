from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class BotMode(str, Enum):
    LIVE = "live"
    BACKTEST = "backtest"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class DecisionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    HOLD = "HOLD"


class StrategyState(str, Enum):
    IDLE = "IDLE"
    SETUP = "SETUP"
    ENTRY_READY = "ENTRY_READY"
    ENTRY_PENDING = "ENTRY_PENDING"
    IN_POSITION = "IN_POSITION"
    EXIT_READY = "EXIT_READY"
    COOLDOWN = "COOLDOWN"
    HALTED = "HALTED"


@dataclass
class SymbolConstraints:
    min_volume: float = 0.01
    max_volume: float = 100.0
    volume_step: float = 0.01
    point: float = 0.0001
    digits: int = 5
    contract_size: float = 1.0
    base_currency: str = ""
    quote_currency: str = ""
    profit_currency: str = ""
    trade_stops_level: float = 0.0
    trade_freeze_level: float = 0.0


@dataclass
class MarketTick:
    time_utc: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None
    time_msc: Optional[int] = None
    flags: Optional[int] = None

    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return (float(self.bid) + float(self.ask)) / 2.0
        for value in (self.last, self.bid, self.ask):
            if value is None:
                continue
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if out > 0:
                return out
        return None


@dataclass
class Position:
    ticket: int
    symbol: str
    side: Side
    volume: float
    price_open: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: str = ""
    magic: Optional[int] = None
    time_open_utc: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderIntent:
    symbol: str
    side: Side
    volume: float
    reason: str
    strategy: str
    comment: str = ""
    magic: Optional[int] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    external_signal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    ok: bool
    status: str
    message: str = ""
    ticket: Optional[int] = None
    retcode: Optional[int] = None
    filled_price: Optional[float] = None
    pnl: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalSignal:
    signal_id: str
    symbol: str
    action: DecisionAction
    reason: str = ""
    strategy: str = "external"
    confidence: float = 1.0
    volume: Optional[float] = None
    ttl_seconds: int = 300
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now_utc: Optional[datetime] = None) -> bool:
        now_utc = now_utc or datetime.now(timezone.utc)
        age_seconds = (now_utc - self.created_at_utc).total_seconds()
        return age_seconds > max(1, int(self.ttl_seconds))


@dataclass
class StrategyEvaluationContext:
    mtf_info: Dict[str, Any] = field(default_factory=dict)
    equity: Optional[float] = None
    equity_peak: Optional[float] = None
    loss_streak: int = 0
    daily_pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.mtf_info:
            payload["mtf_info"] = dict(self.mtf_info)
        if self.equity is not None:
            payload["equity"] = float(self.equity)
        if self.equity_peak is not None:
            payload["equity_peak"] = float(self.equity_peak)
        payload["loss_streak"] = max(0, int(self.loss_streak))
        if self.daily_pnl is not None:
            payload["daily_pnl"] = float(self.daily_pnl)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class StrategyDecision:
    action: DecisionAction
    reason: str
    strategy: str
    confidence: float = 0.0
    volume: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    signal_bar_time: Optional[datetime] = None
    min_hold_bars: Optional[int] = None
    state: Optional[StrategyState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySymbolState:
    state: StrategyState = StrategyState.IDLE
    bias: Optional[Side] = None
    cooldown_bars_remaining: int = 0
    last_reason: str = ""
    entry_price: Optional[float] = None
    peak_price: Optional[float] = None
    trough_price: Optional[float] = None
    last_closed_bar_time: Optional[datetime] = None
    entry_bar_time: Optional[datetime] = None
    pending_order: bool = False
    updated_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "bias": self.bias.value if self.bias else None,
            "cooldown_bars_remaining": self.cooldown_bars_remaining,
            "last_reason": self.last_reason,
            "entry_price": self.entry_price,
            "peak_price": self.peak_price,
            "trough_price": self.trough_price,
            "last_closed_bar_time": self.last_closed_bar_time.isoformat() if self.last_closed_bar_time else None,
            "entry_bar_time": self.entry_bar_time.isoformat() if self.entry_bar_time else None,
            "pending_order": self.pending_order,
            "updated_at_utc": self.updated_at_utc.isoformat(),
            "metadata": self.metadata,
        }


def parse_action(value: Any) -> Optional[DecisionAction]:
    text = str(value or "").strip().upper()
    if text == "BUY":
        return DecisionAction.BUY
    if text == "SELL":
        return DecisionAction.SELL
    if text in {"EXIT", "CLOSE", "FLAT"}:
        return DecisionAction.EXIT
    if text in {"HOLD", "WAIT", "NONE"}:
        return DecisionAction.HOLD
    return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
