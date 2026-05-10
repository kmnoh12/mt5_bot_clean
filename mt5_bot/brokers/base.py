from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.models import MarketTick, OrderIntent, OrderResult, Position, SymbolConstraints


class BrokerGateway(ABC):
    mode: str

    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def heartbeat(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def fetch_bars(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        raise NotImplementedError

    @abstractmethod
    def get_symbol_constraints(self, symbol: str) -> Optional[SymbolConstraints]:
        raise NotImplementedError

    @abstractmethod
    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def send_order(self, intent: OrderIntent) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def modify_position_sl_tp(
        self,
        position: Position,
        sl: Optional[float],
        tp: Optional[float],
        reason: str,
    ) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def close_position(self, position: Position, reason: str) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def close_all_positions(self, reason: str) -> List[OrderResult]:
        raise NotImplementedError

    @abstractmethod
    def account_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    # Optional: some gateways can query broker-side history (e.g., MT5 deals) for a position ticket.
    # Used to reconcile SL/TP auto-closes that didn't go through bot-initiated close_position().
    def get_position_close_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        return None

    # Optional: tick streaming/range fetch for tick-driven strategies.
    # Gateways without tick support can return an empty list (strategy will degrade gracefully).
    def fetch_ticks(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        max_ticks: int = 2000,
    ) -> List[MarketTick]:
        return []

    def step(self) -> bool:
        return True
