import unittest
from typing import Any, Dict, List, Optional

import pandas as pd

from brokers.base import BrokerGateway
from core.models import OrderIntent, OrderResult, Position, Side, SymbolConstraints
from execution.manual_position_guard import ManualPositionGuard


class _GuardBroker(BrokerGateway):
    mode = "backtest"

    def __init__(self) -> None:
        self._positions: Dict[int, Position] = {}
        self.close_calls = 0
        self.modify_calls = 0

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return

    def heartbeat(self) -> bool:
        return True

    def fetch_bars(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if symbol is None:
            return list(self._positions.values())
        return [p for p in self._positions.values() if p.symbol == symbol]

    def get_symbol_constraints(self, symbol: str) -> Optional[SymbolConstraints]:
        return SymbolConstraints(contract_size=1.0)

    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        return OrderResult(ok=True, status="OK")

    def send_order(self, intent: OrderIntent) -> OrderResult:
        return OrderResult(ok=True, status="FILLED")

    def modify_position_sl_tp(self, position: Position, sl: Optional[float], tp: Optional[float], reason: str) -> OrderResult:
        self.modify_calls += 1
        current = self._positions.get(position.ticket)
        if current is not None:
            current.sl = sl
        return OrderResult(ok=True, status="MODIFIED", ticket=position.ticket)

    def close_position(self, position: Position, reason: str) -> OrderResult:
        self.close_calls += 1
        return OrderResult(ok=False, status="ERROR", message="simulated close fail", ticket=position.ticket)

    def close_all_positions(self, reason: str) -> List[OrderResult]:
        return []

    def account_info(self) -> Dict[str, Any]:
        return {}


class ManualPositionGuardTests(unittest.TestCase):
    def _position(self, pnl: float, ticket: int = 1) -> Position:
        return Position(
            ticket=ticket,
            symbol="GOLD",
            side=Side.BUY,
            volume=1.0,
            price_open=100.0,
            sl=None,
            tp=None,
            magic=0,
            metadata={"floating_pnl": pnl, "swap": 0.0, "commission": 0.0},
        )

    def test_no_immediate_close_right_after_open(self) -> None:
        broker = _GuardBroker()
        guard = ManualPositionGuard(config={})
        pos = self._position(pnl=1.0)
        broker._positions[pos.ticket] = pos
        report = guard.run_cycle([pos], broker=broker, now_ts=1000.0)
        self.assertEqual(broker.close_calls, 0)
        self.assertEqual(len(report["protected_symbols"]), 1)

    def test_trigger_on_peak_retrace_below_80_percent(self) -> None:
        broker = _GuardBroker()
        guard = ManualPositionGuard(config={"retain_ratio": 0.8, "min_activation_profit_usd": 5.0})
        pos = self._position(pnl=100.0)
        broker._positions[pos.ticket] = pos
        guard.run_cycle([pos], broker=broker, now_ts=1000.0)

        pos.metadata["floating_pnl"] = 79.0
        guard.run_cycle([pos], broker=broker, now_ts=1040.0)
        self.assertEqual(broker.close_calls, 1)

    def test_no_trigger_below_activation_threshold(self) -> None:
        broker = _GuardBroker()
        guard = ManualPositionGuard(config={"retain_ratio": 0.8, "min_activation_profit_usd": 5.0})
        pos = self._position(pnl=4.9)
        broker._positions[pos.ticket] = pos
        guard.run_cycle([pos], broker=broker, now_ts=1000.0)
        pos.metadata["floating_pnl"] = 0.1
        guard.run_cycle([pos], broker=broker, now_ts=1005.0)
        self.assertEqual(broker.close_calls, 0)

    def test_close_retry_cooldown_blocks_repeated_attempts(self) -> None:
        broker = _GuardBroker()
        guard = ManualPositionGuard(config={"close_retry_cooldown_seconds": 30})
        pos = self._position(pnl=100.0)
        broker._positions[pos.ticket] = pos
        guard.run_cycle([pos], broker=broker, now_ts=1000.0)
        pos.metadata["floating_pnl"] = 50.0
        guard.run_cycle([pos], broker=broker, now_ts=1010.0)
        guard.run_cycle([pos], broker=broker, now_ts=1020.0)
        self.assertEqual(broker.close_calls, 1)
        guard.run_cycle([pos], broker=broker, now_ts=1045.0)
        self.assertEqual(broker.close_calls, 2)


if __name__ == "__main__":
    unittest.main()
