import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from brokers.base import BrokerGateway
from core.models import DecisionAction, OrderIntent, OrderResult, Position, Side, StrategyDecision, SymbolConstraints
from execution.order_manager import OrderManager
from execution.risk_engine import RiskEngine
from storage.json_store import JsonStore


class _NoopNotifier:
    def send_trade(self, _message: str) -> None:
        return

    def send_error(self, _message: str) -> None:
        return


class _FakeStopsBroker(BrokerGateway):
    mode = "backtest"

    def __init__(self, anchor_price: float = 100.0) -> None:
        self.anchor_price = float(anchor_price)
        self.precheck_calls = 0
        self.last_intent: Optional[OrderIntent] = None
        self._positions: Dict[int, Position] = {}

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return

    def heartbeat(self) -> bool:
        return True

    def fetch_bars(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:  # noqa: A003
        return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if symbol is None:
            return list(self._positions.values())
        sym = str(symbol or "").upper()
        return [p for p in self._positions.values() if str(p.symbol).upper() == sym]

    def get_symbol_constraints(self, symbol: str) -> Optional[SymbolConstraints]:
        return SymbolConstraints(
            min_volume=0.01,
            max_volume=1.0,
            volume_step=0.01,
            point=0.01,
            digits=2,
            contract_size=1.0,
            quote_currency="USD",
            profit_currency="USD",
            trade_stops_level=50.0,  # 50 points * 0.01 = 0.50 price distance
            trade_freeze_level=0.0,
        )

    def get_latest_price(self, symbol: str) -> Optional[float]:
        if str(symbol or "").upper() == "BTCUSD":
            return float(self.anchor_price)
        return None

    def _invalid_stops(self, intent: OrderIntent, constraints: SymbolConstraints) -> bool:
        anchor = float(self.anchor_price)
        point = float(constraints.point or 0.0)
        min_distance = float(constraints.trade_stops_level or 0.0) * point

        sl = float(intent.sl) if intent.sl is not None else None
        tp = float(intent.tp) if intent.tp is not None else None

        if sl is None:
            return True

        if intent.side == Side.BUY:
            if sl >= anchor:
                return True
            if (anchor - sl) < min_distance:
                return True
            if tp is not None:
                if tp <= anchor:
                    return True
                if (tp - anchor) < min_distance:
                    return True
        else:
            if sl <= anchor:
                return True
            if (sl - anchor) < min_distance:
                return True
            if tp is not None:
                if tp >= anchor:
                    return True
                if (anchor - tp) < min_distance:
                    return True
        return False

    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        self.precheck_calls += 1
        self.last_intent = intent
        constraints = self.get_symbol_constraints(intent.symbol) or SymbolConstraints()
        if self._invalid_stops(intent, constraints):
            return OrderResult(ok=False, status="CHECK_REJECTED", message="invalid_stops", retcode=10016)
        return OrderResult(ok=True, status="CHECK_OK", message="ok", retcode=0)

    def send_order(self, intent: OrderIntent) -> OrderResult:
        self.last_intent = intent
        pos = Position(
            ticket=1,
            symbol=intent.symbol,
            side=intent.side,
            volume=float(intent.volume),
            price_open=float(self.anchor_price),
            sl=float(intent.sl) if intent.sl is not None else None,
            tp=float(intent.tp) if intent.tp is not None else None,
        )
        self._positions[1] = pos
        return OrderResult(ok=True, status="FILLED", ticket=1, filled_price=float(self.anchor_price))

    def modify_position_sl_tp(self, position: Position, sl: Optional[float], tp: Optional[float], reason: str) -> OrderResult:
        current = self._positions.get(int(position.ticket))
        if current is None:
            return OrderResult(ok=False, status="MODIFY_NOT_FOUND", message="not found")
        current.sl = sl
        current.tp = tp
        return OrderResult(ok=True, status="MODIFIED", ticket=current.ticket)

    def close_position(self, position: Position, reason: str) -> OrderResult:
        self._positions.pop(int(position.ticket), None)
        return OrderResult(ok=True, status="CLOSED", ticket=int(position.ticket), filled_price=float(self.anchor_price))

    def close_all_positions(self, reason: str) -> List[OrderResult]:
        out = [self.close_position(p, reason) for p in list(self._positions.values())]
        return out

    def account_info(self) -> Dict[str, Any]:
        return {"equity": 1000.0, "balance": 1000.0, "currency": "USD", "open_positions": len(self._positions)}


class StopTpClampTests(unittest.TestCase):
    def test_buy_auto_adjusts_invalid_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeStopsBroker(anchor_price=100.0)
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.1},
                risk_engine=RiskEngine({}),
                dry_run=False,
            )

            decision = StrategyDecision(
                action=DecisionAction.BUY,
                reason="TEST_BUY_INVALID_STOPS",
                strategy="unit",
                sl=99.8,
                tp=100.2,
                metadata={"signal_close": 100.0},
            )
            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.2},
                decision=decision,
                current_position=None,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.ok)
            self.assertGreaterEqual(int(broker.precheck_calls), 2)
            self.assertIsNotNone(broker.last_intent)
            assert broker.last_intent is not None
            self.assertAlmostEqual(float(broker.last_intent.sl or 0.0), 99.5, places=8)
            self.assertAlmostEqual(float(broker.last_intent.tp or 0.0), 100.5, places=8)
            self.assertTrue(bool(broker.last_intent.metadata.get("stop_adjustment_applied", False)))

    def test_sell_auto_adjusts_invalid_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeStopsBroker(anchor_price=100.0)
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.1},
                risk_engine=RiskEngine({}),
                dry_run=False,
            )

            decision = StrategyDecision(
                action=DecisionAction.SELL,
                reason="TEST_SELL_INVALID_STOPS",
                strategy="unit",
                sl=100.2,
                tp=99.8,
                metadata={"signal_close": 100.0},
            )
            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.2},
                decision=decision,
                current_position=None,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.ok)
            self.assertGreaterEqual(int(broker.precheck_calls), 2)
            self.assertIsNotNone(broker.last_intent)
            assert broker.last_intent is not None
            self.assertAlmostEqual(float(broker.last_intent.sl or 0.0), 100.5, places=8)
            self.assertAlmostEqual(float(broker.last_intent.tp or 0.0), 99.5, places=8)
            self.assertTrue(bool(broker.last_intent.metadata.get("stop_adjustment_applied", False)))


if __name__ == "__main__":
    unittest.main()

