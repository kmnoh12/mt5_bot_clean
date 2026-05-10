import json
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


class _Broker(BrokerGateway):
    mode = "backtest"

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return

    def heartbeat(self) -> bool:
        return True

    def fetch_bars(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        return []

    def get_symbol_constraints(self, symbol: str) -> Optional[SymbolConstraints]:
        return SymbolConstraints(contract_size=1.0)

    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        return OrderResult(ok=True, status="OK", retcode=0)

    def send_order(self, intent: OrderIntent) -> OrderResult:
        return OrderResult(ok=True, status="FILLED", ticket=1, filled_price=100.0)

    def modify_position_sl_tp(self, position: Position, sl: Optional[float], tp: Optional[float], reason: str) -> OrderResult:
        return OrderResult(ok=True, status="MODIFIED")

    def close_position(self, position: Position, reason: str) -> OrderResult:
        return OrderResult(ok=True, status="CLOSED", ticket=position.ticket, filled_price=100.0, retcode=10009)

    def close_all_positions(self, reason: str) -> List[OrderResult]:
        return []

    def account_info(self) -> Dict[str, Any]:
        return {"equity": 1000.0, "balance": 1000.0, "currency": "USD"}


class TradeLedgerNormalizationTests(unittest.TestCase):
    def test_trade_ledger_normalized_emitted_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonStore(Path(tmpdir) / "state.json", Path(tmpdir) / "events.jsonl")
            manager = OrderManager(
                broker=_Broker(),
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.01},
                risk_engine=RiskEngine({}),
                dry_run=False,
            )
            position = Position(
                ticket=77,
                symbol="BTCUSD",
                side=Side.BUY,
                volume=0.01,
                price_open=100.0,
                metadata={"swap": 0.0, "commission": 0.0},
            )
            decision = StrategyDecision(
                action=DecisionAction.EXIT,
                reason="TEST_EXIT",
                strategy="trend_regime_sm",
                metadata={"exit_attempt_no": 2},
            )
            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.01},
                decision=decision,
                current_position=position,
            )
            self.assertIsNotNone(result)
            lines = [json.loads(x) for x in Path(tmpdir, "events.jsonl").read_text(encoding="utf-8").splitlines()]
            ledgers = [x for x in lines if x.get("event") == "trade_ledger"]
            normalized = [x for x in lines if x.get("event") == "trade_ledger_normalized"]
            self.assertEqual(len(ledgers), 1)
            self.assertEqual(len(normalized), 1)
            self.assertEqual(int(ledgers[0].get("exit_attempt_no", 0)), 2)
            self.assertEqual(ledgers[0].get("pnl_status"), "known")


if __name__ == "__main__":
    unittest.main()
