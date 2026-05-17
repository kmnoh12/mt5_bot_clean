from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    sys.modules["pandas"] = pandas_stub

from core.models import DecisionAction, OrderIntent, OrderResult, Position, Side, StrategyDecision, StrategyState, SymbolConstraints
from execution.order_manager import OrderManager
from execution.risk_manager import RiskEngine
from storage.json_store import JsonStore
from strategies.base import BaseStateMachineStrategy


class _NoopNotifier:
    @staticmethod
    def send_trade(_message: str) -> None:
        return

    @staticmethod
    def send_error(_message: str) -> None:
        return


class _TinyStrategy(BaseStateMachineStrategy):
    def __init__(self) -> None:
        super().__init__(name="tiny", config={"min_cooldown_bars": 3})

    def _evaluate_impl(self, symbol, bars, position, st):  # noqa: ANN001, ARG002
        return StrategyDecision(action=DecisionAction.HOLD, reason="noop", strategy=self.name)


class _FakeStopsBroker:
    mode = "backtest"

    def __init__(self, anchor_price: float = 100.0) -> None:
        self.anchor_price = float(anchor_price)
        self.precheck_calls = 0
        self._positions: Dict[int, Position] = {}

    def get_symbol_constraints(self, symbol: str) -> Optional[SymbolConstraints]:  # noqa: ARG002
        return SymbolConstraints(
            min_volume=0.01,
            max_volume=1.0,
            volume_step=0.01,
            point=0.01,
            digits=2,
            contract_size=1.0,
            quote_currency="USD",
            profit_currency="USD",
            trade_stops_level=50.0,
            trade_freeze_level=0.0,
        )

    def get_latest_price(self, symbol: str) -> Optional[float]:  # noqa: ARG002
        return float(self.anchor_price)

    def account_info(self) -> Dict[str, Any]:
        return {"equity": 1000.0, "balance": 1000.0, "currency": "USD", "open_positions": len(self._positions)}

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:  # noqa: ARG002
        return []

    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        self.precheck_calls += 1
        sl = float(intent.sl or 0.0)
        if intent.side == Side.BUY and sl > 99.5:
            return OrderResult(ok=False, status="CHECK_REJECTED", message="invalid_stops", retcode=10016)
        return OrderResult(ok=True, status="CHECK_OK", message="ok", retcode=0)

    def send_order(self, intent: OrderIntent) -> OrderResult:  # noqa: ARG002
        raise AssertionError("send_order must not run when adjusted stop exceeds cap")


class BlockerChainNoPandasTests(unittest.TestCase):
    def test_no_trade_rejected_entry_result_does_not_apply_trade_cooldown(self) -> None:
        strategy = _TinyStrategy()
        state = strategy.get_symbol_state("BTCUSD")
        state.state = StrategyState.ENTRY_PENDING
        state.pending_order = True
        state.cooldown_bars_remaining = 0

        strategy.apply_order_result(
            symbol="BTCUSD",
            decision=StrategyDecision(action=DecisionAction.BUY, reason="unit", strategy="tiny"),
            result=OrderResult(ok=False, status="RISK_PLAN_FAILED", message="min lot over cap"),
        )

        self.assertEqual(state.state, StrategyState.IDLE)
        self.assertFalse(state.pending_order)
        self.assertEqual(state.cooldown_bars_remaining, 0)
        self.assertEqual(state.last_reason, "RISK_PLAN_FAILED")

    def test_invalid_stop_widening_blocks_when_adjusted_loss_exceeds_cap(self) -> None:
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
                execution_cfg={
                    "default_volume": 0.01,
                    "max_expected_loss_usd_by_symbol": {"BTCUSD": 0.003},
                },
                risk_engine=RiskEngine({"dynamic_lot_enabled": False}),
                dry_run=False,
            )

            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.01},
                decision=StrategyDecision(
                    action=DecisionAction.BUY,
                    reason="TEST_BUY_INVALID_STOPS_CAP",
                    strategy="unit",
                    sl=99.8,
                    tp=101.0,
                    metadata={"signal_close": 100.0, "risk_per_unit": 0.2},
                ),
                current_position=None,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertFalse(result.ok)
            self.assertEqual(result.status, "EXPECTED_LOSS_CAP_AFTER_STOP_ADJUSTMENT")
            self.assertEqual(broker.precheck_calls, 1)
            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(
                event.get("event") == "invalid_stops_adjustment_blocked_by_cap"
                and event.get("details", {}).get("expected_loss_usd", 0) > 0.003
                for event in events
            ))


if __name__ == "__main__":
    unittest.main()
