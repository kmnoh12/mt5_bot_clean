import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from brokers.base import BrokerGateway
from core.models import (
    DecisionAction,
    OrderIntent,
    OrderResult,
    Position,
    Side,
    StrategyDecision,
    SymbolConstraints,
)
from execution.order_manager import OrderManager
from execution.risk_engine import RiskEngine
from storage.json_store import JsonStore


class _NoopNotifier:
    def send_trade(self, _message: str) -> None:
        return

    def send_error(self, _message: str) -> None:
        return


class _FakeBroker(BrokerGateway):
    mode = "backtest"

    def __init__(self, fail_first_precheck: bool = True, account_currency: str = "USD") -> None:
        self.precheck_calls = 0
        self.last_intent: Optional[OrderIntent] = None
        self.last_close_volume: Optional[float] = None
        self._positions: Dict[int, Position] = {}
        self.fail_first_precheck = bool(fail_first_precheck)
        self.account_currency = str(account_currency or "USD").upper()
        self.fx_rates: Dict[str, float] = {"USDKRW": 1350.0}

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
        symbol_text = str(symbol or "").upper()
        if symbol_text == "SILVER":
            return SymbolConstraints(
                min_volume=0.01,
                max_volume=1.0,
                volume_step=0.01,
                point=0.01,
                digits=2,
                contract_size=5000.0,
                quote_currency="USD",
                profit_currency="USD",
            )
        return SymbolConstraints(
            min_volume=0.01,
            max_volume=1.0,
            volume_step=0.01,
            point=0.01,
            digits=2,
            contract_size=1.0,
            quote_currency="USD",
            profit_currency="USD",
        )

    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        self.precheck_calls += 1
        if self.fail_first_precheck and self.precheck_calls == 1:
            return OrderResult(ok=False, status="CHECK_REJECTED", message="invalid_volume", retcode=10014)
        return OrderResult(ok=True, status="CHECK_OK", message="ok", retcode=0)

    def send_order(self, intent: OrderIntent) -> OrderResult:
        self.last_intent = intent
        pos = Position(
            ticket=1,
            symbol=intent.symbol,
            side=intent.side,
            volume=intent.volume,
            price_open=100.0,
            sl=intent.sl,
            tp=intent.tp,
        )
        self._positions[1] = pos
        return OrderResult(ok=True, status="FILLED", ticket=1, filled_price=100.0)

    def modify_position_sl_tp(self, position: Position, sl: Optional[float], tp: Optional[float], reason: str) -> OrderResult:
        current = self._positions.get(position.ticket)
        if current is None:
            return OrderResult(ok=False, status="MODIFY_NOT_FOUND", message="not found")
        current.sl = sl
        current.tp = tp
        return OrderResult(ok=True, status="MODIFIED", ticket=position.ticket)

    def close_position(self, position: Position, reason: str) -> OrderResult:
        self.last_close_volume = float(position.volume)
        self._positions.pop(position.ticket, None)
        return OrderResult(ok=True, status="CLOSED", ticket=position.ticket, filled_price=101.0)

    def close_all_positions(self, reason: str) -> List[OrderResult]:
        out = [self.close_position(p, reason) for p in list(self._positions.values())]
        return out

    def account_info(self) -> Dict[str, Any]:
        equity = 1000.0 if self.account_currency == "USD" else 10_000_000.0
        return {
            "equity": equity,
            "balance": equity,
            "open_positions": len(self._positions),
            "currency": self.account_currency,
        }

    def get_latest_price(self, symbol: str) -> Optional[float]:
        value = self.fx_rates.get(str(symbol or "").upper())
        return float(value) if value is not None else None


class OrderManagerTests(unittest.TestCase):
    def test_entry_uses_sltp_and_retries_on_10014(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeBroker()
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            risk = RiskEngine(
                {
                    "risk_per_trade_pct": 0.015,
                    "daily_loss_limit_pct": 0.06,
                    "session_loss_limit_pct": 0.12,
                    "max_consecutive_losses": 5,
                }
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.1},
                risk_engine=risk,
                dry_run=False,
            )

            decision = StrategyDecision(
                action=DecisionAction.BUY,
                reason="TEST_ENTRY",
                strategy="trend_regime_sm",
                sl=99.0,
                tp=102.0,
                metadata={"signal_close": 100.0},
            )
            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.2},
                decision=decision,
                current_position=None,
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.ok)
            self.assertGreaterEqual(broker.precheck_calls, 2)
            self.assertIsNotNone(broker.last_intent)
            self.assertIsNotNone(broker.last_intent.sl)
            self.assertIsNotNone(broker.last_intent.tp)

    def test_entry_allows_none_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeBroker(fail_first_precheck=False)
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            risk = RiskEngine(
                {
                    "risk_per_trade_pct": 0.015,
                    "daily_loss_limit_pct": 0.06,
                    "session_loss_limit_pct": 0.12,
                    "max_consecutive_losses": 5,
                }
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.1},
                risk_engine=risk,
                dry_run=False,
            )

            decision = StrategyDecision(
                action=DecisionAction.BUY,
                reason="TEST_ENTRY_NO_TP",
                strategy="trend_regime_sm",
                sl=99.0,
                tp=None,
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
            self.assertIsNotNone(broker.last_intent)
            assert broker.last_intent is not None
            self.assertIsNone(broker.last_intent.tp)

    def test_hold_modify_allows_none_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeBroker(fail_first_precheck=False)
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            risk = RiskEngine(
                {
                    "risk_per_trade_pct": 0.015,
                    "daily_loss_limit_pct": 0.06,
                    "session_loss_limit_pct": 0.12,
                    "max_consecutive_losses": 5,
                }
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.1},
                risk_engine=risk,
                dry_run=False,
            )

            position = Position(
                ticket=7,
                symbol="BTCUSD",
                side=Side.BUY,
                volume=0.1,
                price_open=100.0,
                sl=98.0,
                tp=None,
            )
            broker._positions[position.ticket] = position
            decision = StrategyDecision(
                action=DecisionAction.HOLD,
                reason="TEST_HOLD_UPDATE",
                strategy="trend_regime_sm",
                sl=99.0,
                tp=None,
            )

            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.2},
                decision=decision,
                current_position=position,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.ok)
            self.assertEqual(float(broker._positions[position.ticket].sl), 99.0)
            self.assertIsNone(broker._positions[position.ticket].tp)

    def test_exit_pnl_includes_swap_and_commission_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeBroker()
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            risk = RiskEngine(
                {
                    "risk_per_trade_pct": 0.015,
                    "daily_loss_limit_pct": 0.06,
                    "session_loss_limit_pct": 0.12,
                    "max_consecutive_losses": 5,
                }
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.1},
                risk_engine=risk,
                dry_run=False,
            )
            position = Position(
                ticket=9,
                symbol="BTCUSD",
                side=Side.BUY,
                volume=1.0,
                price_open=100.0,
                sl=95.0,
                tp=120.0,
                magic=0,
                metadata={"swap": -1.0, "commission": -2.0},
            )
            broker._positions[position.ticket] = position
            decision = StrategyDecision(
                action=DecisionAction.EXIT,
                reason="TEST_EXIT",
                strategy="trend_regime_sm",
            )

            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.2},
                decision=decision,
                current_position=position,
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.ok)
            self.assertAlmostEqual(float(result.pnl), -2.0)

    def test_exit_uses_partial_volume_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeBroker(fail_first_precheck=False)
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            risk = RiskEngine(
                {
                    "risk_per_trade_pct": 0.015,
                    "daily_loss_limit_pct": 0.06,
                    "session_loss_limit_pct": 0.12,
                    "max_consecutive_losses": 5,
                }
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.1},
                risk_engine=risk,
                dry_run=False,
            )
            position = Position(
                ticket=10,
                symbol="BTCUSD",
                side=Side.BUY,
                volume=0.10,
                price_open=100.0,
                sl=95.0,
                tp=120.0,
                magic=0,
            )
            broker._positions[position.ticket] = position
            decision = StrategyDecision(
                action=DecisionAction.EXIT,
                reason="STAGE_A_PARTIAL_CLOSE",
                strategy="trend_regime_sm",
                volume=0.05,
            )

            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.1},
                decision=decision,
                current_position=position,
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.ok)
            self.assertAlmostEqual(float(broker.last_close_volume or 0.0), 0.05)
            self.assertTrue(bool(decision.metadata.get("is_partial", False)))
            self.assertEqual(float(decision.metadata.get("position_volume_before", 0.0)), 0.10)

    def test_entry_uses_fx_rate_for_cross_currency_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeBroker(fail_first_precheck=False, account_currency="KRW")
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            risk = RiskEngine(
                {
                    "risk_per_trade_pct": 0.015,
                    "daily_loss_limit_pct": 0.06,
                    "session_loss_limit_pct": 0.12,
                    "max_consecutive_losses": 5,
                }
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.2},
                risk_engine=risk,
                dry_run=False,
            )

            decision = StrategyDecision(
                action=DecisionAction.SELL,
                reason="FX_ENTRY_TEST",
                strategy="trend_regime_sm",
                sl=86.3,
                tp=85.0,
                metadata={"signal_close": 85.3},
            )
            result = manager.process_decision(
                instrument={"symbol": "SILVER", "volume": 0.2},
                decision=decision,
                current_position=None,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.ok)
            self.assertIsNotNone(broker.last_intent)
            assert broker.last_intent is not None
            self.assertAlmostEqual(float(broker.last_intent.volume), 0.03)

    def test_entry_risk_plan_failure_returns_error_result_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = _FakeBroker(fail_first_precheck=False)
            store = JsonStore(
                state_path=Path(tmpdir) / "state.json",
                events_path=Path(tmpdir) / "events.jsonl",
            )
            risk = RiskEngine(
                {
                    "risk_per_trade_pct": 0.001,
                    "daily_loss_limit_pct": 0.06,
                    "session_loss_limit_pct": 0.12,
                    "max_consecutive_losses": 5,
                }
            )
            manager = OrderManager(
                broker=broker,
                store=store,
                notifier=_NoopNotifier(),
                execution_cfg={"default_volume": 0.2},
                risk_engine=risk,
                dry_run=False,
            )

            decision = StrategyDecision(
                action=DecisionAction.BUY,
                reason="TEST_RISK_PLAN_FAIL",
                strategy="trend_regime_sm",
                sl=100.0,
                tp=600.0,
                metadata={"signal_close": 400.0},
            )
            result = manager.process_decision(
                instrument={"symbol": "BTCUSD", "volume": 0.2},
                decision=decision,
                current_position=None,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertFalse(result.ok)
            self.assertEqual(result.status, "RISK_PLAN_FAILED")
            self.assertIn("MIN_VOLUME_EXCEEDS_RISK_LIMIT", result.message)


if __name__ == "__main__":
    unittest.main()
