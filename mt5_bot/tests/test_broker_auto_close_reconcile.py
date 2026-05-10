import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.models import Position, Side
from core.runtime import TradingRuntime


class _FakeStore:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def append_event(self, payload: Dict[str, Any]) -> None:
        self.events.append(dict(payload))


class _FakeBroker:
    def __init__(self, close_info: Dict[str, Any]) -> None:
        self._close_info = dict(close_info)
        self.calls: List[int] = []

    def get_position_close_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        self.calls.append(int(ticket))
        info = dict(self._close_info)
        info["ticket"] = int(ticket)
        return info


class BrokerAutoCloseReconcileTests(unittest.TestCase):
    def test_position_disappeared_emits_exit_and_ledgers(self) -> None:
        runtime = TradingRuntime.__new__(TradingRuntime)
        runtime.store = _FakeStore()
        runtime.broker = _FakeBroker(
            {
                "pnl": -0.45,
                "exit_price": 99.0,
                "close_reason": "SL",
                "close_time_utc": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            }
        )
        runtime.config = {"universe": [{"symbol": "BTCUSD", "strategy": "legacy_lsr"}]}
        runtime._last_positions_by_ticket = {
            "123": Position(
                ticket=123,
                symbol="BTCUSD",
                side=Side.BUY,
                volume=0.01,
                price_open=100.0,
                time_open_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        }
        runtime._recent_closed_tickets = {}

        closed_calls: List[Dict[str, Any]] = []

        def _on_position_closed(*, symbol: str, position: Position, result: Any, reason: str, hold_seconds: Any = None) -> None:
            closed_calls.append(
                {
                    "symbol": symbol,
                    "ticket": int(position.ticket),
                    "reason": reason,
                    "pnl": getattr(result, "pnl", None),
                    "hold_seconds": hold_seconds,
                }
            )

        runtime._on_position_closed = _on_position_closed

        runtime._detect_and_record_broker_closed_positions([])

        self.assertEqual(runtime.broker.calls, [123])
        events = runtime.store.events
        self.assertGreaterEqual(sum(1 for e in events if e.get("event") == "position_exit"), 1)
        self.assertGreaterEqual(sum(1 for e in events if e.get("event") == "trade_ledger"), 1)
        self.assertGreaterEqual(sum(1 for e in events if e.get("event") == "trade_ledger_normalized"), 1)

        exit_events = [e for e in events if e.get("event") == "position_exit"]
        self.assertEqual(int(exit_events[0].get("result", {}).get("ticket", 0)), 123)
        self.assertTrue(str(exit_events[0].get("reason", "")).startswith("BROKER_AUTO_CLOSE"))

        ledgers = [e for e in events if e.get("event") == "trade_ledger"]
        self.assertEqual(int(ledgers[0].get("ticket", 0)), 123)
        self.assertEqual(str(ledgers[0].get("strategy", "")), "legacy_lsr")
        self.assertEqual(str(ledgers[0].get("pnl_status", "")), "known")
        self.assertAlmostEqual(float(ledgers[0].get("realized_pnl", 0.0) or 0.0), -0.45, places=8)

        self.assertEqual(len(closed_calls), 1)
        self.assertEqual(int(closed_calls[0].get("ticket", 0)), 123)


if __name__ == "__main__":
    unittest.main()

