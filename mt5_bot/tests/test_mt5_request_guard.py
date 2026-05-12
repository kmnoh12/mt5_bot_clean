import os
import unittest
from unittest.mock import patch

import brokers.mt5_live as mt5_live
from brokers.mt5_live import MT5LiveGateway
from core.models import OrderIntent, Position, Side


class _FakeResult:
    def __init__(self) -> None:
        self.retcode = 10009
        self.comment = "Request executed"
        self.order = 123
        self.price = 1.0

    def _asdict(self):
        return {
            "retcode": self.retcode,
            "comment": self.comment,
            "order": self.order,
            "price": self.price,
        }


class _FakeMt5:
    def __init__(self) -> None:
        self.calls = []
        self._last_error = (-2, 'Invalid "comment" argument')

    def order_send(self, request):
        self.calls.append(dict(request))
        if request.get("comment", "") != "":
            self._last_error = (-2, 'Invalid "comment" argument')
            return None
        self._last_error = (0, "OK")
        return _FakeResult()

    def last_error(self):
        return self._last_error


class _ExplodingTradeMt5:
    def order_check(self, request):
        raise AssertionError("order_check must not be called while live order gate is closed")

    def order_send(self, request):
        raise AssertionError("order_send must not be called while live order gate is closed")


class _FakeNotifier:
    def __init__(self) -> None:
        self.errors = []

    def send_error(self, message: str) -> None:
        self.errors.append(str(message))


class MT5RequestGuardTests(unittest.TestCase):
    def test_live_order_gate_blocks_trade_apis_by_default(self) -> None:
        old_mt5 = mt5_live.mt5
        mt5_live.mt5 = _ExplodingTradeMt5()
        try:
            gateway = MT5LiveGateway(config={"mt5": {}, "general": {"dry_run": True}, "execution": {}})
            gateway.connected = True
            intent = OrderIntent(
                symbol="BTCUSD",
                side=Side.BUY,
                volume=0.01,
                reason="test",
                strategy="unit",
            )
            position = Position(
                ticket=7,
                symbol="BTCUSD",
                side=Side.BUY,
                volume=0.01,
                price_open=100.0,
                magic=1,
            )

            results = [
                gateway.precheck_order(intent),
                gateway.submit_order(intent),
                gateway.send_order(intent),
                gateway.modify_position_sl_tp(position, sl=99.0, tp=None, reason="unit"),
                gateway.close_position(position, reason="unit"),
            ]
            results.extend(gateway.close_all_positions(reason="unit"))

            self.assertTrue(results)
            for result in results:
                self.assertFalse(result.ok)
                self.assertEqual(result.status, "LIVE_TRADING_BLOCKED")
                self.assertIn("live_order_gate", result.raw)
        finally:
            mt5_live.mt5 = old_mt5

    def test_live_order_gate_requires_env_confirmation(self) -> None:
        with patch.dict(os.environ, {"MT5_ALLOW_LIVE_TRADING": "YES_I_ACCEPT_RISK"}, clear=False):
            gateway = MT5LiveGateway(
                config={
                    "mt5": {},
                    "general": {"dry_run": False},
                    "execution": {"live_trading_enabled": True},
                }
            )
        self.assertTrue(gateway.orders_allowed)

    def test_retry_without_comment_on_invalid_comment(self) -> None:
        old_mt5 = mt5_live.mt5
        fake = _FakeMt5()
        mt5_live.mt5 = fake
        try:
            gateway = MT5LiveGateway(config={"mt5": {}, "general": {}, "broker_request_guard": {}})
            request = {"action": 1, "symbol": "BTCUSD", "comment": "close:한글#bad"}
            res, meta, code, message = gateway._order_send_with_comment_retry(request, context="test")
            self.assertIsNotNone(res)
            self.assertIsNone(code)
            self.assertIsNone(message)
            self.assertTrue(meta["comment_sanitized_changed"])
            self.assertTrue(meta["retried_without_comment"])
            self.assertTrue(meta["retry_success"])
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(fake.calls[1].get("comment"), "")
        finally:
            mt5_live.mt5 = old_mt5

    def test_ipc_threshold_does_not_trigger_fatal_stop(self) -> None:
        notifier = _FakeNotifier()
        gateway = MT5LiveGateway(
            config={
                "mt5": {},
                "general": {"reconnect": {"max_ipc_failures_before_halt": 3}},
                "broker_request_guard": {},
            },
            notifier=notifier,
        )
        for _ in range(4):
            self.assertTrue(gateway._handle_ipc_failure(context="submit_order:BTCUSD", code=-10005, message="IPC timeout"))
        self.assertIsNone(gateway.fatal_error())
        self.assertTrue(gateway._ipc_threshold_reported)
        self.assertGreaterEqual(len(notifier.errors), 1)


if __name__ == "__main__":
    unittest.main()
