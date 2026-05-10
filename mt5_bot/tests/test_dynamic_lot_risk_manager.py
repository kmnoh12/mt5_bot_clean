import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from core.models import SymbolConstraints
from execution.risk_manager import RiskEngine
from execution import risk_manager as risk_manager_module


class _FakeMt5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(
        self,
        *,
        balance: float = 10_000.0,
        currency: str = "USD",
        min_volume: float = 0.01,
        max_volume: float = 100.0,
        volume_step: float = 0.01,
        point: float = 0.01,
        ask: float = 100.0,
        bid: float = 99.9,
        order_calc_mode: str = "normal",
        pip_value_per_lot: float = 100.0,
    ) -> None:
        self._account = SimpleNamespace(balance=balance, currency=currency)
        self._info = SimpleNamespace(
            volume_min=min_volume,
            volume_max=max_volume,
            volume_step=volume_step,
            point=point,
        )
        self._tick = SimpleNamespace(ask=ask, bid=bid)
        self._last_error = (0, "OK")
        self._order_calc_mode = order_calc_mode
        self._pip_value_per_lot = float(pip_value_per_lot)

    def last_error(self):
        return self._last_error

    def symbol_info(self, symbol: str):
        return self._info

    def symbol_select(self, symbol: str, selected: bool):
        return True

    def symbol_info_tick(self, symbol: str):
        return self._tick

    def account_info(self):
        return self._account

    def order_calc_profit(self, action: int, symbol: str, volume: float, price_open: float, price_close: float):
        if self._order_calc_mode == "none":
            self._last_error = (-1, "order_calc_profit failed")
            return None
        if self._order_calc_mode == "zero":
            return 0.0
        distance = abs(float(price_open) - float(price_close))
        return -distance * self._pip_value_per_lot * float(volume)


class DynamicLotRiskManagerTests(unittest.TestCase):
    def test_calculate_dynamic_lot_normal(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.01, "dynamic_lot_enabled": True})
        fake_mt5 = _FakeMt5(order_calc_mode="normal")
        with patch.object(risk_manager_module, "mt5", fake_mt5):
            lot = engine.calculate_dynamic_lot(
                symbol="GOLD",
                side="buy",
                sl_price=99.0,
                entry_price=100.0,
                fail_mode="block",
            )
        self.assertIsNotNone(lot)
        self.assertAlmostEqual(float(lot or 0.0), 1.0, places=6)
        self.assertAlmostEqual(float(engine._last_dynamic_lot_meta.get("expected_pnl_usd", 0.0)), -100.0, places=6)

    def test_calculate_dynamic_lot_order_calc_none_fallback(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.01, "dynamic_lot_enabled": True})
        fake_mt5 = _FakeMt5(order_calc_mode="none")
        with patch.object(risk_manager_module, "mt5", fake_mt5):
            lot = engine.calculate_dynamic_lot(
                symbol="GOLD",
                side="buy",
                sl_price=99.0,
                entry_price=100.0,
                default_lot=0.03,
                fail_mode="fallback",
            )
        self.assertIsNotNone(lot)
        self.assertAlmostEqual(float(lot or 0.0), 0.03, places=6)

    def test_calculate_dynamic_lot_order_calc_zero_block(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.01, "dynamic_lot_enabled": True})
        fake_mt5 = _FakeMt5(order_calc_mode="zero")
        with patch.object(risk_manager_module, "mt5", fake_mt5):
            lot = engine.calculate_dynamic_lot(
                symbol="GOLD",
                side="buy",
                sl_price=99.0,
                entry_price=100.0,
                fail_mode="block",
            )
        self.assertIsNone(lot)
        self.assertEqual(engine._last_dynamic_lot_meta.get("reason"), "NON_POSITIVE_EXPECTED_LOSS_1LOT")

    def test_calculate_dynamic_lot_blocks_when_min_volume_exceeds_risk(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.0005,  # 0.05%
                "dynamic_lot_enabled": True,
                "dynamic_lot_min_volume_policy": "block",
            }
        )
        fake_mt5 = _FakeMt5(balance=5_000.0, min_volume=0.1, volume_step=0.1, pip_value_per_lot=100.0)
        with patch.object(risk_manager_module, "mt5", fake_mt5):
            lot = engine.calculate_dynamic_lot(
                symbol="GOLD",
                side="buy",
                sl_price=99.0,
                entry_price=100.0,
                fail_mode="block",
            )
        self.assertIsNone(lot)
        self.assertEqual(engine._last_dynamic_lot_meta.get("reason"), "MIN_VOLUME_EXCEEDS_RISK_LIMIT")

    def test_calculate_dynamic_lot_uses_floor_step(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.0095, "dynamic_lot_enabled": True})  # 0.95%
        fake_mt5 = _FakeMt5(min_volume=0.1, max_volume=10.0, volume_step=0.1, pip_value_per_lot=100.0)
        with patch.object(risk_manager_module, "mt5", fake_mt5):
            lot = engine.calculate_dynamic_lot(
                symbol="GOLD",
                side="buy",
                sl_price=99.0,
                entry_price=100.0,
                fail_mode="block",
            )
        # risk_amount=95, expected_loss_1lot=100 => raw_lot=0.95 -> floor to 0.9
        self.assertIsNotNone(lot)
        self.assertAlmostEqual(float(lot or 0.0), 0.9, places=6)

    def test_plan_entry_volume_uses_dynamic_lot_and_emits_expected_pnl(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.01, "dynamic_lot_enabled": True})
        constraints = SymbolConstraints(min_volume=0.01, max_volume=10.0, volume_step=0.01, contract_size=1.0)
        fake_mt5 = _FakeMt5(order_calc_mode="normal")
        with patch.object(risk_manager_module, "mt5", fake_mt5):
            volume, err, meta = engine.plan_entry_volume(
                constraints=constraints,
                equity=10_000.0,
                entry_price=100.0,
                sl_price=99.0,
                requested_volume=2.0,
                side="buy",
                symbol="GOLD",
            )
        self.assertIsNone(err)
        self.assertAlmostEqual(float(volume or 0.0), 1.0, places=6)
        self.assertEqual(meta.get("volume_source"), "dynamic_order_calc_profit")
        self.assertIn("expected_pnl_usd", meta)


if __name__ == "__main__":
    unittest.main()
