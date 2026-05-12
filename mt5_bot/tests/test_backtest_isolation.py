import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


class BacktestIsolationTests(unittest.TestCase):
    def test_backtest_gateway_loads_csv_without_metatrader5_import(self) -> None:
        real_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "MetaTrader5":
                raise AssertionError("BacktestGateway must not import MetaTrader5")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            csv_path = data_dir / "BTCUSD_TIMEFRAME_M5.csv"
            rows = ["time,open,high,low,close,volume"]
            for index in range(80):
                price = 100.0 + index
                rows.append(f"{1700000000 + index * 60},{price},{price + 1},{price - 1},{price + 0.5},10")
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            with patch("builtins.__import__", side_effect=guarded_import):
                from brokers.backtest import BacktestGateway

                gateway = BacktestGateway(
                    universe=[{"symbol": "BTCUSD", "timeframe": "TIMEFRAME_M5"}],
                    general_cfg={"bars_per_request": 50},
                    backtest_cfg={"data_dir": str(data_dir)},
                    execution_cfg={},
                )

            self.assertTrue(gateway.connect())
            bars = gateway.fetch_bars("BTCUSD", "TIMEFRAME_M5", 10)
            self.assertIsNotNone(bars)
            assert bars is not None
            self.assertEqual(len(bars), 10)


if __name__ == "__main__":
    unittest.main()
