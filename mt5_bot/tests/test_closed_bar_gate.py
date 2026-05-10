import unittest

import pandas as pd

from core.bar_gate import ClosedBarGate


class ClosedBarGateTests(unittest.TestCase):
    def test_should_evaluate_only_on_new_closed_bar(self) -> None:
        gate = ClosedBarGate()
        bars = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:01:00Z",
                        "2026-01-01T00:02:00Z",
                    ],
                    utc=True,
                ),
                "open": [1.0, 1.1, 1.2],
                "high": [1.2, 1.3, 1.4],
                "low": [0.9, 1.0, 1.1],
                "close": [1.1, 1.2, 1.3],
            }
        )

        should_eval, closed_time = gate.should_evaluate("BTCUSD", bars)
        self.assertTrue(should_eval)
        self.assertIsNotNone(closed_time)

        should_eval_again, _ = gate.should_evaluate("BTCUSD", bars)
        self.assertFalse(should_eval_again)

        new_row = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01T00:03:00Z"], utc=True),
                "open": [1.3],
                "high": [1.5],
                "low": [1.2],
                "close": [1.4],
            }
        )
        bars2 = pd.concat([bars, new_row], ignore_index=True)

        should_eval_new, _ = gate.should_evaluate("BTCUSD", bars2)
        self.assertTrue(should_eval_new)


if __name__ == "__main__":
    unittest.main()
