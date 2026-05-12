import unittest

from core.performance_metrics import compute_backtest_metrics, walk_forward_splits


class PerformanceMetricsTests(unittest.TestCase):
    def test_computes_trade_quality_cost_drawdown_and_churn(self) -> None:
        metrics = compute_backtest_metrics(
            [
                {"realized_pnl": 10.0, "commission": -1.0, "hold_seconds": 600},
                {"pnl": -5.0, "realized_cost_usd": 1.0, "hold_seconds": 120},
                {"profit": 1.0, "fee": 0.5, "hold_seconds": 60},
            ],
            tiny_pnl_threshold=2.0,
            quick_exit_seconds=180.0,
        )

        self.assertEqual(metrics["trades"], 3)
        self.assertAlmostEqual(metrics["win_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["avg_win"], 5.5)
        self.assertAlmostEqual(metrics["avg_loss"], -5.0)
        self.assertAlmostEqual(metrics["expectancy"], 2.0)
        self.assertAlmostEqual(metrics["profit_factor"], 11.0 / 5.0)
        self.assertAlmostEqual(metrics["max_drawdown"], 5.0)
        self.assertAlmostEqual(metrics["fees_cost_estimate"], 2.5)
        self.assertAlmostEqual(metrics["exposure_seconds"], 780.0)
        self.assertEqual(metrics["quick_exit_count"], 2)
        self.assertEqual(metrics["churn_count"], 1)

    def test_walk_forward_splits_are_chronological_and_non_overlapping(self) -> None:
        splits = walk_forward_splits(total_rows=100, train_rows=50, test_rows=20, step_rows=20)

        self.assertEqual(
            splits,
            [
                {"fold": 1, "train_start": 0, "train_end": 50, "test_start": 50, "test_end": 70},
                {"fold": 2, "train_start": 20, "train_end": 70, "test_start": 70, "test_end": 90},
            ],
        )


if __name__ == "__main__":
    unittest.main()
