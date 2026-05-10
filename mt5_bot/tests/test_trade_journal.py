import tempfile
import unittest
from pathlib import Path

from execution.trade_journal import TradeJournal


class TradeJournalTests(unittest.TestCase):
    def test_writes_markdown_and_flags_tiny_quick_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = TradeJournal(
                {
                    "enabled": True,
                    "output_dir": tmpdir,
                    "tiny_pnl_threshold_usd": 2.0,
                    "quick_exit_window_seconds": 300,
                    "big_loss_threshold_usd": -10.0,
                }
            )
            out = journal.record_trade(
                symbol="BTCUSD",
                reason="TEST_EXIT",
                pnl=1.0,
                hold_seconds=120,
                entry_price=100.0,
                exit_price=101.0,
            )
            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(out["classification"], "CHURN")
            target = Path(out["path"])
            self.assertTrue(target.exists())
            text = target.read_text(encoding="utf-8")
            self.assertIn("BTCUSD", text)
            self.assertIn("class=CHURN", text)


if __name__ == "__main__":
    unittest.main()
