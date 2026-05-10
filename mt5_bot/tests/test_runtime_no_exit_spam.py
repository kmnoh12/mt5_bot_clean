import unittest

from execution.exit_retry_guard import ExitRetryGuard


class RuntimeNoExitSpamTests(unittest.TestCase):
    def test_backoff_prevents_minute_level_exit_spam(self) -> None:
        guard = ExitRetryGuard()
        allow1, _, _ = guard.should_allow(ticket=9, reason="TREND_REGIME_EXIT:BUY_TRAIL_BREACH", now_ts=1000.0)
        self.assertTrue(allow1)
        guard.on_attempt(ticket=9, reason="TREND_REGIME_EXIT:BUY_TRAIL_BREACH", now_ts=1000.0, success=False)

        # Within first backoff window (30s), retry should be blocked.
        allow2, cooldown, _ = guard.should_allow(ticket=9, reason="TREND_REGIME_EXIT:BUY_TRAIL_BREACH", now_ts=1010.0)
        self.assertFalse(allow2)
        self.assertGreater(cooldown, 0.0)


if __name__ == "__main__":
    unittest.main()
