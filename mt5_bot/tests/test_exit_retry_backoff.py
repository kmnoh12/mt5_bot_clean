import unittest

from execution.exit_retry_guard import ExitRetryGuard


class ExitRetryGuardTests(unittest.TestCase):
    def test_backoff_progression(self) -> None:
        guard = ExitRetryGuard()
        allow, _, attempt = guard.should_allow(ticket=100, reason="R1", now_ts=1000.0)
        self.assertTrue(allow)
        self.assertEqual(attempt, 0)

        fail1 = guard.on_attempt(ticket=100, reason="R1", now_ts=1000.0, success=False)
        self.assertEqual(fail1["backoff_seconds"], 30.0)

        allow2, cooldown2, attempt2 = guard.should_allow(ticket=100, reason="R1", now_ts=1010.0)
        self.assertFalse(allow2)
        self.assertGreater(cooldown2, 0)
        self.assertEqual(attempt2, 1)

        allow3, _, attempt3 = guard.should_allow(ticket=100, reason="R1", now_ts=1031.0)
        self.assertTrue(allow3)
        self.assertEqual(attempt3, 1)

        fail2 = guard.on_attempt(ticket=100, reason="R1", now_ts=1031.0, success=False)
        self.assertEqual(fail2["backoff_seconds"], 60.0)

        success = guard.on_attempt(ticket=100, reason="R1", now_ts=1100.0, success=True)
        self.assertTrue(success["success"])
        allow4, _, attempt4 = guard.should_allow(ticket=100, reason="R1", now_ts=1101.0)
        self.assertTrue(allow4)
        self.assertEqual(attempt4, 0)


if __name__ == "__main__":
    unittest.main()
