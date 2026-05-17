import unittest

from utils.liquidity import classify_lsr_confirmation_quality


class LsrConfirmationQualityTests(unittest.TestCase):
    def test_unconfirmed_weak_reclaim_chase_is_review_bucketed(self) -> None:
        flags = classify_lsr_confirmation_quality(
            confirmation_path="reclaim_only",
            retest_confirmed=False,
            reclaim_distance_atr=0.2,
            sweep_depth_atr=1.0,
            reclaim_to_sweep_depth_ratio=0.2,
            displacement_ratio=2.4,
        )

        self.assertTrue(flags["lsr_unconfirmed_reclaim"])
        self.assertTrue(flags["weak_reclaim_after_deep_sweep"])
        self.assertTrue(flags["entry_chased_extension"])
        self.assertTrue(flags["lsr_unconfirmed_reclaim_chase"])
        self.assertEqual(flags["review_bucket"], "unconfirmed_reclaim_chase")
        self.assertTrue(flags["review_only"])
        self.assertLess(flags["confirmation_score"], 0.4)
        self.assertEqual(flags["confirmation_band"], "weak")
        self.assertIn("weak_reclaim_after_deep_sweep", flags["confirmation_score_components"])

    def test_retest_confirmed_is_not_marked_unconfirmed(self) -> None:
        flags = classify_lsr_confirmation_quality(
            confirmation_path="retest",
            retest_confirmed=True,
            reclaim_distance_atr=0.3,
            sweep_depth_atr=1.2,
            reclaim_to_sweep_depth_ratio=0.25,
            displacement_ratio=2.5,
        )

        self.assertFalse(flags["lsr_unconfirmed_reclaim"])
        self.assertFalse(flags["weak_reclaim_after_deep_sweep"])
        self.assertTrue(flags["entry_chased_extension"])
        self.assertFalse(flags["lsr_unconfirmed_reclaim_chase"])
        self.assertEqual(flags["review_bucket"], "confirmed_retest")
        self.assertGreaterEqual(flags["confirmation_score"], 0.7)
        self.assertEqual(flags["confirmation_band"], "clean")
        self.assertIn("confirmed_retest", flags["confirmation_score_components"])

    def test_late_unconfirmed_reclaim_is_chase_diagnostic(self) -> None:
        flags = classify_lsr_confirmation_quality(
            confirmation_path="reclaim_only",
            retest_confirmed=False,
            reclaim_distance_atr=0.2,
            sweep_depth_atr=0.2,
            reclaim_to_sweep_depth_ratio=1.0,
            displacement_ratio=1.1,
            time_from_sweep_to_reclaim_sec=780,
            reclaim_window_sec=900,
        )

        self.assertTrue(flags["lsr_unconfirmed_reclaim"])
        self.assertTrue(flags["late_window_reclaim"])
        self.assertAlmostEqual(flags["reclaim_window_elapsed_ratio"], 780 / 900)
        self.assertTrue(flags["lsr_unconfirmed_reclaim_chase"])
        self.assertEqual(flags["review_bucket"], "unconfirmed_reclaim_chase")
        self.assertLess(flags["confirmation_score"], 0.7)

    def test_negative_sweep_age_is_invalid_timing_not_late_reclaim(self) -> None:
        flags = classify_lsr_confirmation_quality(
            confirmation_path="reclaim_only",
            retest_confirmed=False,
            reclaim_distance_atr=0.4,
            sweep_depth_atr=0.8,
            reclaim_to_sweep_depth_ratio=0.5,
            displacement_ratio=1.0,
            time_from_sweep_to_reclaim_sec=-60,
            reclaim_window_sec=900,
        )

        self.assertTrue(flags["invalid_reclaim_timing"])
        self.assertIsNone(flags["reclaim_window_elapsed_ratio"])
        self.assertFalse(flags["late_window_reclaim"])
        self.assertLess(flags["confirmation_score"], 0.4)

    def test_shallow_unconfirmed_reclaim_is_chase_diagnostic(self) -> None:
        flags = classify_lsr_confirmation_quality(
            confirmation_path="tick_reclaim",
            retest_confirmed=False,
            reclaim_distance_atr=0.24,
            sweep_depth_atr=0.3,
            reclaim_to_sweep_depth_ratio=0.8,
            displacement_ratio=1.0,
            time_from_sweep_to_reclaim_sec=20,
            reclaim_window_sec=120,
        )

        self.assertTrue(flags["lsr_unconfirmed_reclaim"])
        self.assertTrue(flags["shallow_reclaim_confirmation"])
        self.assertEqual(flags["shallow_reclaim_threshold_atr"], 0.25)
        self.assertTrue(flags["lsr_unconfirmed_reclaim_chase"])
        self.assertEqual(flags["review_bucket"], "unconfirmed_reclaim_chase")
        self.assertEqual(flags["confirmation_band"], "mixed")

    def test_retest_with_shallow_cushion_is_not_unconfirmed_chase(self) -> None:
        flags = classify_lsr_confirmation_quality(
            confirmation_path="retest",
            retest_confirmed=True,
            reclaim_distance_atr=0.1,
            sweep_depth_atr=1.0,
            reclaim_to_sweep_depth_ratio=0.1,
            displacement_ratio=1.0,
        )

        self.assertFalse(flags["lsr_unconfirmed_reclaim"])
        self.assertFalse(flags["shallow_reclaim_confirmation"])
        self.assertFalse(flags["lsr_unconfirmed_reclaim_chase"])
        self.assertEqual(flags["review_bucket"], "confirmed_retest")
        self.assertEqual(flags["confirmation_band"], "clean")


if __name__ == "__main__":
    unittest.main()
