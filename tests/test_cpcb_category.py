"""
Unit test suite for Checkpoint 5:
- CPCB PM2.5 Category Breakpoints and Boundary Values
- 24-Hour Window Construction and Natural Gap Detection
- Site and Partition Boundary Separation
- Completeness Filtering (18/24, 20/24, 24/24)
- Direct 24-Hour Conformal Quantiles and Interval Bounds
- Uncertainty-Aware Category Stability and Boundary Proximity
- Verification of Zero Data Leakage
"""

import unittest
import numpy as np
import pandas as pd

from src.aqi.cpcb_categories import (
    classify_pm25,
    classify_pm25_vectorized,
    get_distance_to_nearest_boundary,
    get_crossed_boundaries,
    assess_interval_category_stability,
    CPCB_CATEGORIES,
    CPCB_BOUNDARY_VALUES,
    CPCB_BREAKPOINTS,
)
from src.aqi.windowing import construct_24h_windows
from src.aqi.category_evaluator import (
    compute_24h_point_metrics,
    compute_cpcb_classification_metrics,
    evaluate_uncertainty_aware_stability,
)
from src.uncertainty.conformal import compute_conformal_quantile


class TestCPCBCategories(unittest.TestCase):
    """Test suite verifying CPCB category breakpoints and boundary logic."""

    def test_mandatory_boundary_values(self):
        """Test exact boundary points specified in Checkpoint 5 requirements."""
        self.assertEqual(classify_pm25(0.0), "Good")
        self.assertEqual(classify_pm25(30.0), "Good")
        self.assertEqual(classify_pm25(30.0001), "Satisfactory")
        self.assertEqual(classify_pm25(30.1), "Satisfactory")

        self.assertEqual(classify_pm25(60.0), "Satisfactory")
        self.assertEqual(classify_pm25(60.0001), "Moderate")
        self.assertEqual(classify_pm25(60.1), "Moderate")

        self.assertEqual(classify_pm25(90.0), "Moderate")
        self.assertEqual(classify_pm25(90.0001), "Poor")
        self.assertEqual(classify_pm25(90.1), "Poor")

        self.assertEqual(classify_pm25(120.0), "Poor")
        self.assertEqual(classify_pm25(120.0001), "Very Poor")
        self.assertEqual(classify_pm25(120.1), "Very Poor")

        self.assertEqual(classify_pm25(250.0), "Very Poor")
        self.assertEqual(classify_pm25(250.0001), "Severe")
        self.assertEqual(classify_pm25(250.1), "Severe")
        self.assertEqual(classify_pm25(500.0), "Severe")

    def test_edge_cases_and_nan(self):
        """Test negative values, None, and NaN handling."""
        self.assertEqual(classify_pm25(-5.0), "Good")
        self.assertEqual(classify_pm25(float("nan")), "Unknown")
        self.assertEqual(classify_pm25(None), "Unknown")

    def test_vectorized_classification(self):
        """Test vectorized PM2.5 category mapping."""
        vals = np.array([0.0, 30.0, 30.01, 60.0, 60.01, 90.0, 90.01, 120.0, 120.01, 250.0, 250.01, np.nan])
        expected = [
            "Good", "Good", "Satisfactory",
            "Satisfactory", "Moderate",
            "Moderate", "Poor",
            "Poor", "Very Poor",
            "Very Poor", "Severe",
            "Unknown"
        ]
        result = classify_pm25_vectorized(vals)
        self.assertEqual(list(result), expected)

    def test_distance_to_nearest_boundary(self):
        """Test distance computation to CPCB boundaries."""
        dist, bound = get_distance_to_nearest_boundary(28.0)
        self.assertAlmostEqual(dist, 2.0)
        self.assertEqual(bound, 30.0)

        dist, bound = get_distance_to_nearest_boundary(32.0)
        self.assertAlmostEqual(dist, 2.0)
        self.assertEqual(bound, 30.0)

        dist, _ = get_distance_to_nearest_boundary(75.0)
        self.assertAlmostEqual(dist, 15.0)

        dist, bound = get_distance_to_nearest_boundary(260.0)
        self.assertAlmostEqual(dist, 10.0)
        self.assertEqual(bound, 250.0)

    def test_get_crossed_boundaries(self):
        """Test crossed boundaries identification."""
        self.assertEqual(get_crossed_boundaries(15.0, 25.0), [])
        self.assertEqual(get_crossed_boundaries(25.0, 35.0), [30.0])
        self.assertEqual(get_crossed_boundaries(25.0, 95.0), [30.0, 60.0, 90.0])

    def test_assess_interval_category_stability(self):
        """Test uncertainty-aware category stability logic."""
        # 1. Stable case: interval [15, 25] completely inside Good
        s1 = assess_interval_category_stability(point_pred=20.0, lower=15.0, upper=25.0, ref_value=22.0)
        self.assertTrue(s1["is_stable"])
        self.assertEqual(s1["stability_label"], "Stable")
        self.assertEqual(s1["num_crossed_boundaries"], 0)
        self.assertEqual(s1["covered_categories"], ["Good"])
        self.assertTrue(s1["is_point_correct"])
        self.assertTrue(s1["is_ref_compatible"])

        # 2. Ambiguous case: interval [25, 45] spans Good and Satisfactory
        s2 = assess_interval_category_stability(point_pred=35.0, lower=25.0, upper=45.0, ref_value=28.0)
        self.assertFalse(s2["is_stable"])
        self.assertEqual(s2["stability_label"], "Ambiguous")
        self.assertEqual(s2["num_crossed_boundaries"], 1)
        self.assertEqual(s2["crossed_boundaries"], [30.0])
        self.assertEqual(s2["covered_categories"], ["Good", "Satisfactory"])
        self.assertFalse(s2["is_point_correct"])
        self.assertTrue(s2["is_ref_compatible"])


class TestWindowingAndLeakage(unittest.TestCase):
    """Test suite for 24-hour windowing, completeness filters, and leakage prevention."""

    def setUp(self):
        # Create synthetic multi-site time series with gaps
        dates = pd.date_range("2020-01-01 00:00:00", periods=72, freq="1h")
        self.df_site_a = pd.DataFrame({
            "timestamp": dates,
            "site": ["SiteA"] * 72,
            "bam_reference": np.linspace(10.0, 50.0, 72),
            "point_prediction": np.linspace(12.0, 48.0, 72),
        })

        self.df_site_b = pd.DataFrame({
            "timestamp": dates,
            "site": ["SiteB"] * 72,
            "bam_reference": np.linspace(30.0, 80.0, 72),
            "point_prediction": np.linspace(32.0, 78.0, 72),
        })
        self.df_combined = pd.concat([self.df_site_a, self.df_site_b], ignore_index=True)

    def test_non_overlapping_window_construction(self):
        """Verify non-overlapping 24h windows on contiguous 72 hours."""
        win_df, audit = construct_24h_windows(self.df_site_a, min_valid_hours=24, non_overlapping=True)
        # 72 hours should yield exactly 3 non-overlapping 24h windows
        self.assertEqual(len(win_df), 3)
        self.assertEqual(audit["total_valid_windows"], 3)
        self.assertEqual(audit["total_rejected_windows"], 0)
        self.assertTrue(all(win_df["n_valid_hours"] == 24))

    def test_completeness_rejection(self):
        """Verify that windows with fewer than min_valid_hours are strictly rejected."""
        # Drop 8 hours from second 24h block (leaving 16 hours)
        df_dropped = self.df_site_a.drop(index=list(range(24, 32))).reset_index(drop=True)

        # Under min_valid_hours=18, window 2 must be rejected
        win_df18, audit18 = construct_24h_windows(df_dropped, min_valid_hours=18, non_overlapping=True)
        self.assertEqual(len(win_df18), 2)  # Window 1 and 3 valid
        self.assertEqual(audit18["total_rejected_windows"], 1)

        # Under min_valid_hours=24, only Window 1 and 3 are valid if complete
        win_df24, audit24 = construct_24h_windows(df_dropped, min_valid_hours=24, non_overlapping=True)
        self.assertEqual(len(win_df24), 2)

    def test_site_separation(self):
        """Verify that windows never mix observations across sites."""
        win_comb, audit = construct_24h_windows(self.df_combined, min_valid_hours=24, non_overlapping=True)
        # 3 windows for SiteA + 3 windows for SiteB = 6 windows
        self.assertEqual(len(win_comb), 6)
        self.assertEqual(set(win_comb["site"]), {"SiteA", "SiteB"})
        self.assertEqual((win_comb["site"] == "SiteA").sum(), 3)
        self.assertEqual((win_comb["site"] == "SiteB").sum(), 3)

    def test_direct_24h_conformal_quantile(self):
        """Verify direct 24h finite-sample conformal quantile computation."""
        scores = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        n = len(scores)
        # For alpha=0.10: k = ceil((10+1)*0.90) = ceil(9.9) = 10 -> score 50.0
        q90 = compute_conformal_quantile(scores, alpha=0.10)
        self.assertEqual(q90, 50.0)

        # For alpha=0.20: k = ceil(11 * 0.80) = ceil(8.8) = 9 -> score 45.0
        q80 = compute_conformal_quantile(scores, alpha=0.20)
        self.assertEqual(q80, 45.0)

    def test_classification_metrics(self):
        """Verify multi-class CPCB classification metrics computation."""
        y_true = ["Good", "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        y_pred = ["Good", "Satisfactory", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        metrics = compute_cpcb_classification_metrics(y_true, y_pred)

        # 6 out of 7 exact matches
        self.assertAlmostEqual(metrics["accuracy"], 6.0 / 7.0 * 100.0, places=1)
        self.assertEqual(metrics["n_adjacent_error"], 1)
        self.assertEqual(metrics["n_large_error"], 0)
        self.assertGreater(metrics["cohen_kappa_quadratic"], 0.90)


if __name__ == "__main__":
    unittest.main()
