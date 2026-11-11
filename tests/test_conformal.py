"""
Unit tests for Conformal Uncertainty Estimation (Checkpoint 4):
- Finite-sample quantile formula: k = ceil((n + 1)(1 - alpha))
- Alpha edge cases (alpha -> 0, alpha -> 1)
- Winkler interval score calculation
- Empirical coverage and interval width calculations
- Prediction-time regime assignment (Humidity and Predicted Pollution)
- No BAM leakage into regime assignments
- Minimum group threshold and pooled fallback logic
- Temporal block construction strictly within segment and site boundaries
- Split manager four-way chronological partition integrity
"""

import unittest
import numpy as np
import pandas as pd

from src.uncertainty.conformal import (
    compute_conformal_quantile,
    compute_winkler_interval_score,
    compute_uncertainty_metrics,
    ConformalCalibrator,
)
from src.uncertainty.regime_conformal import (
    assign_humidity_regime,
    assign_predicted_pollution_regime,
    RegimeConformalCalibrator,
)
from src.uncertainty.block_conformal import (
    construct_temporal_blocks,
    TemporalBlockConformalCalibrator,
)
from src.uncertainty.split_manager import create_conformal_splits


class TestConformalQuantiles(unittest.TestCase):
    def test_finite_sample_quantile_exact_formula(self):
        """Test exact order statistic rank k = ceil((n + 1)(1 - alpha))."""
        # n = 99 calibration scores: 1.0, 2.0, ..., 99.0
        scores = np.arange(1.0, 100.0)
        n = len(scores)  # 99

        # For alpha = 0.10: k = ceil(100 * 0.90) = 90 -> score = 90.0
        q_90 = compute_conformal_quantile(scores, alpha=0.10)
        self.assertEqual(q_90, 90.0)

        # For alpha = 0.05: k = ceil(100 * 0.95) = 95 -> score = 95.0
        q_95 = compute_conformal_quantile(scores, alpha=0.05)
        self.assertEqual(q_95, 95.0)

    def test_quantile_finite_sample_conservative_inflation(self):
        """Verify that finite-sample formula is slightly more conservative than naive percentile."""
        # n = 19 scores: 1 to 19
        scores = np.arange(1.0, 20.0)
        # k = ceil(20 * 0.90) = 18 -> 18.0
        q_90 = compute_conformal_quantile(scores, alpha=0.10)
        self.assertEqual(q_90, 18.0)

    def test_alpha_boundary_cases(self):
        """Test boundary conditions for small n or extreme alpha."""
        scores = np.array([5.0, 10.0, 15.0])
        # alpha very small -> k = ceil(4 * 0.999) = 4 > 3 -> clamps to max score 15.0
        q_ext = compute_conformal_quantile(scores, alpha=0.001)
        self.assertEqual(q_ext, 15.0)

        # alpha large -> k = ceil(4 * 0.05) = 1 -> clamps to min score 5.0
        q_low = compute_conformal_quantile(scores, alpha=0.95)
        self.assertEqual(q_low, 5.0)

    def test_winkler_interval_score_penalties(self):
        """Verify Winkler score correctly penalizes both width and under/over coverage."""
        y_true = np.array([50.0, 10.0, 90.0])
        lower = np.array([40.0, 20.0, 40.0])  # point 0: covered (40-60); point 1: under (y=10 < l=20); point 2: over (y=90 > u=60)
        upper = np.array([60.0, 40.0, 60.0])
        alpha = 0.10

        score = compute_winkler_interval_score(y_true, lower, upper, alpha)
        # Point 0: width = 20, penalty = 0 -> 20
        # Point 1: width = 20, penalty = (2 / 0.10) * (20 - 10) = 200 -> 220
        # Point 2: width = 20, penalty = (2 / 0.10) * (90 - 60) = 600 -> 620
        # Mean = (20 + 220 + 620) / 3 = 860 / 3 = 286.67
        self.assertAlmostEqual(score, 860.0 / 3.0, places=2)

    def test_conformal_calibrator_predict_interval(self):
        calibrator = ConformalCalibrator()
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0, 52.0, 58.0, 72.0, 78.0, 92.0, 98.0])  # residual always 2.0
        calibrator.fit(y_true, y_pred, alphas=[0.10])

        test_pred = np.array([50.0])
        low, up = calibrator.predict_interval(test_pred, alpha=0.10)
        self.assertAlmostEqual(low[0], 48.0)
        self.assertAlmostEqual(up[0], 52.0)


class TestRegimeConformal(unittest.TestCase):
    def test_humidity_regime_assignment(self):
        rh = np.array([25.0, 50.0, 75.0, np.nan])
        regimes = assign_humidity_regime(rh)
        self.assertEqual(regimes[0], "Low (<40% RH)")
        self.assertEqual(regimes[1], "Medium (40-65% RH)")
        self.assertEqual(regimes[2], "High (>65% RH)")
        self.assertEqual(regimes[3], "Unknown RH")

    def test_predicted_pollution_regime_assignment(self):
        """Verify predicted pollution regimes rely strictly on y_pred, not ground-truth BAM."""
        y_pred = np.array([35.0, 95.0, 180.0])
        regimes = assign_predicted_pollution_regime(y_pred)
        self.assertEqual(regimes[0], "Low (<=60 ug/m3)")
        self.assertEqual(regimes[1], "Moderate (60-120 ug/m3)")
        self.assertEqual(regimes[2], "Severe (>120 ug/m3)")

    def test_minimum_sample_threshold_fallback(self):
        """Verify small sample regime falls back to pooled quantile."""
        np.random.seed(42)
        n = 200
        y_t = np.random.uniform(20, 100, size=n)
        y_p = y_t + np.random.normal(0, 5, size=n)

        # Regimes: Group A has 190 samples, Group B has only 10 samples (< min_group_size=30)
        regimes = np.array(["GroupA"] * 190 + ["GroupB"] * 10)

        calibrator = RegimeConformalCalibrator(conditioning_variable="test", min_group_size=30, alphas=[0.10])
        calibrator.fit(y_t, y_p, regimes)

        self.assertFalse(calibrator.fallback_flags["GroupA"])
        self.assertTrue(calibrator.fallback_flags["GroupB"])

        # Check prediction fallback flag
        test_pred = np.array([50.0, 50.0])
        test_reg = np.array(["GroupA", "GroupB"])
        low, up, fb = calibrator.predict_intervals(test_pred, test_reg, alpha=0.10)
        self.assertFalse(fb[0])
        self.assertTrue(fb[1])


class TestBlockConformal(unittest.TestCase):
    def test_temporal_block_construction_and_boundaries(self):
        """Verify blocks do not cross segment or site boundaries."""
        df = pd.DataFrame({
            "site": ["Delhi"] * 25 + ["Bangalore"] * 25,
            "segment_id": [1] * 15 + [2] * 10 + [3] * 25,
            "bam_reference": np.ones(50),
            "y_pred": np.ones(50),
        })

        # With block_size = 6:
        # Segment 1 (15 rows): 2 blocks of 6 = 12 rows (3 discarded)
        # Segment 2 (10 rows): 1 block of 6 = 6 rows (4 discarded)
        # Segment 3 (25 rows): 4 blocks of 6 = 24 rows (1 discarded)
        # Total blocks = 2 + 1 + 4 = 7
        blocks = construct_temporal_blocks(df, block_size=6)
        self.assertEqual(len(blocks), 7)
        for b in blocks:
            self.assertEqual(len(b), 6)
            # Ensure no block crosses segment
            self.assertEqual(len(np.unique(df.iloc[b]["segment_id"])), 1)
            # Ensure no block crosses site
            self.assertEqual(len(np.unique(df.iloc[b]["site"])), 1)


class TestSplitManager(unittest.TestCase):
    def test_four_way_chronological_splits(self):
        """Verify 60/10/10/20 chronological partitions maintain total count and timestamp order."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "timestamp": dates,
            "site": ["Delhi"] * n,
            "bam_reference": np.random.uniform(10, 100, size=n),
            "pa_cf1": np.random.uniform(10, 100, size=n),
            "pa_atm": np.random.uniform(10, 100, size=n),
            "pa_alt": np.random.uniform(10, 100, size=n),
            "relative_humidity": np.random.uniform(30, 80, size=n),
            "temperature": np.random.uniform(15, 35, size=n),
            "dew_point_raw": np.ones(n),
            "is_dew_point_suspicious": np.zeros(n),
            "time_diff_hours": np.ones(n),
            "is_contiguous_step": np.ones(n),
            "segment_id": np.ones(n),
        })

        tr, vm, cc, te, summary = create_conformal_splits(df)
        self.assertEqual(len(tr) + len(vm) + len(cc) + len(te), n)
        self.assertEqual(len(tr), 60)
        self.assertEqual(len(vm), 10)
        self.assertEqual(len(cc), 10)
        self.assertEqual(len(te), 20)

        # Verify strict timestamp monotonicity
        self.assertTrue(tr["timestamp"].max() < vm["timestamp"].min())
        self.assertTrue(vm["timestamp"].max() < cc["timestamp"].min())
        self.assertTrue(cc["timestamp"].max() < te["timestamp"].min())


if __name__ == "__main__":
    unittest.main()
