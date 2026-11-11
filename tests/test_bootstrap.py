"""
Unit tests for paired moving-block bootstrap in src.evaluation.bootstrap.
"""

import unittest
import numpy as np

from src.evaluation.bootstrap import (
    paired_moving_block_bootstrap,
    block_length_sensitivity_sweep,
    get_segment_block_indices,
)


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = 200
        self.y_true = np.sin(np.linspace(0, 10, self.n)) * 20 + 50
        # Model 1 (Joint): lower error
        self.y_joint = self.y_true + np.random.normal(0, 2.0, self.n)
        # Model 2 (Baseline): higher error
        self.y_base = self.y_true + np.random.normal(0, 5.0, self.n)
        self.segments = np.repeat([1, 2], 100)

    def test_block_indices_respect_segments(self):
        blocks = get_segment_block_indices(self.segments, block_length=24)
        for start, end in blocks:
            # Segment at start must equal segment at end-1
            self.assertEqual(self.segments[start], self.segments[end - 1])

    def test_paired_bootstrap_detects_significant_difference(self):
        res = paired_moving_block_bootstrap(
            self.y_true, self.y_joint, self.y_base, self.segments,
            block_length=24, n_bootstraps=200, random_seed=42
        )
        self.assertIn("delta_rmse_ci", res)
        # delta_rmse should be negative (joint has lower error)
        self.assertLess(res["point_estimates"]["delta_rmse"], 0.0)
        self.assertLess(res["delta_rmse_ci"][1], 0.0)
        self.assertTrue(res["statistically_significant_05"])

    def test_sensitivity_sweep(self):
        df_sweep = block_length_sensitivity_sweep(
            self.y_true, self.y_joint, self.y_base, self.segments,
            block_lengths=[12, 24], n_bootstraps=100, random_seed=42
        )
        self.assertEqual(len(df_sweep), 2)
        self.assertIn("ci_width", df_sweep.columns)


if __name__ == "__main__":
    unittest.main()
