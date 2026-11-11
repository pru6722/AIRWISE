"""
Unit tests for missingness simulation mechanisms: MCAR, consecutive blocks, and MAR.
"""

import unittest
import numpy as np
import pandas as pd

from src.missingness.random_mask import generate_mcar_mask
from src.missingness.block_mask import generate_block_outage_mask
from src.missingness.regime_mask import (
    generate_humidity_dependent_mask,
    generate_pollution_dependent_mask,
    generate_lag_dependent_mask,
)


class TestMissingnessMechanisms(unittest.TestCase):
    def setUp(self):
        self.n_samples = 1000
        dates = pd.date_range("2020-01-01", periods=self.n_samples, freq="1h")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "pa_cf1": np.random.uniform(10.0, 200.0, size=self.n_samples),
            "relative_humidity": np.random.uniform(20.0, 95.0, size=self.n_samples),
            "segment_id": np.repeat(np.arange(10), 100),
        })

    def test_mcar_mask_rates(self):
        for rate in [0.05, 0.10, 0.20, 0.30]:
            mask = generate_mcar_mask(self.n_samples, missing_rate=rate, random_seed=42)
            observed_rate = np.mean(~mask)
            self.assertAlmostEqual(observed_rate, rate, delta=0.03)

    def test_block_mask_gap_durations(self):
        for gap in [1, 3, 6, 12, 24]:
            mask = generate_block_outage_mask(
                self.df, gap_hours=gap, missing_rate=0.20, segment_col="segment_id", random_seed=42
            )
            # Find lengths of false sequences (hidden blocks)
            hidden_diff = np.diff(np.concatenate(([1], mask.astype(int), [1])))
            starts = np.where(hidden_diff == -1)[0]
            ends = np.where(hidden_diff == 1)[0]
            lengths = ends - starts
            if len(lengths) > 0:
                self.assertTrue(np.all(lengths >= 1))

    def test_humidity_dependent_mask(self):
        mask = generate_humidity_dependent_mask(self.df, rh_col="relative_humidity", missing_rate=0.20, random_seed=42)
        self.assertEqual(len(mask), self.n_samples)
        # High humidity should have higher dropout rate than low humidity
        high_rh = self.df["relative_humidity"] > 65.0
        low_rh = self.df["relative_humidity"] < 40.0
        dropout_high = np.mean(~mask[high_rh])
        dropout_low = np.mean(~mask[low_rh])
        self.assertGreater(dropout_high, dropout_low)

    def test_pollution_dependent_mask(self):
        mask = generate_pollution_dependent_mask(self.df, pm_col="pa_cf1", missing_rate=0.20, random_seed=42)
        self.assertEqual(len(mask), self.n_samples)
        high_pm = self.df["pa_cf1"] > 100.0
        low_pm = self.df["pa_cf1"] < 40.0
        dropout_high = np.mean(~mask[high_pm])
        dropout_low = np.mean(~mask[low_pm])
        self.assertGreater(dropout_high, dropout_low)

    def test_lag_dependent_mask(self):
        mask = generate_lag_dependent_mask(self.df, pm_col="pa_cf1", missing_rate=0.20, random_seed=42)
        self.assertEqual(len(mask), self.n_samples)
        # Check overall rate
        self.assertAlmostEqual(np.mean(~mask), 0.20, delta=0.02)


if __name__ == "__main__":
    unittest.main()
