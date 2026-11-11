"""
Unit tests for calibration baseline models: Linear Regression, Random Forest, and XGBoost.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.models.calibration import build_model, train_and_evaluate_calibration_model


class TestCalibrationBaselines(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 200
        self.df = pd.DataFrame({
            "pa_cf1": np.random.uniform(10.0, 150.0, size=n),
            "pa_atm": np.random.uniform(8.0, 120.0, size=n),
            "pa_alt": np.random.uniform(5.0, 50.0, size=n),
            "relative_humidity": np.random.uniform(30.0, 80.0, size=n),
            "temperature": np.random.uniform(15.0, 35.0, size=n),
            "bam_reference": np.random.uniform(15.0, 130.0, size=n),
        })
        self.train_df = self.df.iloc[:120].copy()
        self.val_df = self.df.iloc[120:160].copy()
        self.test_df = self.df.iloc[160:].copy()

    def test_build_model_instances(self):
        lr = build_model("Linear Regression")
        rf = build_model("Random Forest")
        xgb = build_model("XGBoost")
        self.assertIsNotNone(lr)
        self.assertIsNotNone(rf)
        self.assertIsNotNone(xgb)

    def test_train_and_evaluate_linear_regression(self):
        res = train_and_evaluate_calibration_model(
            "Linear Regression", "all_channels", self.train_df, self.val_df, self.test_df
        )
        self.assertIn("train_metrics", res)
        self.assertIn("val_metrics", res)
        self.assertIn("test_metrics", res)
        self.assertTrue(res["test_metrics"]["rmse"] > 0)

    def test_train_and_evaluate_random_forest(self):
        res = train_and_evaluate_calibration_model(
            "Random Forest", "cf1_only", self.train_df, self.val_df, self.test_df
        )
        self.assertTrue(res["test_metrics"]["rmse"] > 0)

    def test_train_and_evaluate_xgboost(self):
        res = train_and_evaluate_calibration_model(
            "XGBoost", "atm_only", self.train_df, self.val_df, self.test_df
        )
        self.assertTrue(res["test_metrics"]["rmse"] > 0)


if __name__ == "__main__":
    unittest.main()
