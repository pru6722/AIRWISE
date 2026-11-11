"""
Unit tests for Pipeline A: SequentialImputeThenCalibrate in src.models.sequential_pipelines.

Tests:
1. Training-only fitting: Imputers, scalers, and calibration models fitted strictly on train_df.
2. BAM reference exclusion: BAM reference column is never used as an input feature for imputation.
3. Partial PM channel preservation: Preserves observed PM channels rather than treating partial dropouts as total outages.
4. Complete PM-channel dropout handling: Handles Scenario 3 gracefully.
5. Complete station outage handling: Handles Scenario 4 gracefully with fallback to training statistics.
6. Output shape and prediction availability: Guarantees 100% finite coverage and correct 1D array shapes.
7. Alternative imputer strategies: Verifies KNN and median imputer backends.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.sequential_pipelines import (
    SequentialImputeThenCalibrate,
    SequentialCalibrateThenImpute,
    CompleteCaseCalibrationReference,
    DEFAULT_PM_COLS,
    DEFAULT_METEO_COLS,
)


class TestSequentialImputeThenCalibrate(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 150
        # Synthetic physical sensor dynamics
        t = np.linspace(0, 24, n)
        pm_base = 50.0 + 30.0 * np.sin(2 * np.pi * t / 24.0) + np.random.normal(0, 2.0, size=n)
        rh = 60.0 + 20.0 * np.cos(2 * np.pi * t / 24.0) + np.random.normal(0, 1.0, size=n)
        temp = 25.0 - 5.0 * np.cos(2 * np.pi * t / 24.0) + np.random.normal(0, 0.5, size=n)

        # PurpleAir channels correlate with true PM and humidity
        cf1 = pm_base * 1.05 + 0.1 * rh + np.random.normal(0, 1.0, size=n)
        atm = pm_base * 0.98 + 0.08 * rh + np.random.normal(0, 1.0, size=n)
        alt = pm_base * 1.02 + 0.12 * rh + np.random.normal(0, 1.0, size=n)
        bam = pm_base * 1.00 + np.random.normal(0, 1.5, size=n)

        self.df = pd.DataFrame({
            "pa_cf1": cf1.astype(np.float32),
            "pa_atm": atm.astype(np.float32),
            "pa_alt": alt.astype(np.float32),
            "relative_humidity": rh.astype(np.float32),
            "temperature": temp.astype(np.float32),
            "hour_sin": np.sin(2 * np.pi * t / 24.0).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * t / 24.0).astype(np.float32),
            "bam_reference": bam.astype(np.float32),
        })

        # Strict chronological split: 100 train, 50 test
        self.train_df = self.df.iloc[:100].copy().reset_index(drop=True)
        self.test_df = self.df.iloc[100:].copy().reset_index(drop=True)

    def test_training_only_fitting(self):
        """Verify that imputer, scalers, and calibration models are fitted exclusively on train_df."""
        pipe = SequentialImputeThenCalibrate(
            imputer_type="knn",
            model_name="Linear Regression",
            scale_features=True,
            random_state=42,
        )
        pipe.fit(self.train_df)

        self.assertTrue(pipe.is_fitted_)
        self.assertIsNotNone(pipe.train_medians_)
        self.assertIsNotNone(pipe.scaler_)

        # Verify training medians match train_df exactly, not test_df
        for col in pipe.feature_cols_:
            expected_median = float(self.train_df[col].median())
            self.assertAlmostEqual(pipe.train_medians_[col], expected_median, places=4)

        # Inject extreme outlier strictly into test_df
        contaminated_test_df = self.test_df.copy()
        contaminated_test_df.loc[0, "temperature"] = 99999.0
        contaminated_test_df.loc[1, "pa_cf1"] = 88888.0

        # Predict on contaminated test data
        preds = pipe.predict(contaminated_test_df)
        self.assertEqual(len(preds), len(contaminated_test_df))

        # Check that pipeline internal training statistics are completely unchanged
        for col in pipe.feature_cols_:
            expected_median = float(self.train_df[col].median())
            self.assertAlmostEqual(pipe.train_medians_[col], expected_median, places=4)

    def test_bam_reference_exclusion(self):
        """Verify that BAM reference is never included in the imputer feature set."""
        pipe = SequentialImputeThenCalibrate(
            imputer_type="knn",
            model_name="Linear Regression",
            random_state=42,
        )
        pipe.fit(self.train_df)

        # BAM reference must NOT be in imputer feature columns
        self.assertNotIn("bam_reference", pipe.feature_cols_)
        self.assertNotIn("bam_reference", pipe.calib_feature_cols_)
        self.assertEqual(pipe.imputer_.n_features_in_, len(pipe.feature_cols_))

        # Modify test BAM to NaN and extreme values; predictions and imputation must be identical
        test_df_corrupted_bam = self.test_df.copy()
        test_df_corrupted_bam["bam_reference"] = np.nan
        preds_1 = pipe.predict(self.test_df)
        preds_2 = pipe.predict(test_df_corrupted_bam)
        np.testing.assert_allclose(preds_1, preds_2, err_msg="BAM target in test_df must not affect predictions.")

    def test_partial_pm_channel_preservation(self):
        """
        Verify that when 1 or 2 PM channels are missing (Scenario 2),
        observed PM channels are strictly preserved while missing ones are imputed.
        """
        pipe = SequentialImputeThenCalibrate(
            imputer_type="knn",
            model_name="Linear Regression",
            random_state=42,
        )
        pipe.fit(self.train_df)

        # Mask ONLY pa_cf1 in the test set, keeping pa_atm and pa_alt observed
        n_test = len(self.test_df)
        mask_df = pd.DataFrame(True, index=self.test_df.index, columns=pipe.feature_cols_)
        mask_df["pa_cf1"] = False  # Only pa_cf1 missing

        diag = pipe.predict_with_diagnostics(self.test_df, mask=mask_df)
        completed = diag["completed_features"]

        # 1. pa_atm and pa_alt must be exactly preserved
        np.testing.assert_allclose(
            completed["pa_atm"].to_numpy(),
            self.test_df["pa_atm"].to_numpy(),
            err_msg="Observed pa_atm must be strictly preserved during partial dropout.",
        )
        np.testing.assert_allclose(
            completed["pa_alt"].to_numpy(),
            self.test_df["pa_alt"].to_numpy(),
            err_msg="Observed pa_alt must be strictly preserved during partial dropout.",
        )

        # 2. pa_cf1 must be imputed (finite and distinct from NaN)
        self.assertFalse(completed["pa_cf1"].isna().any(), "Imputed pa_cf1 must not contain NaNs.")

        # 3. All rows should be classified as Scenario 2 (Partial PM missing)
        np.testing.assert_array_equal(
            diag["outage_scenarios"],
            np.full(n_test, 2),
            err_msg="All rows should be flagged as Scenario 2 (Partial PM missing).",
        )
        self.assertFalse(np.all(diag["m_pm"]))

    def test_complete_pm_dropout_handling(self):
        """Verify that Scenario 3 (all PM missing, meteo observed) is handled gracefully."""
        pipe = SequentialImputeThenCalibrate(
            imputer_type="knn",
            model_name="Linear Regression",
            random_state=42,
        )
        pipe.fit(self.train_df)

        n_test = len(self.test_df)
        mask_df = pd.DataFrame(True, index=self.test_df.index, columns=pipe.feature_cols_)
        for col in pipe.pm_cols:
            mask_df[col] = False  # All PM missing

        diag = pipe.predict_with_diagnostics(self.test_df, mask=mask_df)
        completed = diag["completed_features"]

        # Meteo preserved
        for col in pipe.meteo_cols:
            np.testing.assert_allclose(
                completed[col].to_numpy(),
                self.test_df[col].to_numpy(),
                err_msg=f"Meteorological feature {col} must be preserved in Scenario 3.",
            )

        # PM imputed and finite
        for col in pipe.pm_cols:
            self.assertFalse(completed[col].isna().any())

        # Scenario 3 flag
        np.testing.assert_array_equal(diag["outage_scenarios"], np.full(n_test, 3))
        self.assertEqual(len(diag["predictions"]), n_test)
        self.assertFalse(np.isnan(diag["predictions"]).any())

    def test_complete_station_outage_handling(self):
        """Verify that Scenario 4 (all PM and meteo missing) falls back gracefully to training statistics."""
        pipe = SequentialImputeThenCalibrate(
            imputer_type="knn",
            model_name="Linear Regression",
            random_state=42,
        )
        pipe.fit(self.train_df)

        n_test = len(self.test_df)
        mask_df = pd.DataFrame(False, index=self.test_df.index, columns=pipe.feature_cols_)

        diag = pipe.predict_with_diagnostics(self.test_df, mask=mask_df)
        completed = diag["completed_features"]

        # No NaNs in completed features
        self.assertFalse(completed.isna().any().any())

        # Scenario 4 flag
        np.testing.assert_array_equal(diag["outage_scenarios"], np.full(n_test, 4))
        self.assertEqual(diag["coverage"], 1.0)
        self.assertFalse(np.isnan(diag["predictions"]).any())

    def test_output_shapes_and_prediction_availability(self):
        """Verify prediction shapes, 100% coverage, and evaluation breakdown."""
        pipe = SequentialImputeThenCalibrate(
            imputer_type="knn",
            model_name="Random Forest",
            random_state=42,
        )
        pipe.fit(self.train_df)

        # Mixed missingness: 20 rows complete, 15 rows partial PM missing, 10 rows all PM missing, 5 rows blackout
        mask_df = pd.DataFrame(True, index=self.test_df.index, columns=pipe.feature_cols_)
        mask_df.iloc[20:35, 0] = False  # cf1 missing
        mask_df.iloc[35:45, :len(pipe.pm_cols)] = False  # all PM missing
        mask_df.iloc[45:, :] = False  # complete station blackout

        eval_res = pipe.evaluate(self.test_df, mask=mask_df)

        self.assertIn("overall_metrics", eval_res)
        self.assertIn("missing_pm_metrics", eval_res)
        self.assertIn("observed_pm_metrics", eval_res)
        self.assertIn("scenario_metrics", eval_res)

        # Check RMSE and MAE exist and are positive
        self.assertGreater(eval_res["overall_metrics"]["rmse"], 0.0)
        self.assertGreater(eval_res["overall_metrics"]["mae"], 0.0)

        # Check coverage
        diag = eval_res["diagnostics"]
        self.assertEqual(len(diag["predictions"]), len(self.test_df))
        self.assertFalse(np.isnan(diag["predictions"]).any())

    def test_alternative_imputers(self):
        """Verify Median and Iterative imputer backends."""
        for imputer_type in ["median", "iterative"]:
            pipe = SequentialImputeThenCalibrate(
                imputer_type=imputer_type,
                model_name="Linear Regression",
                random_state=42,
            )
            pipe.fit(self.train_df)
            preds = pipe.predict(self.test_df)
            self.assertEqual(len(preds), len(self.test_df))
            self.assertFalse(np.isnan(preds).any())


class TestPipelineB(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 150
        t = np.linspace(0, 24, n)
        pm_base = 50.0 + 30.0 * np.sin(2 * np.pi * t / 24.0) + np.random.normal(0, 2.0, size=n)
        rh = 60.0 + 20.0 * np.cos(2 * np.pi * t / 24.0) + np.random.normal(0, 1.0, size=n)
        temp = 25.0 - 5.0 * np.cos(2 * np.pi * t / 24.0) + np.random.normal(0, 0.5, size=n)

        cf1 = pm_base * 1.05 + 0.1 * rh + np.random.normal(0, 1.0, size=n)
        atm = pm_base * 0.98 + 0.08 * rh + np.random.normal(0, 1.0, size=n)
        alt = pm_base * 1.02 + 0.12 * rh + np.random.normal(0, 1.0, size=n)
        bam = pm_base * 1.00 + np.random.normal(0, 1.5, size=n)

        self.df = pd.DataFrame({
            "pa_cf1": cf1.astype(np.float32),
            "pa_atm": atm.astype(np.float32),
            "pa_alt": alt.astype(np.float32),
            "relative_humidity": rh.astype(np.float32),
            "temperature": temp.astype(np.float32),
            "hour_sin": np.sin(2 * np.pi * t / 24.0).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * t / 24.0).astype(np.float32),
            "day_of_year_sin": np.zeros(n, dtype=np.float32),
            "day_of_year_cos": np.ones(n, dtype=np.float32),
            "bam_reference": bam.astype(np.float32),
        })

        self.train_df = self.df.iloc[:100].copy().reset_index(drop=True)
        self.test_df = self.df.iloc[100:].copy().reset_index(drop=True)

    def test_pipeline_b_training_only_fitting_and_bam_isolation(self):
        """Verify calibration and reconstruction models in Pipeline B are fitted on train_df without BAM as input."""
        pipe_b = SequentialCalibrateThenImpute(
            model_name="Linear Regression",
            imputer_type="rf",
            random_state=42,
        )
        pipe_b.fit(self.train_df)

        self.assertTrue(pipe_b.is_fitted_)
        self.assertIsNotNone(pipe_b.train_cal_median_)
        # Verify BAM reference is NOT in reconstruction imputer features
        self.assertNotIn("bam_reference", pipe_b.covariate_cols_)

        # Verify test BAM has zero influence on predictions
        test_corrupt = self.test_df.copy()
        test_corrupt["bam_reference"] = 999999.0
        preds1 = pipe_b.predict(self.test_df)
        preds2 = pipe_b.predict(test_corrupt)
        np.testing.assert_allclose(preds1, preds2)

    def test_pipeline_b_outage_handling_and_100pct_coverage(self):
        """Verify Pipeline B handles partial PM missing, all PM missing, and blackout with 100% finite coverage."""
        pipe_b = SequentialCalibrateThenImpute(
            model_name="Linear Regression",
            imputer_type="rf",
            random_state=42,
        )
        pipe_b.fit(self.train_df)

        # Mixed mask: 20 rows observed, 15 rows partial PM, 10 rows all PM missing, 5 rows blackout
        mask_df = pd.DataFrame(True, index=self.test_df.index, columns=pipe_b.feature_cols_)
        mask_df.iloc[20:35, 0] = False  # pa_cf1 missing
        mask_df.iloc[35:45, :3] = False  # all PM missing
        mask_df.iloc[45:, :] = False    # complete station blackout

        diag = pipe_b.predict_with_diagnostics(self.test_df, mask=mask_df)
        preds = diag["predictions"]

        self.assertEqual(len(preds), len(self.test_df))
        self.assertFalse(np.isnan(preds).any(), "Pipeline B must have 100% finite coverage without NaNs.")
        self.assertEqual(diag["coverage"], 1.0)

        # Scenarios checked
        scenarios = diag["outage_scenarios"]
        self.assertEqual(scenarios[0], 1)   # All PM observed
        self.assertEqual(scenarios[25], 2)  # Partial PM missing
        self.assertEqual(scenarios[40], 3)  # All PM missing, meteo on
        self.assertEqual(scenarios[48], 4)  # Complete blackout


class TestPipelineD(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 150
        t = np.linspace(0, 24, n)
        pm_base = 50.0 + 30.0 * np.sin(2 * np.pi * t / 24.0) + np.random.normal(0, 2.0, size=n)
        rh = 60.0 + 20.0 * np.cos(2 * np.pi * t / 24.0) + np.random.normal(0, 1.0, size=n)
        temp = 25.0 - 5.0 * np.cos(2 * np.pi * t / 24.0) + np.random.normal(0, 0.5, size=n)

        cf1 = pm_base * 1.05 + 0.1 * rh
        atm = pm_base * 0.98 + 0.08 * rh
        alt = pm_base * 1.02 + 0.12 * rh
        bam = pm_base * 1.00

        self.df = pd.DataFrame({
            "pa_cf1": cf1.astype(np.float32),
            "pa_atm": atm.astype(np.float32),
            "pa_alt": alt.astype(np.float32),
            "relative_humidity": rh.astype(np.float32),
            "temperature": temp.astype(np.float32),
            "bam_reference": bam.astype(np.float32),
        })

        self.train_df = self.df.iloc[:100].copy().reset_index(drop=True)
        self.test_df = self.df.iloc[100:].copy().reset_index(drop=True)

    def test_pipeline_d_no_silent_imputation_and_partial_coverage(self):
        """Verify Pipeline D never silently imputes missing points and returns NaNs with exact coverage."""
        pipe_d = CompleteCaseCalibrationReference(
            model_name="Linear Regression",
            random_state=42,
        )
        pipe_d.fit(self.train_df)

        self.assertTrue(pipe_d.is_fitted_)
        self.assertEqual(pipe_d.n_train_complete_, len(self.train_df))

        # 20% missingness mask (10 missing out of 50)
        mask = np.ones(len(self.test_df), dtype=bool)
        mask[10:20] = False

        diag = pipe_d.predict_with_diagnostics(self.test_df, mask=mask)
        preds = diag["predictions"]

        # Missing points must be NaN (NO silent imputation)
        self.assertTrue(np.isnan(preds[~mask]).all(), "Missing points must be NaN in Pipeline D.")
        self.assertFalse(np.isnan(preds[mask]).any(), "Observed points must be finite.")
        self.assertEqual(diag["coverage"], 0.80)
        self.assertEqual(diag["n_eligible"], 40)
        self.assertEqual(diag["n_dropped"], 10)

        # Evaluation restricted to eligible observed subset
        eval_res = pipe_d.evaluate(self.test_df, mask=mask)
        self.assertIsNotNone(eval_res["observed_pm_metrics"])
        self.assertIsNone(eval_res["missing_pm_metrics"])
        self.assertEqual(eval_res["missing_pm_availability"], 0.0)
        self.assertIn("Restricted to matched observed subset", eval_res["ranking_eligibility"])


if __name__ == "__main__":
    unittest.main()

