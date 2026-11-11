"""
AIRWISE Dashboard Data Service & Foundation Tests
Verifies data loading, schema validation, configuration consistency,
and slice filtering for the upgraded research dashboard.
"""

import unittest
import numpy as np
import pandas as pd

from dashboard.utils.paths import (
    CLEANED_PARQUET_PATH,
    BENCHMARK_PREDICTIONS_PARQUET,
    SCENARIO_METRICS_OVERALL_CSV,
    SCENARIO_METRICS_MISSING_PM_CSV,
    SCENARIO_METRICS_COMMON_SUPPORT_CSV,
    BOOTSTRAP_SIGNIFICANCE_CSV,
    BOOTSTRAP_SENSITIVITY_CSV,
    SITE_STRATIFIED_METRICS_CSV,
    JOINT_TCN_V2_CHECKPOINT_PATH,
    JOINT_TCN_V2_METADATA_PATH,
)
from dashboard.config import (
    DATASET_SPLITS,
    PIPELINES,
    SCENARIOS,
    EVALUATION_POPULATIONS,
    SITES,
)
from dashboard.services.data_service import (
    load_cleaned_dataset,
    load_benchmark_predictions,
    load_phase3c_metrics,
    load_joint_tcn_v2_metadata,
    filter_benchmark_predictions,
    compute_evaluation_metrics,
)


class TestDashboardDataService(unittest.TestCase):
    """Test suite for dashboard data service and Phase 3C integrations."""

    def test_phase3c_paths_exist(self):
        """Verify that all Phase 3C file artifacts exist on disk."""
        self.assertTrue(BENCHMARK_PREDICTIONS_PARQUET.exists(), f"Missing {BENCHMARK_PREDICTIONS_PARQUET}")
        self.assertTrue(SCENARIO_METRICS_OVERALL_CSV.exists(), f"Missing {SCENARIO_METRICS_OVERALL_CSV}")
        self.assertTrue(SCENARIO_METRICS_MISSING_PM_CSV.exists(), f"Missing {SCENARIO_METRICS_MISSING_PM_CSV}")
        self.assertTrue(SCENARIO_METRICS_COMMON_SUPPORT_CSV.exists(), f"Missing {SCENARIO_METRICS_COMMON_SUPPORT_CSV}")
        self.assertTrue(BOOTSTRAP_SIGNIFICANCE_CSV.exists(), f"Missing {BOOTSTRAP_SIGNIFICANCE_CSV}")
        self.assertTrue(BOOTSTRAP_SENSITIVITY_CSV.exists(), f"Missing {BOOTSTRAP_SENSITIVITY_CSV}")
        self.assertTrue(SITE_STRATIFIED_METRICS_CSV.exists(), f"Missing {SITE_STRATIFIED_METRICS_CSV}")
        self.assertTrue(JOINT_TCN_V2_CHECKPOINT_PATH.exists(), f"Missing {JOINT_TCN_V2_CHECKPOINT_PATH}")
        self.assertTrue(JOINT_TCN_V2_METADATA_PATH.exists(), f"Missing {JOINT_TCN_V2_METADATA_PATH}")

    def test_config_consistency(self):
        """Verify configuration constants match Phase 3C specifications."""
        # Splits
        self.assertEqual(DATASET_SPLITS["train"], 11149)
        self.assertEqual(DATASET_SPLITS["validation"], 1857)
        self.assertEqual(DATASET_SPLITS["conformal"], 1858)
        self.assertEqual(DATASET_SPLITS["test"], 3720)
        self.assertEqual(DATASET_SPLITS["total"], 18584)

        # Pipelines
        self.assertEqual(set(PIPELINES.keys()), {"A", "B", "C", "D"})

        # Scenarios
        expected_scenarios = {
            "MCAR_20pct", "Block_1h", "Block_3h", "Block_6h",
            "Block_12h", "Block_24h", "MAR_RH", "MAR_Lag", "MNAR_PM"
        }
        self.assertEqual(set(SCENARIOS.keys()), expected_scenarios)

        # Evaluation populations
        self.assertEqual(set(EVALUATION_POPULATIONS.keys()), {"overall", "missing_pm", "common_support"})

    def test_load_cleaned_dataset(self):
        """Verify clean dataset loading."""
        df = load_cleaned_dataset()
        self.assertEqual(len(df), 18584)
        self.assertTrue("bam_reference" in df.columns)
        self.assertTrue("site" in df.columns)
        self.assertTrue("timestamp" in df.columns)

    def test_load_benchmark_predictions(self):
        """Verify loading of benchmark predictions parquet file."""
        df = load_benchmark_predictions()
        self.assertEqual(len(df), 3720)
        self.assertEqual(df.shape[1], 49)

        # Verify key columns
        self.assertTrue("timestamp" in df.columns)
        self.assertTrue("site" in df.columns)
        self.assertTrue("bam_reference" in df.columns)

        # Verify scenario predictions and masks
        for sc in SCENARIOS.keys():
            self.assertTrue(f"mask_{sc}" in df.columns, f"Missing mask_{sc}")
            for p in ["A", "B", "C", "D"]:
                self.assertTrue(f"pred_{p}_{sc}" in df.columns, f"Missing pred_{p}_{sc}")

    def test_load_phase3c_metrics(self):
        """Verify all Phase 3C metric tables load correctly."""
        keys = ["overall", "missing_pm", "common_support", "bootstrap_significance", "bootstrap_sensitivity", "site_stratified"]
        for k in keys:
            df = load_phase3c_metrics(k)
            self.assertFalse(df.empty, f"Metric table {k} is empty")
            self.assertTrue(len(df) > 0)

        # Check overall table structure
        df_overall = load_phase3c_metrics("overall")
        self.assertEqual(len(df_overall), 36)  # 9 scenarios x 4 pipelines
        self.assertTrue("rmse" in df_overall.columns)
        self.assertTrue("pipeline_code" in df_overall.columns)

    def test_load_joint_tcn_v2_metadata(self):
        """Verify Joint TCN v2 metadata loading."""
        meta = load_joint_tcn_v2_metadata()
        self.assertTrue("causality_verification" in meta)
        self.assertTrue(meta["causality_verification"]["passed"])
        self.assertEqual(meta["causality_verification"]["max_deviation"], 0.0)
        self.assertEqual(meta["causality_verification"]["total_checks"], 144)
        self.assertTrue("history" in meta)
        self.assertEqual(len(meta["history"]), 16)
        self.assertEqual(meta["training_summary"]["best_epoch"], 6)

    def test_filter_benchmark_predictions_population_modes(self):
        """Verify population mode filtering disaggregates populations accurately."""
        df = load_benchmark_predictions()

        # Overall
        df_overall = filter_benchmark_predictions(df, scenario_id="MCAR_20pct", population_mode="overall")
        self.assertEqual(len(df_overall), 3720)

        # Missing-PM
        df_missing = filter_benchmark_predictions(df, scenario_id="MCAR_20pct", population_mode="missing_pm")
        self.assertEqual(len(df_missing), 769)  # Exactly 769 hidden points in 20% MCAR
        self.assertTrue((~df_missing["mask_MCAR_20pct"]).all())

        # Common support
        df_common = filter_benchmark_predictions(df, scenario_id="MCAR_20pct", population_mode="common_support")
        self.assertEqual(len(df_common), 2951)  # 3720 - 769 = 2951
        self.assertTrue(df_common["mask_MCAR_20pct"].all())

    def test_filter_benchmark_predictions_site(self):
        """Verify site filtering."""
        df = load_benchmark_predictions()
        df_blr = filter_benchmark_predictions(df, scenario_id="MCAR_20pct", site="Bangalore")
        self.assertEqual(len(df_blr), 1489)
        self.assertTrue((df_blr["site"] == "Bangalore").all())

        df_del = filter_benchmark_predictions(df, scenario_id="MCAR_20pct", site="Delhi")
        self.assertEqual(len(df_del), 1502)
        self.assertTrue((df_del["site"] == "Delhi").all())

        df_ham = filter_benchmark_predictions(df, scenario_id="MCAR_20pct", site="Hamirpur")
        self.assertEqual(len(df_ham), 729)
        self.assertTrue((df_ham["site"] == "Hamirpur").all())

    def test_compute_evaluation_metrics(self):
        """Verify metric calculation helper."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 19.0, 31.0, 42.0])
        metrics = compute_evaluation_metrics(y_true, y_pred)

        self.assertEqual(metrics["n_samples"], 4)
        # diff = [2, -1, 1, 2], mean_diff = 1.0 (mbe)
        self.assertAlmostEqual(metrics["mbe"], 1.0)
        # mae = (2 + 1 + 1 + 2)/4 = 1.5
        self.assertAlmostEqual(metrics["mae"], 1.5)
        # mse = (4 + 1 + 1 + 4)/4 = 2.5, rmse = sqrt(2.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5))
        # mean_yt = 25.0, rel_bias_pct = (1.0/25.0)*100 = 4.0%
        self.assertAlmostEqual(metrics["rel_bias_pct"], 4.0)

        # Test NaN handling
        y_true_nan = np.array([10.0, np.nan, 30.0])
        y_pred_nan = np.array([12.0, 20.0, 31.0])
        metrics_nan = compute_evaluation_metrics(y_true_nan, y_pred_nan)
        self.assertEqual(metrics_nan["n_samples"], 2)


if __name__ == "__main__":
    unittest.main()
