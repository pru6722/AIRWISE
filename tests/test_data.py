"""
Unit and integration tests for data ingestion, validation, and preprocessing modules.
Uses standard library unittest to run seamlessly in any Python environment.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.data.ingestion import load_raw_workbook
from src.data.validation import audit_timestamps, audit_dew_point_physics, audit_numerical_ranges
from src.data.preprocessing import preprocess_and_flag_dataset, load_cleaned_dataset


class TestDataEngineeringPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.excel_path = cls.project_root / "data" / "raw" / "Campmier_Manuscript_Data.xlsx"
        if not cls.excel_path.exists():
            raise FileNotFoundError(f"Dataset file missing at {cls.excel_path}")
        cls.combined_df, cls.site_dfs = load_raw_workbook(cls.excel_path)

    def test_dataset_row_counts_and_sites(self):
        self.assertEqual(len(self.combined_df), 18584)
        self.assertEqual(set(self.site_dfs.keys()), {"Delhi", "Hamirpur", "Bangalore"})
        self.assertEqual(len(self.site_dfs["Delhi"]), 7504)
        self.assertEqual(len(self.site_dfs["Hamirpur"]), 3639)
        self.assertEqual(len(self.site_dfs["Bangalore"]), 7441)

    def test_zero_nulls_and_duplicates(self):
        self.assertEqual(self.combined_df.isna().sum().sum(), 0)
        for site, df in self.site_dfs.items():
            self.assertEqual(df["timestamp"].duplicated().sum(), 0)

    def test_non_negative_pm25_measurements(self):
        pm25_cols = ["bam_reference", "pa_cf1", "pa_atm", "pa_alt"]
        for col in pm25_cols:
            min_val = self.combined_df[col].min()
            self.assertGreaterEqual(min_val, 0.0)

    def test_dew_point_physical_inconsistency(self):
        for site, df in self.site_dfs.items():
            dp_audit = audit_dew_point_physics(df)
            self.assertEqual(dp_audit["inconsistent_td_greater_than_t_pct"], 100.0)
            self.assertEqual(dp_audit["converted_f_to_c_inconsistent_count"], 0)

    def test_preprocessing_and_segmentation(self):
        clean_df = preprocess_and_flag_dataset(self.combined_df)
        self.assertEqual(len(clean_df), len(self.combined_df))
        self.assertIn("is_dew_point_suspicious", clean_df.columns)
        self.assertIn("segment_id", clean_df.columns)
        self.assertIn("time_diff_hours", clean_df.columns)
        self.assertEqual(clean_df["is_dew_point_suspicious"].sum(), len(clean_df))
        self.assertGreater(clean_df["segment_id"].nunique(), 100)

    def test_cleaned_parquet_persisted(self):
        parquet_path = self.project_root / "data" / "processed" / "cleaned_data.parquet"
        self.assertTrue(parquet_path.exists())
        df = load_cleaned_dataset(parquet_path)
        self.assertEqual(len(df), 18584)
        self.assertEqual(df["bam_reference"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
