"""
AIRWISE Dashboard Step 3 (Priority Part 2 + Input-Isolation Audit) Comprehensive Test Suite
Covers:
  1. Home (dashboard/app.py) & all 7 Streamlit page route resolutions / switch_page transitions.
  2. Interactive UI & filter state tests on Pages 01, 02, 05, 06, and 07.
  3. Accurate artifact-grounded figures across Uncertainty, CPCB, and Big Data pages.
  4. DuckDB SQL read-only guardrails, catalog completeness, and CSV download controls.
  5. Joint TCN Synthetic Outage Simulator Input-Isolation Audit:
     - Verifies `bam_reference` cannot influence inference (`y_cal`, `y_rec`, `y_final`).
     - Verifies `mask == 0` zeroes all masked PM channels so input PM perturbations inside
       the simulated outage window have bit-for-bit zero effect on predicted outputs.
  6. Zero SyntaxWarning & zero use_container_width deprecation warnings.
  7. Frozen model checkpoint & Phase 3C parquet SHA-256 preservation.
"""

import hashlib
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.uncertainty.split_manager import create_conformal_splits
from dashboard.services.data_service import (
    load_cleaned_dataset,
    load_metrics_table,
    load_caching_benchmark,
)
from dashboard.services.sql_service import (
    execute_safe_query,
    get_warehouse_catalog,
)
from dashboard.services.tcn_service import run_sequence_inference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(PROJECT_ROOT / "dashboard" / "app.py")
PAGE_01 = str(PROJECT_ROOT / "dashboard" / "pages" / "01_Research_Overview.py")
PAGE_02 = str(PROJECT_ROOT / "dashboard" / "pages" / "02_Dataset_Explorer.py")
PAGE_05 = str(PROJECT_ROOT / "dashboard" / "pages" / "05_Uncertainty_Analysis.py")
PAGE_06 = str(PROJECT_ROOT / "dashboard" / "pages" / "06_CPCB_Reliability.py")
PAGE_07 = str(PROJECT_ROOT / "dashboard" / "pages" / "07_Big_Data_Analytics.py")

ALL_ROUTES = [
    "pages/01_Research_Overview.py",
    "pages/02_Dataset_Explorer.py",
    "pages/03_Calibration_Reconstruction.py",
    "pages/04_Joint_TCN.py",
    "pages/05_Uncertainty_Analysis.py",
    "pages/06_CPCB_Reliability.py",
    "pages/07_Big_Data_Analytics.py",
]

FROZEN_SHA256 = {
    "outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt": (
        "4a0c5704257fd556e53cfc23b1c26439881009f8aabb423b3f6f2752e7c5217c"
    ),
    "outputs/models/conformal/frozen_joint_tcn_checkpoint.pt": (
        "3eae6ea2b0f6c7abdbca72df672c61f068d78b4e49c9006e0eb54ac8a2fa7350"
    ),
    "outputs/experiments/phase3C/benchmark_predictions.parquet": (
        "87e5263742403644d96605f9bc742de6e5009a29d8eb5509c149b8b154a4f662"
    ),
}


def _collect_text(at: AppTest) -> str:
    """Flattens visible Markdown/Info/Warning/Error/Caption text from an AppTest run."""
    chunks = []
    for attr in ("markdown", "info", "warning", "error", "caption", "success", "title", "subheader"):
        for el in getattr(at, attr, []):
            val = getattr(el, "value", None) or getattr(el, "body", None)
            if val is not None:
                chunks.append(str(val))
    return "\n".join(chunks)


def _assert_no_bad_warnings(caught):
    """Asserts no SyntaxWarning or Streamlit use_container_width deprecation warnings occurred."""
    bad = [
        w for w in caught
        if issubclass(w.category, SyntaxWarning)
        or "use_container_width" in str(w.message)
    ]
    if bad:
        formatted = "; ".join(f"{w.category.__name__}: {w.message}" for w in bad)
        raise AssertionError(f"Unexpected warnings emitted: {formatted}")


class TestDashboardAllPagesAndAudit(unittest.TestCase):
    """Verification of Home + Pages 01, 02, 05, 06, 07, routing, SQL, and TCN isolation."""

    def test_app_home_and_all_page_link_routes_resolve(self):
        """Home page renders cleanly and every navigation route target exists and switches cleanly."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(APP_PATH, default_timeout=45)
            at.run()
        self.assertFalse(at.exception, msg=f"app.py failed: {at.exception}")
        _assert_no_bad_warnings(caught)

        home_text = _collect_text(at)
        self.assertIn("No single reconstruction strategy is uniformly optimal", home_text)
        self.assertIn("11,149", home_text)
        self.assertIn("3,720", home_text)
        self.assertIn("63.83%", home_text)
        self.assertIn("74.58%", home_text)

        # Verify st.page_link targets exist cleanly on disk and switch_page succeeds for every page
        page_links = at.get("page_link")
        resolved_targets = [str(el.page) for el in page_links]
        for route in ALL_ROUTES:
            full_target = PROJECT_ROOT / "dashboard" / route
            self.assertTrue(full_target.exists(), msg=f"Missing route file: {full_target}")
            expected_stem_tail = Path(route).stem.split("_", 1)[1]
            self.assertTrue(
                any(expected_stem_tail in t or route in t for t in resolved_targets),
                msg=f"Route {route} not exposed on dashboard/app.py page_links: {resolved_targets}",
            )

        for route in ALL_ROUTES:
            with warnings.catch_warnings(record=True) as caught_sw:
                warnings.simplefilter("always")
                nav = AppTest.from_file(APP_PATH, default_timeout=60)
                nav.run()
                nav.switch_page(route).run()
            self.assertFalse(nav.exception, msg=f"Route switch to {route} failed: {nav.exception}")
            _assert_no_bad_warnings(caught_sw)

    def test_page_01_research_overview_standalone_and_splits(self):
        """Page 01 renders standalone without page_link errors and displays authoritative splits."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(PAGE_01, default_timeout=45)
            at.run()
        self.assertFalse(at.exception, msg=f"Page 01 failed: {at.exception}")
        _assert_no_bad_warnings(caught)

        txt = _collect_text(at)
        self.assertIn("11,149", txt)
        self.assertIn("1,857", txt)
        self.assertIn("1,858", txt)
        self.assertIn("3,720", txt)
        self.assertIn("63.83%", txt)
        self.assertIn("74.58%", txt)

    def test_page_02_dataset_explorer_filters_and_downloads(self):
        """Page 02 filters by site/scenario cleanly and exposes curated + raw workbook downloads."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(PAGE_02, default_timeout=45)
            at.run()
            self.assertFalse(at.exception, msg=f"Page 02 init failed: {at.exception}")

            # Switch site filter & scenario mask selector
            at.selectbox("_sb_site").set_value("Delhi")
            mnar_options = [o for o in at.selectbox("_sb_scenario").options if "MNAR" in o]
            self.assertTrue(mnar_options)
            at.selectbox("_sb_scenario").set_value(mnar_options[0])
            at.run()
        self.assertFalse(at.exception, msg=f"Page 02 filter update failed: {at.exception}")
        _assert_no_bad_warnings(caught)

        dls = at.get("download_button")
        self.assertGreaterEqual(len(dls), 2)

    def test_page_05_uncertainty_analysis_modes_and_frozen_metrics(self):
        """Page 05 exposes 90%/95%/Block modes, simultaneous coverage disclosure, and CSV export."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(PAGE_05, default_timeout=45)
            at.run()
            self.assertFalse(at.exception, msg=f"Page 05 init failed: {at.exception}")

            txt = _collect_text(at)
            self.assertIn("77.02%", txt)
            self.assertIn("86.40%", txt)
            self.assertIn("95.70%", txt)
            self.assertIn("63.83%", txt)
            self.assertIn("74.58%", txt)
            self.assertIn("120/188", txt)
            self.assertIn("44/59", txt)
            self.assertIn("1.95%", txt)
            self.assertIn("N_cal = 7", txt)
            self.assertNotIn("> N/A", txt)

            # Toggle 95% mode + Temporal Block 12h + Delhi filter
            at.radio("unc_cov_mode").set_value("95% Interval")
            at.selectbox("unc_method_mode").set_value("Temporal Block (12h: ±44.34 / ±100.42)")
            at.selectbox("_sb_site").set_value("Delhi")
            at.run()
        self.assertFalse(at.exception, msg=f"Page 05 interaction failed: {at.exception}")
        _assert_no_bad_warnings(caught)

        dls = at.get("download_button")
        self.assertGreaterEqual(len(dls), 1)

    def test_page_06_cpcb_reliability_models_disclosures_and_export(self):
        """Page 06 toggles CM models, discloses PM2.5-only scope & 0/4 Severe recall, exports 24h CSV."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(PAGE_06, default_timeout=45)
            at.run()
            self.assertFalse(at.exception, msg=f"Page 06 init failed: {at.exception}")

            txt = _collect_text(at)
            self.assertIn("PM2.5-Only Evaluation (Not Full Multi-Pollutant NAQI)", txt)
            self.assertIn("0 / 4", txt)
            self.assertIn(">250 µg/m³", txt)
            self.assertIn("18/24", txt)
            self.assertIn("25.39 / 0.8798", txt)
            self.assertIn("MAE: 16.48", txt)
            self.assertIn("72.44%", txt)

            # Verify canonical and alias loader keys both return populated CPCB 24h tables
            self.assertEqual(len(load_metrics_table("checkpoint5_point_metrics")), 10)
            self.assertEqual(len(load_metrics_table("checkpoint5_24h_point_metrics")), 10)
            self.assertEqual(len(load_metrics_table("checkpoint5_coverage_metrics")), 10)
            self.assertEqual(len(load_metrics_table("checkpoint5_24h_coverage_metrics")), 10)

            at.selectbox("cpcb_cm_model_sel").set_value("Raw PurpleAir (ATM Factory Calibration)")
            at.radio("cpcb_cm_norm").set_value("Row-Normalized Recall (%)")
            at.run()
        self.assertFalse(at.exception, msg=f"Page 06 CM model toggle failed: {at.exception}")
        _assert_no_bad_warnings(caught)

        dls = at.get("download_button")
        self.assertGreaterEqual(len(dls), 1)

    def test_page_07_big_data_sql_security_catalog_and_accurate_benchmarks(self):
        """Page 07 shows true caching/scaling benchmarks, full catalog, SQL console, and CSV download."""
        # Verify catalog and SQL security engine directly
        catalog = get_warehouse_catalog()
        self.assertIn("sensor_format_parquet", catalog["table_name"].tolist())
        self.assertIn("sensor_part_site_year_month", catalog["table_name"].tolist())

        df_q1, _, err_q1 = execute_safe_query("SELECT site, COUNT(*) AS n FROM sensor_format_parquet GROUP BY site;")
        self.assertIsNone(err_q1)
        self.assertEqual(len(df_q1), 3)
        self.assertEqual(int(df_q1["n"].sum()), 18584)

        for blocked_sql in [
            "DROP TABLE sensor_format_parquet;",
            "SELECT * FROM read_parquet('/tmp/test.parquet');",
            "SELECT COUNT(*) FROM sensor_format_parquet; DELETE FROM sensor_format_parquet;",
        ]:
            _, _, err_blk = execute_safe_query(blocked_sql)
            self.assertIsNotNone(err_blk)
            self.assertIn("Rejected", err_blk)

        # Verify Page 07 UI uses true JSON/CSV metrics without stale hardcoded values
        cache_json = load_caching_benchmark()
        self.assertAlmostEqual(float(cache_json["cache_speedup"]), 1.2, places=2)
        self.assertAlmostEqual(float(cache_json["median_uncached_sec"]), 0.134, places=3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(PAGE_07, default_timeout=45)
            at.run()
        self.assertFalse(at.exception, msg=f"Page 07 failed: {at.exception}")
        _assert_no_bad_warnings(caught)

        txt = _collect_text(at)
        self.assertIn("local[*]", txt)
        self.assertIn("0.134s", txt)
        self.assertIn("0.111s", txt)
        self.assertIn("1.20×", txt)
        self.assertIn("ORC -84.1% | Parquet -73.0%", txt)
        self.assertIn("CSV 2.126 MB -> Parquet 0.573 MB -> ORC 0.338 MB", txt)
        self.assertNotIn("-0.0%", txt)
        self.assertNotIn("2.42x", txt)
        self.assertNotIn("983,398", txt)

        # Verify canonical and alias loader keys return storage format benchmarks
        self.assertEqual(len(load_metrics_table("checkpoint6_storage_formats")), 3)
        self.assertEqual(len(load_metrics_table("checkpoint6_storage")), 3)

        dls = at.get("download_button")
        self.assertGreaterEqual(len(dls), 1)

    def test_joint_tcn_simulator_input_isolation_audit(self):
        """
        Audit Requirement:
        1. Verify `bam_reference` is never leaked to Joint TCN `v2` inference (`run_sequence_inference`).
        2. Verify synthetic mask `m == 0` zeroes masked PM channels so perturbing PM inputs inside
           the outage window has zero effect on predicted outputs.
        """
        df_clean = load_cleaned_dataset()
        _, _, _, test_df, _ = create_conformal_splits(df_clean)
        df_win = test_df[test_df["site"] == "Delhi"].iloc[:168].copy().reset_index(drop=True)

        mask = np.ones(len(df_win), dtype=np.float32)
        mask[40:52] = 0.0  # 12-hour simulated PM outage

        # 1. BAM Reference Leakage Test
        y_cal_act, y_rec_act, y_fin_act = run_sequence_inference(df_win, mask, model_variant="v2")

        df_bam_zero = df_win.copy()
        df_bam_zero["bam_reference"] = 0.0
        y_cal_z, y_rec_z, y_fin_z = run_sequence_inference(df_bam_zero, mask, model_variant="v2")

        df_bam_extreme = df_win.copy()
        df_bam_extreme["bam_reference"] = 99999.0
        y_cal_ex, y_rec_ex, y_fin_ex = run_sequence_inference(df_bam_extreme, mask, model_variant="v2")

        df_bam_dropped = df_win.drop(columns=["bam_reference"])
        y_cal_dr, y_rec_dr, y_fin_dr = run_sequence_inference(df_bam_dropped, mask, model_variant="v2")

        for act, z, ex, dr in [
            (y_cal_act, y_cal_z, y_cal_ex, y_cal_dr),
            (y_rec_act, y_rec_z, y_rec_ex, y_rec_dr),
            (y_fin_act, y_fin_z, y_fin_ex, y_fin_dr),
        ]:
            np.testing.assert_allclose(act, z, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(act, ex, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(act, dr, rtol=0.0, atol=0.0)

        # 2. Masked PM Channel Zeroing Test
        df_pm_corrupted = df_win.copy()
        df_pm_corrupted.loc[40:51, ["pa_cf1", "pa_atm", "pa_alt"]] = 999.0
        _, y_rec_cor, y_fin_cor = run_sequence_inference(df_pm_corrupted, mask, model_variant="v2")

        # Inside outage (`mask == 0`), masked PM perturbations must cause zero change to y_rec/y_final
        np.testing.assert_allclose(y_rec_act, y_rec_cor, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(y_fin_act, y_fin_cor, rtol=0.0, atol=0.0)

    def test_frozen_artifact_sha256_integrity(self):
        """Ensure frozen checkpoints and Phase 3C parquet hashes remain byte-identical."""
        for rel_path, expected_sha in FROZEN_SHA256.items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), msg=f"Missing frozen file: {full_path}")
            digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_sha, msg=f"SHA-256 mismatch on {rel_path}")


if __name__ == "__main__":
    unittest.main()
