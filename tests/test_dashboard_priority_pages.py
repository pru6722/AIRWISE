"""
AIRWISE Dashboard Step 3 (Priority Part 1) Integration & UI Tests
Verifies upgraded pages:
- dashboard/pages/03_Calibration_Reconstruction.py
- dashboard/pages/04_Joint_TCN.py
- Date-boundary inclusive filtering fix in data_service.py (3,720 full test rows)
- Service default checkpoint in tcn_service.py ("v2" causal epoch-6)
- Frozen artifact SHA-256 immutability
"""

import hashlib
import inspect
import unittest
import warnings
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.components.sidebar import DEFAULT_MIN_DATE, DEFAULT_MAX_DATE
from dashboard.services.data_service import (
    load_benchmark_predictions,
    filter_benchmark_predictions,
    load_metrics_table,
)
from dashboard.services.tcn_service import run_sequence_inference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_03 = str(PROJECT_ROOT / "dashboard" / "pages" / "03_Calibration_Reconstruction.py")
PAGE_04 = str(PROJECT_ROOT / "dashboard" / "pages" / "04_Joint_TCN.py")

FROZEN_SHA256 = {
    "outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt": "4a0c5704257fd556e53cfc23b1c26439881009f8aabb423b3f6f2752e7c5217c",
    "outputs/models/conformal/frozen_joint_tcn_checkpoint.pt": "3eae6ea2b0f6c7abdbca72df672c61f068d78b4e49c9006e0eb54ac8a2fa7350",
    "outputs/experiments/phase3C/benchmark_predictions.parquet": "87e5263742403644d96605f9bc742de6e5009a29d8eb5509c149b8b154a4f662",
}


class TestDashboardPriorityPages(unittest.TestCase):
    """Streamlit AppTest verification for Priority Part 1 pages (03 & 04)."""

    def test_inclusive_end_date_captures_all_3720_benchmark_rows(self):
        """Ensure default calendar range includes final-day Hamirpur timestamps up to 22:30:00."""
        df_bench = load_benchmark_predictions()
        self.assertEqual(len(df_bench), 3720)

        filtered = filter_benchmark_predictions(
            df_bench,
            scenario_id="MCAR_20pct",
            site="All Sites",
            population_mode="overall",
            start_date=DEFAULT_MIN_DATE,
            end_date=DEFAULT_MAX_DATE,
        )
        self.assertEqual(len(filtered), 3720)

        filtered_single_end = filter_benchmark_predictions(
            df_bench,
            scenario_id="MCAR_20pct",
            site="Hamirpur",
            population_mode="overall",
            start_date=DEFAULT_MAX_DATE,
            end_date=DEFAULT_MAX_DATE,
        )
        self.assertGreater(len(filtered_single_end), 0)

    def test_tcn_service_defaults_to_v2_and_ablation_table_loads(self):
        """Verify run_sequence_inference defaults to v2 causal checkpoint & ablation loads."""
        sig = inspect.signature(run_sequence_inference)
        self.assertEqual(sig.parameters["model_variant"].default, "v2")

        df_ablation = load_metrics_table("joint_tcn_ablation")
        self.assertEqual(len(df_ablation), 6)
        self.assertIn("ablation_configuration", df_ablation.columns)

    def test_page_03_calibration_reconstruction_default_and_filtered(self):
        """Run Page 03 with AppTest across default and filtered states."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(PAGE_03, default_timeout=45)
            at.run()

            self.assertEqual(len(at.exception), 0, f"Page 03 exceptions: {at.exception}")
            # Verify CSV download buttons exist on Page 03
            self.assertGreaterEqual(len(at.button), 1)
            # Check dataframe outputs exist (overview table, site table, bootstrap table, etc.)
            self.assertGreaterEqual(len(at.dataframe), 2)

            # Verify bootstrap table labels use Tail Proportion and never raw p_value / p-value headers
            for df_el in at.dataframe:
                cols_lower = [str(c).lower() for c in df_el.value.columns]
                self.assertNotIn("p_value", cols_lower)
                self.assertNotIn("p-value", cols_lower)

            has_tail_header = any(
                "Tail Proportion" in str(col)
                for df_el in at.dataframe
                for col in df_el.value.columns
            )
            self.assertTrue(has_tail_header, "Expected 'Tail Proportion' header in bootstrap dataframe.")

            # Switch sidebar filters: Delhi, Block_24h, Pipeline C, Missing-PM Subgroup
            at.sidebar.selectbox(key="_sb_site").set_value("Delhi")
            at.sidebar.selectbox(key="_sb_scenario").set_value("Block_24h")
            at.sidebar.selectbox(key="_sb_pipeline").set_value("C")
            at.sidebar.selectbox(key="_sb_population").set_value("missing_pm")
            at.run()

            self.assertEqual(len(at.exception), 0, f"Page 03 filtered exceptions: {at.exception}")

            # Verify Pipeline D complete-case qualification works without crash
            at.sidebar.selectbox(key="_sb_pipeline").set_value("D")
            at.sidebar.selectbox(key="_sb_population").set_value("overall")
            at.run()
            self.assertEqual(len(at.exception), 0, f"Page 03 Pipeline D exceptions: {at.exception}")

        # Verify no use_container_width deprecation or SyntaxWarning from Page 03
        for w in caught:
            msg = str(w.message)
            self.assertNotIn("use_container_width", msg)
            self.assertNotEqual(w.category, SyntaxWarning, f"SyntaxWarning emitted: {msg}")

    def test_page_04_joint_tcn_default_and_filtered(self):
        """Run Page 04 with AppTest, verifying metadata, causal audit, ablation & live v2 simulator."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            at = AppTest.from_file(PAGE_04, default_timeout=45)
            at.run()

            self.assertEqual(len(at.exception), 0, f"Page 04 exceptions: {at.exception}")

            all_markdown = "\n".join(m.value for m in at.markdown)
            all_info = "\n".join(i.value for i in at.info)
            all_success = "\n".join(s.value for s in at.success)

            self.assertIn("Joint Causal TCN v2", all_markdown)
            self.assertIn("144", all_markdown + all_success)
            self.assertIn("6", all_markdown + all_info)

            # Confirm dataframes include config, validation metrics, history, scenarios, ablation, and sim
            self.assertGreaterEqual(len(at.dataframe), 4)

            # Confirm historical ablation rows from CSV appear cleanly
            ablation_found = any(
                "ablation_configuration" in df_el.value.columns
                for df_el in at.dataframe
            )
            self.assertTrue(ablation_found, "Expected joint_tcn_ablation.csv table to render.")

            # Filter interactions on Page 04 + simulator preset update
            at.sidebar.selectbox(key="_sb_site").set_value("Hamirpur")
            at.sidebar.selectbox(key="_sb_scenario").set_value("MAR_RH")
            at.sidebar.selectbox(key="_sb_population").set_value("missing_pm")
            at.selectbox(key="sim_outage_preset").set_value("Diurnal Outage (24h)")
            at.run()

            self.assertEqual(len(at.exception), 0, f"Page 04 filtered exceptions: {at.exception}")

        for w in caught:
            msg = str(w.message)
            self.assertNotIn("use_container_width", msg)
            self.assertNotEqual(w.category, SyntaxWarning, f"SyntaxWarning emitted: {msg}")

    def test_frozen_research_artifacts_immutable(self):
        """Verify SHA-256 digests of frozen Phase 3C checkpoints & benchmark predictions."""
        for rel_path, expected_sha in FROZEN_SHA256.items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Missing frozen artifact: {rel_path}")
            digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_sha, f"Checksum mismatch on {rel_path}")


if __name__ == "__main__":
    unittest.main()
