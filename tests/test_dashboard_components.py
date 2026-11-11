"""
AIRWISE Dashboard Components Unit Tests
Verifies sidebar state management, backward compatibility, reusable Plotly charts,
empty-state error safety, bootstrap chart terminology, and architecture schematics.
"""

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.config import THEME, CPCB_CATEGORIES
from dashboard.services.data_service import (
    load_cleaned_dataset,
    load_benchmark_predictions,
    load_phase3c_metrics,
    load_joint_tcn_v2_metadata,
    load_metrics_table,
)
from dashboard.components.sidebar import (
    DEFAULT_FILTERS,
    init_sidebar_state,
    get_global_filters,
    reset_global_filters,
    render_sidebar,
    render_benchmark_sidebar,
)
from dashboard.components.charts import (
    apply_dark_layout,
    create_timeseries_chart,
    create_scatter_comparison,
    create_uncertainty_ribbon_chart,
    create_cpcb_confusion_matrix,
    create_pipeline_comparison_bar,
    create_outage_degradation_chart,
    create_residual_distribution_chart,
    create_residual_vs_reference_chart,
    create_bootstrap_forest_chart,
    create_training_loss_chart,
    create_site_stratified_chart,
    create_channel_availability_chart,
    create_correlation_heatmap,
    create_storage_comparison_chart,
    PLOTLY_CHART_CONFIG,
)
from dashboard.components.architecture import (
    render_research_workflow_diagram,
    render_four_pipelines_diagram,
    render_joint_tcn_diagram,
    render_conformal_split_diagram,
)
from dashboard.components.metric_cards import render_kpi_row


class TestDashboardComponents(unittest.TestCase):
    """Tests for shared UI components in dashboard/components/."""

    def test_plotly_chart_config(self):
        """Ensure Plotly chart config disables logo and enables PNG download."""
        self.assertFalse(PLOTLY_CHART_CONFIG["displaylogo"])
        self.assertEqual(PLOTLY_CHART_CONFIG["toImageButtonOptions"]["format"], "png")

    def test_empty_state_safety_charts(self):
        """Verify all chart helpers gracefully handle empty DataFrames."""
        empty_df = pd.DataFrame()
        self.assertIsInstance(create_timeseries_chart(empty_df, ["bam_reference"]), go.Figure)
        self.assertIsInstance(create_scatter_comparison(empty_df, "x", "y", "X", "Y"), go.Figure)
        self.assertIsInstance(create_uncertainty_ribbon_chart(empty_df, "p", "l", "u"), go.Figure)
        self.assertIsInstance(create_pipeline_comparison_bar(empty_df), go.Figure)
        self.assertIsInstance(create_outage_degradation_chart(empty_df), go.Figure)
        self.assertIsInstance(create_residual_distribution_chart(empty_df, "r", "p"), go.Figure)
        self.assertIsInstance(create_residual_vs_reference_chart(empty_df, "r", "p"), go.Figure)
        self.assertIsInstance(create_bootstrap_forest_chart(empty_df), go.Figure)
        self.assertIsInstance(create_training_loss_chart([]), go.Figure)
        self.assertIsInstance(create_site_stratified_chart(empty_df), go.Figure)
        self.assertIsInstance(create_channel_availability_chart(empty_df), go.Figure)
        self.assertIsInstance(create_correlation_heatmap(empty_df, ["a", "b"]), go.Figure)
        self.assertIsInstance(create_storage_comparison_chart(empty_df), go.Figure)

    def test_phase3c_pipeline_and_degradation_charts(self):
        """Verify Phase 3C comparative and outage degradation charts on real data."""
        df_overall = load_phase3c_metrics("overall")
        df_missing = load_phase3c_metrics("missing_pm")

        fig_single = create_pipeline_comparison_bar(df_overall, metric_col="rmse", scenario_id="MCAR_20pct")
        self.assertEqual(len(fig_single.data), 4)  # A, B, C, D

        fig_all = create_pipeline_comparison_bar(df_overall, metric_col="rmse", scenario_id=None)
        self.assertEqual(len(fig_all.data), 4)

        fig_deg = create_outage_degradation_chart(df_missing, metric_col="rmse")
        # Pipeline D produces no predictions on missing PM intervals, so 3 curves (A, B, C)
        self.assertEqual(len(fig_deg.data), 3)

    def test_residual_charts_handle_nans(self):
        """Verify residual distribution & scatter handle Pipeline D NaNs safely."""
        df_bench = load_benchmark_predictions()
        fig_dist = create_residual_distribution_chart(
            df_bench, ref_col="bam_reference", pred_col="pred_D_MCAR_20pct", pipeline_name="Pipeline D"
        )
        self.assertGreaterEqual(len(fig_dist.data), 1)

        fig_vs_ref = create_residual_vs_reference_chart(
            df_bench, ref_col="bam_reference", pred_col="pred_D_MCAR_20pct"
        )
        self.assertGreaterEqual(len(fig_vs_ref.data), 1)

    def test_bootstrap_charts_terminology_and_intervals(self):
        """Verify bootstrap charts render cleanly and NEVER label tail proportion as p-value."""
        df_sig = load_phase3c_metrics("bootstrap_significance")
        df_sens = load_phase3c_metrics("bootstrap_sensitivity")

        fig_forest = create_bootstrap_forest_chart(df_sig, mode="scenarios")
        self.assertGreater(len(fig_forest.data), 1)
        fig_json = fig_forest.to_json().lower()
        self.assertNotIn("p-value", fig_json)
        self.assertNotIn("p_value", fig_json)

        fig_sweep = create_bootstrap_forest_chart(df_sens, mode="sensitivity")
        self.assertGreater(len(fig_sweep.data), 1)
        sweep_json = fig_sweep.to_json().lower()
        self.assertNotIn("p-value", sweep_json)
        self.assertNotIn("p_value", sweep_json)

    def test_training_loss_and_dataset_charts(self):
        """Verify training loss curve, site stratified, availability, heatmap, and storage charts."""
        meta = load_joint_tcn_v2_metadata()
        fig_loss = create_training_loss_chart(meta["history"], best_epoch=meta["training_summary"]["best_epoch"])
        self.assertEqual(len(fig_loss.data), 4)

        df_site = load_phase3c_metrics("site_stratified")
        fig_site = create_site_stratified_chart(df_site, metric_col="rmse", scenario_id="MCAR_20pct")
        # site_stratified_metrics.csv covers full-population pipelines A, B, C (9 scenarios * 3 sites * 3 pipelines = 81 rows)
        self.assertEqual(len(fig_site.data), 3)

        df_clean = load_cleaned_dataset().head(500)
        fig_avail = create_channel_availability_chart(df_clean)
        self.assertEqual(len(fig_avail.data), 1)

        fig_corr = create_correlation_heatmap(
            df_clean, ["bam_reference", "pa_atm", "pa_cf1", "relative_humidity", "temperature"]
        )
        self.assertEqual(len(fig_corr.data), 1)

        df_storage = load_metrics_table("checkpoint6_storage_formats")
        fig_storage = create_storage_comparison_chart(df_storage)
        self.assertEqual(len(fig_storage.data), 1)

    def test_sidebar_state_management_via_apptest(self):
        """Verify sidebar renders cleanly, returns expected structures, and persists/resets state."""
        from streamlit.testing.v1 import AppTest

        script = """
import streamlit as st
from dashboard.components.sidebar import (
    render_sidebar,
    render_benchmark_sidebar,
    get_global_filters,
    reset_global_filters,
)

site, s_dt, e_dt = render_sidebar(
    show_scenario_filter=True,
    show_pipeline_filter=True,
    show_population_filter=True,
    show_metric_filter=True,
    active_page="Test Route",
)
filters = get_global_filters()
st.write(f"RET={site}|{filters['scenario_id']}|{filters['pipeline_code']}|{filters['population_mode']}")
"""
        at = AppTest.from_string(script, default_timeout=15)
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("RET=All Sites|MCAR_20pct|B|overall" in m.value for m in at.markdown))

        # Test changing site and scenario in sidebar
        at.sidebar.selectbox[0].set_value("Delhi").run()
        at.sidebar.selectbox[1].set_value("Block_24h").run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("RET=Delhi|Block_24h|B|overall" in m.value for m in at.markdown))

        # Test Reset button restores defaults
        at.sidebar.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("RET=All Sites|MCAR_20pct|B|overall" in m.value for m in at.markdown))

    def test_architecture_diagrams_via_apptest(self):
        """Verify all four architecture schematics render cleanly with zero errors."""
        from streamlit.testing.v1 import AppTest

        script = """
from dashboard.components.architecture import (
    render_research_workflow_diagram,
    render_four_pipelines_diagram,
    render_joint_tcn_diagram,
    render_conformal_split_diagram,
)
from dashboard.components.metric_cards import render_kpi_row

render_kpi_row([])  # Empty state guard
render_research_workflow_diagram()
render_four_pipelines_diagram(active_pipeline="C")
render_joint_tcn_diagram()
render_conformal_split_diagram()
"""
        at = AppTest.from_string(script, default_timeout=15)
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.markdown), 4)


if __name__ == "__main__":
    unittest.main()
