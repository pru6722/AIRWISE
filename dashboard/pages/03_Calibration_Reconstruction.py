"""
AIRWISE Dashboard: Calibration & Reconstruction Lab
Phase 3C Four-Pipeline Comparative Benchmark across Nine Missingness Regimes,
Evaluation Population Isolation, Residual Diagnostics, Outage Degradation,
and Moving-Block Bootstrap Statistical Uncertainty.
"""

import sys
from pathlib import Path

# Ensure project root and dashboard package are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
for p in [str(PROJECT_ROOT), str(DASHBOARD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.config import (
    THEME,
    SITES,
    PIPELINES,
    SCENARIOS,
    EVALUATION_POPULATIONS,
    FEATURE_SETS,
)
from dashboard.components.sidebar import (
    render_benchmark_sidebar,
    DEFAULT_MIN_DATE,
    DEFAULT_MAX_DATE,
)
from dashboard.components.metric_cards import render_kpi_row, render_metric_card
from dashboard.components.architecture import render_four_pipelines_diagram
from dashboard.components.charts import (
    PLOTLY_CHART_CONFIG,
    create_pipeline_comparison_bar,
    create_outage_degradation_chart,
    create_timeseries_chart,
    create_scatter_comparison,
    create_residual_distribution_chart,
    create_residual_vs_reference_chart,
    create_site_stratified_chart,
    create_bootstrap_forest_chart,
)
from dashboard.services.data_service import (
    load_benchmark_predictions,
    load_phase3c_metrics,
    load_cleaned_dataset,
    filter_benchmark_predictions,
    compute_evaluation_metrics,
)
from dashboard.services.calibration_service import (
    run_live_calibration_inference,
    ALGORITHMS,
    PARADIGMS,
    FEATURE_SET_NAMES,
)
from dashboard.utils.validation import validate_sensor_inputs

# Page Setup
st.set_page_config(
    page_title="AIRWISE — Calibration & Reconstruction Lab",
    layout="wide",
    page_icon="⚙️",
)

# Inject Dark Theme CSS
css_path = DASHBOARD_DIR / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Persistent Shared Benchmark Sidebar
filters = render_benchmark_sidebar(
    show_site_filter=True,
    show_date_filter=True,
    show_scenario_filter=True,
    show_pipeline_filter=True,
    show_population_filter=True,
    show_metric_filter=True,
    active_page="Calibration & Reconstruction Lab",
)

selected_site = filters["site"]
start_date = filters["start_date"]
end_date = filters["end_date"]
scenario_id = filters["scenario_id"]
pipeline_code = filters["pipeline_code"]
population_mode = filters["population_mode"]
metric_col = filters["metric"]

sc_info = SCENARIOS.get(scenario_id, {"name": scenario_id, "category": "", "description": ""})
pipe_info = PIPELINES.get(pipeline_code, {"name": pipeline_code, "description": ""})
pop_info = EVALUATION_POPULATIONS.get(population_mode, {"name": population_mode, "description": ""})

# Page Header
st.title("⚙️ Calibration & Reconstruction Lab")
st.markdown(
    """
    Compare **Pipeline A (Impute → Calibrate)**, **Pipeline B (Calibrate → Impute)**,
    **Pipeline C (Joint Causal TCN v2)**, and **Pipeline D (Complete-Case Reference)**
    across all **nine pre-registered missingness scenarios** under strictly isolated evaluation populations.
    """
)

# Active Configuration Banner
st.markdown(
    f"""
    <div class="research-alert" style="margin-bottom: 14px;">
        <strong>Active Benchmark Slice:</strong>
        <span class="badge-pipeline" style="background:rgba(0,168,255,0.16);color:#00a8ff;border:1px solid #00a8ff;margin-left:6px;">
            Scenario: {sc_info['name']}
        </span>
        <span class="badge-pipeline" style="background:rgba(0,210,196,0.16);color:{pipe_info['color']};border:1px solid {pipe_info['color']};margin-left:6px;">
            Primary: {pipe_info['name']}
        </span>
        <span class="badge-population" style="margin-left:6px;">
            Population: {pop_info['name']}
        </span>
        <span class="badge-pipeline" style="background:rgba(148,163,184,0.14);color:#cbd5e1;border:1px solid #475569;margin-left:6px;">
            Station: {selected_site}
        </span>
        <div style="font-size:0.82rem;color:#94a3b8;margin-top:6px;">
            <strong>Mechanism Note:</strong> {sc_info['description']} &nbsp;|&nbsp;
            <strong>Population Rule:</strong> {pop_info['description']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Scenario-Specific Scientific Disclosures
if scenario_id == "MNAR_PM":
    st.markdown(
        """
        <div class="research-warning">
            <strong>MNAR-PM Synthetic Oracle Disclosure:</strong>
            Missingness in <code>MNAR_PM</code> is triggered by unobserved extreme PM concentrations via the benchmark oracle generator.
            No BAM targets or oracle thresholds reach any prediction pipeline; masks serve strictly as synthetic stress-test conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif scenario_id == "Block_24h":
    st.markdown(
        """
        <div class="research-warning">
            <strong>Outage Coalescence Disclosure (Nominal 24h vs Merged Duration):</strong>
            Nominal 24-hour block insertion within bounded temporal segments allows adjacent randomly placed blocks to coalesce,
            generating continuous empirical outages averaging <strong>32.61 hours</strong> (maximum merged duration: <strong>63 hours</strong>).
        </div>
        """,
        unsafe_allow_html=True,
    )

# Load Frozen Benchmark Data & Tables
df_bench = load_benchmark_predictions()
df_overall_csv = load_phase3c_metrics("overall")
df_missing_csv = load_phase3c_metrics("missing_pm")
df_common_csv = load_phase3c_metrics("common_support")
df_site_csv = load_phase3c_metrics("site_stratified")
df_boot_sig = load_phase3c_metrics("bootstrap_significance")
df_boot_sens = load_phase3c_metrics("bootstrap_sensitivity")

pop_table_map = {
    "overall": df_overall_csv,
    "missing_pm": df_missing_csv,
    "common_support": df_common_csv,
}
active_csv_table = pop_table_map.get(population_mode, df_overall_csv)

# Filter benchmark predictions strictly by site, date, scenario, and population mode
is_full_test_slice = (
    selected_site == "All Sites"
    and start_date == DEFAULT_MIN_DATE
    and end_date == DEFAULT_MAX_DATE
)

df_filtered = filter_benchmark_predictions(
    df_bench,
    scenario_id=scenario_id,
    site=selected_site,
    start_date=start_date,
    end_date=end_date,
    population_mode=population_mode,
)

# Compute/Retrieve KPI metrics for the selected pipeline + scenario + population
pred_col = f"pred_{pipeline_code}_{scenario_id}"
mask_col = f"mask_{scenario_id}"

if is_full_test_slice and not active_csv_table.empty:
    row_match = active_csv_table[
        (active_csv_table["scenario_id"] == scenario_id)
        & (active_csv_table["pipeline_code"] == pipeline_code)
    ]
    if not row_match.empty:
        r0 = row_match.iloc[0]
        kpi_rmse = f"{r0['rmse']:.2f} µg/m³" if pd.notna(r0["rmse"]) else "N/A (No Missing Impute)"
        kpi_mae = f"{r0['mae']:.2f} µg/m³" if pd.notna(r0["mae"]) else "N/A"
        kpi_r2 = f"{r0['r2']:.4f}" if pd.notna(r0["r2"]) else "N/A"
        kpi_cov = f"{r0['coverage_pct']:.2f}%"
        kpi_n = f"{int(r0['n_samples']):,}"
        kpi_src_note = f"Frozen Phase 3C Metric ({pop_info['name']})"
    else:
        kpi_rmse = kpi_mae = kpi_r2 = kpi_cov = kpi_n = "Unavailable"
        kpi_src_note = "Artifact entry unavailable"
else:
    # Filtered slice computed dynamically from frozen predictions
    if not df_filtered.empty and pred_col in df_filtered.columns:
        slice_m = compute_evaluation_metrics(
            df_filtered["bam_reference"].to_numpy(dtype=float),
            df_filtered[pred_col].to_numpy(dtype=float),
        )
        cov_pct = (slice_m["n_samples"] / len(df_filtered) * 100.0) if len(df_filtered) > 0 else 0.0
        kpi_rmse = f"{slice_m['rmse']:.2f} µg/m³" if pd.notna(slice_m["rmse"]) else "N/A"
        kpi_mae = f"{slice_m['mae']:.2f} µg/m³" if pd.notna(slice_m["mae"]) else "N/A"
        kpi_r2 = f"{slice_m['r2']:.4f}" if pd.notna(slice_m["r2"]) else "N/A"
        kpi_cov = f"{cov_pct:.2f}%"
        kpi_n = f"{slice_m['n_samples']:,} / {len(df_filtered):,}"
    else:
        kpi_rmse = kpi_mae = kpi_r2 = kpi_cov = kpi_n = "0"
    kpi_src_note = f"Filtered Slice ({selected_site} | {start_date} to {end_date})"

render_kpi_row([
    {"title": "RMSE", "value": kpi_rmse, "subtitle": kpi_src_note, "color": pipe_info["color"]},
    {"title": "MAE", "value": kpi_mae, "subtitle": sc_info["name"], "color": THEME["accent_cyan"]},
    {"title": "R² Score", "value": kpi_r2, "subtitle": f"Pipeline {pipeline_code}", "color": THEME["accent_teal"]},
    {"title": "Prediction Coverage", "value": kpi_cov, "subtitle": "Eligible / Total in Slice", "color": THEME["accent_amber"] if pipeline_code == "D" else THEME["accent_teal"]},
    {"title": "Evaluated Samples (N)", "value": kpi_n, "subtitle": pop_info["name"], "color": THEME["accent_purple"]},
])

# Explicit Pipeline D Coverage Qualification
if pipeline_code == "D" or population_mode == "overall":
    st.markdown(
        """
        <div class="research-alert" style="border-left-color: #ffb86c; margin-top: 4px; margin-bottom: 16px;">
            <strong>Pipeline D Complete-Case Eligibility Rule:</strong>
            Pipeline D performs zero imputation and outputs predictions <strong>only when PM channels are observed</strong> (<code>m_t = 1</code>).
            On the Missing-PM subgroup (<code>m_t = 0</code>), Pipeline D produces zero predictions. Under Full Test evaluation, its lower coverage
            (e.g., 79.33% on 20% MCAR) reflects complete-case selection bias and must never be directly compared to 100%-coverage reconstruction RMSEs.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Main Lab Tabs
tab_overview, tab_ts, tab_residuals, tab_sites_outages, tab_bootstrap, tab_methodology = st.tabs([
    "📊 Four-Pipeline Benchmark",
    "📈 Time-Series & Scatter",
    "🎯 Residual Diagnostics",
    "🌍 Site & Outage Degradation",
    "⚖️ Moving-Block Bootstrap",
    "🔬 Methodology & Live Calibrator",
])

# -----------------------------------------------------------------------------
# TAB 1: FOUR-PIPELINE BENCHMARK COMPARISON
# -----------------------------------------------------------------------------
with tab_overview:
    st.markdown("### Structural Pipeline Architecture Overview")
    render_four_pipelines_diagram(active_pipeline=pipeline_code)

    st.markdown("### Comparative Performance Across Pipelines & Missingness Regimes")

    # Determine table data (frozen table if full slice, dynamic aggregation if site/date filtered)
    if is_full_test_slice and not active_csv_table.empty:
        table_df = active_csv_table.copy()
        source_badge = f"Frozen Official Phase 3C Table — {pop_info['name']}"
    else:
        rows_dyn = []
        for sid, smeta in SCENARIOS.items():
            sub_df = filter_benchmark_predictions(
                df_bench,
                scenario_id=sid,
                site=selected_site,
                start_date=start_date,
                end_date=end_date,
                population_mode=population_mode,
            )
            for code in ["A", "B", "C", "D"]:
                pcol = f"pred_{code}_{sid}"
                if sub_df.empty or pcol not in sub_df.columns:
                    continue
                m = compute_evaluation_metrics(
                    sub_df["bam_reference"].to_numpy(dtype=float),
                    sub_df[pcol].to_numpy(dtype=float),
                )
                n_total_slice = len(sub_df)
                cov = (m["n_samples"] / n_total_slice * 100.0) if n_total_slice > 0 else 0.0
                rows_dyn.append({
                    "scenario_id": sid,
                    "scenario_name": smeta["name"],
                    "category": smeta["category"],
                    "pipeline_code": code,
                    "pipeline_name": PIPELINES[code]["name"],
                    "evaluated_population": pop_info["name"],
                    "n_samples": m["n_samples"],
                    "coverage_pct": round(cov, 2),
                    "missing_predictions": int(n_total_slice - m["n_samples"]),
                    "rmse": round(m["rmse"], 2) if pd.notna(m["rmse"]) else np.nan,
                    "mae": round(m["mae"], 2) if pd.notna(m["mae"]) else np.nan,
                    "r2": round(m["r2"], 4) if pd.notna(m["r2"]) else np.nan,
                    "mbe": round(m["mbe"], 2) if pd.notna(m["mbe"]) else np.nan,
                })
        table_df = pd.DataFrame(rows_dyn)
        source_badge = f"Computed Dynamically from Frozen Predictions ({selected_site})"

    col_c1, col_c2 = st.columns([1.0, 1.4])
    with col_c1:
        fig_single = create_pipeline_comparison_bar(
            table_df,
            metric_col=metric_col,
            scenario_id=scenario_id,
            population_label=pop_info["name"],
        )
        st.plotly_chart(fig_single, config=PLOTLY_CHART_CONFIG, width="stretch")
    with col_c2:
        fig_all = create_pipeline_comparison_bar(
            table_df,
            metric_col=metric_col,
            scenario_id=None,
            population_label=pop_info["name"],
        )
        st.plotly_chart(fig_all, config=PLOTLY_CHART_CONFIG, width="stretch")

    st.markdown(f"#### Quantitative Comparison Matrix ({source_badge})")
    scope_toggle = st.radio(
        "Table Display Scope:",
        options=["Active Scenario Only", "All Nine Scenarios"],
        horizontal=True,
        key="tab1_scope_toggle",
    )

    disp_table = (
        table_df[table_df["scenario_id"] == scenario_id].copy()
        if scope_toggle == "Active Scenario Only"
        else table_df.copy()
    )

    if not disp_table.empty:
        preferred_cols = [
            "scenario_name",
            "pipeline_code",
            "pipeline_name",
            "evaluated_population",
            "n_samples",
            "coverage_pct",
            "missing_predictions",
            "rmse",
            "mae",
            "r2",
            "mbe",
        ]
        show_cols = [c for c in preferred_cols if c in disp_table.columns]
        st.dataframe(disp_table[show_cols], width="stretch", hide_index=True)

        csv_export = disp_table[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download Comparison Report as CSV ({len(disp_table)} rows)",
            data=csv_export,
            file_name=f"airwise_benchmark_{population_mode}_{scenario_id.lower()}.csv",
            mime="text/csv",
        )
    else:
        st.info("No metric rows available for the selected filters.")


# -----------------------------------------------------------------------------
# TAB 2: PREDICTION VS BAM TIME-SERIES & SCATTER DIAGNOSTICS
# -----------------------------------------------------------------------------
with tab_ts:
    st.markdown(f"### Prediction Trajectories vs. Ground-Truth BAM — {sc_info['name']} ({pop_info['name']})")

    if df_filtered.empty:
        st.info("No test observations match the selected Station, Date Range, and Evaluation Population filters.")
    else:
        col_overlay_ctrl, col_info_ts = st.columns([2, 1])
        with col_overlay_ctrl:
            overlay_pipelines = st.multiselect(
                "Overlay Pipelines on Time-Series:",
                options=["A", "B", "C", "D"],
                default=[pipeline_code],
                format_func=lambda k: PIPELINES[k]["name"],
            )
        with col_info_ts:
            n_outage = int((~df_filtered[mask_col]).sum()) if mask_col in df_filtered.columns else 0
            st.metric("Missing-PM Timestamps in Slice", f"{n_outage:,} / {len(df_filtered):,}")

        y_cols = ["bam_reference"] + [f"pred_{c}_{scenario_id}" for c in overlay_pipelines]
        labels = {"bam_reference": "BAM-1020 Reference (Ground Truth)"}
        colors = {"bam_reference": THEME["text_primary"]}
        for c in overlay_pipelines:
            col_name = f"pred_{c}_{scenario_id}"
            labels[col_name] = PIPELINES[c]["short_name"]
            colors[col_name] = PIPELINES[c]["color"]

        df_ts_sorted = df_filtered.sort_values("timestamp")
        fig_ts = create_timeseries_chart(
            df_ts_sorted,
            y_cols=y_cols,
            labels=labels,
            colors=colors,
            title=f"Predicted vs. BAM Reference ({sc_info['name']} — {selected_site})",
            show_rangeslider=True,
        )
        st.plotly_chart(fig_ts, config=PLOTLY_CHART_CONFIG, width="stretch")

        st.markdown(f"#### Predicted vs. BAM Reference Scatter ({pipe_info['name']})")
        fig_scat = create_scatter_comparison(
            df_filtered,
            x_col=pred_col,
            y_col="bam_reference",
            x_title=f"{pipe_info['short_name']} Prediction (µg/m³)",
            y_title="BAM-1020 Reference (µg/m³)",
            color_col="site" if selected_site == "All Sites" else None,
            title=f"Agreement Scatter: {pipe_info['short_name']} vs BAM ({sc_info['name']})",
        )
        st.plotly_chart(fig_scat, config=PLOTLY_CHART_CONFIG, width="stretch")

        # Export slice predictions
        export_cols = ["timestamp", "site", "segment_id", "bam_reference", mask_col] + [
            f"pred_{c}_{scenario_id}" for c in ["A", "B", "C", "D"] if f"pred_{c}_{scenario_id}" in df_filtered.columns
        ]
        slice_csv = df_filtered[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download Filtered Test Predictions ({len(df_filtered):,} rows)",
            data=slice_csv,
            file_name=f"airwise_predictions_{scenario_id}_{selected_site.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


# -----------------------------------------------------------------------------
# TAB 3: RESIDUAL DIAGNOSTICS
# -----------------------------------------------------------------------------
with tab_residuals:
    st.markdown(f"### Prediction Residual Diagnostics — {pipe_info['name']} ({sc_info['name']})")

    if df_filtered.empty or pred_col not in df_filtered.columns:
        st.info("No predictions available for residual analysis under the active filters.")
    else:
        valid_res_df = df_filtered.dropna(subset=["bam_reference", pred_col]).copy()
        if valid_res_df.empty:
            st.warning(
                f"No non-null prediction-reference pairs exist for {pipe_info['name']} in {pop_info['name']}. "
                "(Note: Pipeline D generates zero predictions on missing PM intervals.)"
            )
        else:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_hist = create_residual_distribution_chart(
                    valid_res_df,
                    ref_col="bam_reference",
                    pred_col=pred_col,
                    pipeline_name=pipe_info["short_name"],
                )
                st.plotly_chart(fig_hist, config=PLOTLY_CHART_CONFIG, width="stretch")

            with col_r2:
                fig_res_ref = create_residual_vs_reference_chart(
                    valid_res_df,
                    ref_col="bam_reference",
                    pred_col=pred_col,
                    color_col="site" if selected_site == "All Sites" else None,
                )
                st.plotly_chart(fig_res_ref, config=PLOTLY_CHART_CONFIG, width="stretch")

            residuals = valid_res_df[pred_col] - valid_res_df["bam_reference"]
            q_summary = pd.DataFrame({
                "Statistic": ["Count (Valid)", "Mean Bias Error (MBE)", "Median Bias (P50)", "Std Deviation", "5th Percentile (P05)", "25th Percentile (P25)", "75th Percentile (P75)", "95th Percentile (P95)"],
                "Value (µg/m³)": [
                    f"{len(residuals):,}",
                    f"{residuals.mean():+.2f}",
                    f"{residuals.median():+.2f}",
                    f"{residuals.std():.2f}",
                    f"{residuals.quantile(0.05):+.2f}",
                    f"{residuals.quantile(0.25):+.2f}",
                    f"{residuals.quantile(0.75):+.2f}",
                    f"{residuals.quantile(0.95):+.2f}",
                ],
            })
            st.markdown("#### Empirical Residual Quantiles Summary")
            st.dataframe(q_summary, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# TAB 4: SITE-WISE & OUTAGE DURATION ANALYSIS
# -----------------------------------------------------------------------------
with tab_sites_outages:
    st.markdown("### Site-Stratified Performance & Nominal Block-Outage Trajectories")

    col_so1, col_so2 = st.columns(2)
    with col_so1:
        st.markdown("#### Site-Wise Comparison (Full-Coverage Pipelines A, B, C)")
        fig_site = create_site_stratified_chart(
            df_site_csv,
            metric_col=metric_col if metric_col in ["rmse", "mae", "r2"] else "rmse",
            scenario_id=scenario_id,
        )
        st.plotly_chart(fig_site, config=PLOTLY_CHART_CONFIG, width="stretch")
        st.caption("Source: `site_stratified_metrics.csv` (tracks complete-coverage pipelines A, B, C across Bangalore, Delhi, and Hamirpur).")

    with col_so2:
        st.markdown("#### Error Degradation Across Block Outage Scenarios (1h – 24h Nominal)")
        deg_pop = st.radio(
            "Outage Trajectory Population:",
            options=["Missing-PM Subgroup", "Full Test Population"],
            horizontal=True,
            key="outage_deg_pop",
        )
        deg_df = df_missing_csv if deg_pop == "Missing-PM Subgroup" else df_overall_csv
        fig_deg = create_outage_degradation_chart(
            deg_df,
            metric_col=metric_col if metric_col in ["rmse", "mae", "r2"] else "rmse",
            population_label=deg_pop,
        )
        st.plotly_chart(fig_deg, config=PLOTLY_CHART_CONFIG, width="stretch")
        st.caption(
            "Note: Nominal 24h insertion blocks coalesce into continuous merged outages (mean 32.61h, max 63h). "
            "Sequential baselines (A & B) exhibit non-monotonic fluctuations across intermediate block lengths."
        )

    if not df_site_csv.empty:
        site_view = df_site_csv[df_site_csv["scenario_id"] == scenario_id]
        if selected_site != "All Sites":
            site_view = site_view[site_view["site"] == selected_site]
        st.markdown(f"#### Exact Site-Stratified Metrics — {sc_info['name']}")
        st.dataframe(site_view, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# TAB 5: MOVING-BLOCK BOOTSTRAP STATISTICAL UNCERTAINTY
# -----------------------------------------------------------------------------
with tab_bootstrap:
    st.markdown("### Moving-Block Bootstrap Uncertainty & Diurnal Sensitivity Sweep")
    st.markdown(
        """
        <div class="research-warning">
            <strong>Statistical Uncertainty Interpretation Protocol:</strong>
            We use <strong>paired 95% moving-block bootstrap confidence intervals</strong> (resampling 24-hour diurnal blocks, B = 1,000)
            as our primary uncertainty evidence. The uncentered fraction of bootstrap replicates crossing zero is labeled strictly as an
            <strong>Empirical Uncentered Bootstrap Tail Proportion (p̂_tail)</strong> — <em>never</em> as a conventional null-hypothesis test p-value.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_b1, col_b2 = st.columns([1.2, 1.0])
    with col_b1:
        boot_sc_filter = st.selectbox(
            "Filter Bootstrap Forest Plot:",
            options=["All Nine Scenarios", f"Active Scenario Only ({sc_info['name']})"],
            index=0,
        )
        sig_plot_df = (
            df_boot_sig[df_boot_sig["scenario_id"] == scenario_id]
            if boot_sc_filter != "All Nine Scenarios"
            else df_boot_sig
        )
        fig_forest = create_bootstrap_forest_chart(sig_plot_df, mode="scenarios")
        st.plotly_chart(fig_forest, config=PLOTLY_CHART_CONFIG, width="stretch")

    with col_b2:
        fig_sweep = create_bootstrap_forest_chart(df_boot_sens, mode="sensitivity")
        st.plotly_chart(fig_sweep, config=PLOTLY_CHART_CONFIG, width="stretch")
        st.markdown(
            """
            <div class="research-alert" style="font-size:0.84rem;">
                <strong>Critical Diurnal Autocorrelation Finding:</strong>
                Under short 12h blocks on 20% MCAR, the 95% CI ([-5.27, -0.37]) excludes zero (p̂_tail = 0.010).
                Once the 24-hour diurnal atmospheric cycle is respected (b = 24h), the interval expands across zero
                ([-4.31, +0.70], p̂_tail = 0.084), demonstrating that short-block advantages do not establish robust superiority.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Paired Bootstrap Confidence Intervals (Canonical b = 24h, B = 1,000)")
    if not df_boot_sig.empty:
        # Format table cleanly without using "p-value" label
        sig_disp = df_boot_sig.copy()
        sig_disp["95% CI Excludes Zero?"] = (sig_disp["ci_lower_95"] > 0) | (sig_disp["ci_upper_95"] < 0)
        sig_disp = sig_disp.rename(columns={
            "delta_rmse": "Point ΔRMSE (µg/m³)",
            "ci_lower_95": "95% CI Lower",
            "ci_upper_95": "95% CI Upper",
            "p_value_rmse": "Tail Proportion (p̂_tail)",
        })
        cols_keep = [
            "scenario_name",
            "comparison",
            "block_length_h",
            "Point ΔRMSE (µg/m³)",
            "95% CI Lower",
            "95% CI Upper",
            "Tail Proportion (p̂_tail)",
            "95% CI Excludes Zero?",
        ]
        st.dataframe(sig_disp[[c for c in cols_keep if c in sig_disp.columns]], width="stretch", hide_index=True)

    st.markdown("#### Block-Length Sensitivity Sweep (b ∈ {12, 24, 48, 72} Hours)")
    if not df_boot_sens.empty:
        sens_disp = df_boot_sens.copy()
        sens_disp["95% CI Excludes Zero?"] = (sens_disp["ci_lower_95"] > 0) | (sens_disp["ci_upper_95"] < 0)
        sens_disp = sens_disp.rename(columns={
            "block_length_hours": "Block Length (h)",
            "delta_rmse": "Point ΔRMSE (µg/m³)",
            "ci_lower_95": "95% CI Lower",
            "ci_upper_95": "95% CI Upper",
            "ci_width": "CI Width (µg/m³)",
            "p_value": "Tail Proportion (p̂_tail)",
        })
        cols_sens = [
            "scenario_id",
            "comparison",
            "Block Length (h)",
            "Point ΔRMSE (µg/m³)",
            "95% CI Lower",
            "95% CI Upper",
            "CI Width (µg/m³)",
            "Tail Proportion (p̂_tail)",
            "95% CI Excludes Zero?",
        ]
        st.dataframe(sens_disp[[c for c in cols_sens if c in sens_disp.columns]], width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# TAB 6: METHODOLOGY EXPLAINERS & ILLUSTRATIVE LIVE CALIBRATION PLAYGROUND
# -----------------------------------------------------------------------------
with tab_methodology:
    st.markdown("### Methodology Explainers & Experimental Reference")

    with st.expander("1. Four Comparative Architectures (A, B, C, D)", expanded=True):
        for code, pinfo in PIPELINES.items():
            st.markdown(f"- **{pinfo['name']}** (`{pinfo['paradigm']}`): {pinfo['description']}")

    with st.expander("2. Nine Missingness Regimes & Outage Mechanics"):
        for sid, smeta in SCENARIOS.items():
            st.markdown(f"- **{smeta['name']}** (`{smeta['category']}`): {smeta['description']}")

    with st.expander("3. Authoritative Evaluation Population Boundaries"):
        for pkey, pmeta in EVALUATION_POPULATIONS.items():
            st.markdown(f"- **{pmeta['name']}** (*{pmeta['subtitle']}*): {pmeta['description']}")

    st.markdown("---")
    st.markdown("### Illustrative Live Single-Step Calibration Playground (Joblib CPU Inference)")
    st.caption(
        "Optional demonstration using serialized single-step calibration regressors fitted on the training partition. "
        "Inputs adjusted here represent hypothetical single-hour sensor vectors, not a benchmark rerun."
    )

    col_ctrl, col_res = st.columns([1, 1.1])
    with col_ctrl:
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_algo = st.selectbox("Algorithm", options=ALGORITHMS, index=0)
        with c2:
            sel_paradigm = st.selectbox("Training Paradigm", options=PARADIGMS, index=0)
        with c3:
            sel_fset = st.selectbox("Channel Subset", options=FEATURE_SET_NAMES, index=0)

        in_pa_atm = st.slider("Synthetic PA_ATM (µg/m³)", 0.0, 400.0, 65.0, 1.0)
        in_pa_cf1 = st.slider("Synthetic PA_CF1 (µg/m³)", 0.0, 600.0, 95.0, 1.0)
        in_pa_alt = st.slider("Synthetic PA_ALT (µg/m³)", 0.0, 300.0, 25.0, 1.0)
        in_rh = st.slider("Synthetic Relative Humidity (%)", 10.0, 95.0, 68.0, 1.0)
        in_temp = st.slider("Synthetic Temperature (°C)", 5.0, 45.0, 26.0, 0.5)

        is_valid, warnings = validate_sensor_inputs(in_pa_cf1, in_pa_atm, in_pa_alt, in_rh, in_temp)
        for w in warnings:
            st.warning(w)

    with col_res:
        pred_val, feat_cols = run_live_calibration_inference(
            paradigm=sel_paradigm,
            algorithm=sel_algo,
            feature_set=sel_fset,
            pa_cf1=in_pa_cf1,
            pa_atm=in_pa_atm,
            pa_alt=in_pa_alt,
            relative_humidity=in_rh,
            temperature=in_temp,
        )
        if pred_val is not None:
            render_metric_card(
                title=f"Single-Step Calibrated PM2.5 ({sel_algo})",
                value=f"{pred_val:.1f} µg/m³",
                subtitle=f"Paradigm: {sel_paradigm} | Subset: {sel_fset}",
                delta=f"Correction vs Raw PA_ATM: {pred_val - in_pa_atm:+.1f} µg/m³",
                color=THEME["accent_teal"],
            )
        else:
            st.error("Selected model joblib artifact unavailable on disk.")
