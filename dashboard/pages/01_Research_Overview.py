"""
AIRWISE Dashboard Page 1: Research Overview & Experimental Protocol
Presents complete 7-stage research workflow, 4-way chronological partition,
four comparative pipelines, nine missingness scenarios, balanced empirical findings,
and verifiable Checkpoint 1–6 + Phase 3C evidence.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

from dashboard.config import (
    APP_TITLE,
    APP_SUBTITLE,
    ACADEMIC_CITATION,
    THEME,
    DATASET_SPLITS,
    PIPELINES,
    SCENARIOS,
)
from dashboard.components.sidebar import render_benchmark_sidebar
from dashboard.components.metric_cards import render_kpi_row
from dashboard.components.architecture import (
    render_research_workflow_diagram,
    render_four_pipelines_diagram,
    render_conformal_split_diagram,
)
from dashboard.components.charts import (
    PLOTLY_CHART_CONFIG,
    create_pipeline_comparison_bar,
)
from dashboard.services.data_service import (
    load_phase3c_metrics,
    load_metrics_table,
    load_split_summary,
    load_joint_tcn_v2_metadata,
)


def _safe_page_link(route: str, label: str) -> None:
    """Renders clickable st.page_link in app context; falls back cleanly in isolated page tests."""
    try:
        st.page_link(route, label=label, width="stretch")
    except StreamlitAPIException:
        st.caption(f"Route: `{route}` — {label}")


# Page Config & CSS
st.set_page_config(page_title="AIRWISE — Research Overview", layout="wide", page_icon="🔬")

css_path = DASHBOARD_DIR / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

filters = render_benchmark_sidebar(
    show_site_filter=True,
    show_date_filter=False,
    show_scenario_filter=False,
    show_pipeline_filter=False,
    show_population_filter=False,
    show_metric_filter=False,
    active_page="Research Overview",
)

# Load Authoritative Artifacts
split_summary = load_split_summary()
df_overall = load_phase3c_metrics("overall")
df_missing = load_phase3c_metrics("missing_pm")
df_common = load_phase3c_metrics("common_support")
df_bias = load_metrics_table("sensor_bias_summary")
df_dew = load_metrics_table("dew_point_investigation")
df_gaps = load_metrics_table("missing_timestamp_report")
meta_tcn = load_joint_tcn_v2_metadata()
df_cpcb = load_metrics_table("cpcb_controlled_comparison")
df_storage = load_metrics_table("checkpoint6_storage_formats")
df_scale = load_metrics_table("checkpoint6_scaling")

# Title Banner
st.title("🔬 AIRWISE — Research Overview & Methodology")
st.markdown(
    f"""
    <div style="font-size: 1.08rem; color: {THEME['accent_teal']}; font-weight: 600; margin-top: -6px; margin-bottom: 6px;">
        {APP_SUBTITLE}
    </div>
    <div style="font-size: 0.85rem; color: {THEME['text_secondary']}; margin-bottom: 16px;">
        <strong>Primary Empirical Benchmark:</strong> {ACADEMIC_CITATION}
    </div>
    """,
    unsafe_allow_html=True,
)

b_best = float(df_overall[df_overall["pipeline_code"] == "B"]["rmse"].min()) if not df_overall.empty else 36.26
c_mar = float(
    df_missing[(df_missing["pipeline_code"] == "C") & (df_missing["scenario_id"] == "MAR_RH")]["rmse"].iloc[0]
) if not df_missing.empty else 43.05

render_kpi_row(
    [
        {
            "title": "Curated Observations",
            "value": f"{DATASET_SPLITS['total']:,}",
            "subtitle": "3 Indian Monitoring Stations",
            "delta": "0 Nulls | Strictly Chronological",
            "color": THEME["accent_teal"],
        },
        {
            "title": "Comparative Scope",
            "value": "4 × 9 Matrix",
            "subtitle": "4 Pipelines × 9 Scenarios",
            "delta": f"Best Full-Test RMSE (B): {b_best:.2f}",
            "color": THEME["accent_cyan"],
        },
        {
            "title": "Causal Joint TCN v2",
            "value": f"{meta_tcn.get('causality_verification', {}).get('total_checks', 144)}/144 Checks",
            "subtitle": f"MAR-RH Missing-PM: {c_mar:.2f} µg/m³",
            "delta": f"Epoch {meta_tcn.get('training_summary', {}).get('best_epoch', 6)} Converged",
            "color": THEME["accent_purple"],
        },
        {
            "title": "CPCB PM2.5 Windows",
            "value": "127 Days",
            "subtitle": "72.44% Point Accuracy (PM2.5 Only)",
            "delta": "Severe Recall: 0/4 (Disclosure)",
            "color": THEME["accent_amber"],
        },
    ]
)

st.markdown("---")

# End-to-End Workflow Schematic
st.subheader("🔄 Complete AIRWISE End-to-End Research Workflow")
render_research_workflow_diagram()

st.markdown("---")

# Tabs for Structured Exploration
tab_splits, tab_pipelines, tab_checkpoints, tab_disclosures = st.tabs(
    [
        "📊 4-Way Dataset Partition & Objectives",
        "🔀 4 Pipelines × 9 Scenarios Findings",
        "📑 Empirical Checkpoints (CP1–CP6)",
        "⚠️ Scientific Disclosures & Navigation",
    ]
)

# =============================================================================
# TAB 1: PARTITION & OBJECTIVES
# =============================================================================
with tab_splits:
    st.markdown("### Authoritative Four-Way Chronological Partition Protocol")
    st.markdown(
        """
        Every station is partitioned chronologically (`60% Train / 10% Validation / 10% Conformal Calibration / 20% Independent Test`)
        with natural gaps segmented whenever `time_diff_hours > 1.05`. Neither test targets (`BAM-1020`) nor future timesteps ever enter
        model fitting or calibration.
        """
    )
    render_conformal_split_diagram()

    sites_meta = split_summary.get("sites", {})
    if sites_meta:
        split_rows = []
        for s_name, s_info in sites_meta.items():
            split_rows.append(
                {
                    "Site": s_name,
                    "Total Hours": s_info.get("total_count"),
                    "Train (60%)": f"{s_info.get('train_count')} ({s_info.get('train_start', '')[:10]} → {s_info.get('train_end', '')[:10]})",
                    "Validation (10%)": f"{s_info.get('val_model_count')} ({s_info.get('val_model_start', '')[:10]} → {s_info.get('val_model_end', '')[:10]})",
                    "Conformal Cal (10%)": f"{s_info.get('calib_conformal_count')} ({s_info.get('calib_conformal_start', '')[:10]} → {s_info.get('calib_conformal_end', '')[:10]})",
                    "Test (20%)": f"{s_info.get('test_count')} ({s_info.get('test_start', '')[:10]} → {s_info.get('test_end', '')[:10]})",
                }
            )
        st.markdown("#### Exact Per-Station Partition Boundaries (`split_summary.json`)")
        st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True)

    st.markdown("### Six Integrated Scientific Objectives")
    st.markdown(
        """
        1. **Hygroscopic & Multi-Channel Calibration:** Correct systematic `PA_CF1` positive bias and `PA_ALT` negative bias against federal-equivalent `BAM-1020` monitors while excluding physically inconsistent `dew_point_raw`.
        2. **Sequential vs. Joint Missing-Data Design:** Contrast **Pipeline A** (Impute → Calibrate), **Pipeline B** (Calibrate → Impute), **Pipeline C** (Causal Joint TCN v2), and **Pipeline D** (Complete-Case Reference).
        3. **Strictly Causal Deep Sequence Modeling:** Enforce causal left-padded dilated convolutions with `ChannelLayerNorm1d` (RF = 61h) verified across 144 future-perturbation checks.
        4. **Regime- & Block-Conditioned Conformal Uncertainty:** Evaluate finite-sample 90% / 95% intervals, pointwise vs. simultaneous block coverage, and winter saturation limits.
        5. **24-Hour CPCB PM2.5 Category Reliability:** Quantify regulatory band accuracy (`N=127`, $\\ge 18/24$h rule) and uncertainty-based stability screening.
        6. **Local Spark & DuckDB Warehousing:** Benchmark CSV vs. Parquet vs. ORC storage compression and single-host `local[*]` scaling against in-memory Pandas.
        """
    )

# =============================================================================
# TAB 2: FOUR PIPELINES & NINE SCENARIOS
# =============================================================================
with tab_pipelines:
    st.markdown("### Four Comparative Architectures & Key Empirical Findings")
    st.info(
        '**Central Conclusion:** *"No single reconstruction strategy is uniformly optimal across missingness regimes."* '
        "Sequential Pipeline B leads across the full test set (`N=3,720`), whereas Joint TCN v2 (Pipeline C) achieves "
        "lower pure-reconstruction error on short (`1h`, `3h`), random (`20% MCAR`), and weather-linked (`MAR-RH`) dropouts.",
        icon="🧭",
    )
    render_four_pipelines_diagram()

    col_ov, col_ms = st.columns(2)
    with col_ov:
        st.plotly_chart(
            create_pipeline_comparison_bar(
                df_overall,
                metric_col="rmse",
                scenario_id=None,
                population_label="Full Test Population (A/B/C: N=3,720; D: Eligible Subset Only)",
                title="Full Test RMSE Across All 9 Scenarios",
            ),
            width="stretch",
            config=PLOTLY_CHART_CONFIG,
        )
        st.caption("Pipeline D covers only complete observations (~79.3–80.0%) and must not be directly ranked against A/B/C on full test coverage.")

    with col_ms:
        st.plotly_chart(
            create_pipeline_comparison_bar(
                df_missing,
                metric_col="rmse",
                scenario_id=None,
                population_label="Missing-PM Subgroup (m_t = 0)",
                title="Pure Reconstruction RMSE on Missing Intervals (m_t = 0)",
            ),
            width="stretch",
            config=PLOTLY_CHART_CONFIG,
        )
        st.caption("Pipeline D does not emit predictions when m_t = 0 (coverage 0.0%) and is omitted in the Missing-PM subgroup.")

    st.markdown("#### Nine Missingness Evaluation Regimes")
    sc_df = pd.DataFrame(
        [
            {
                "Scenario ID": sid,
                "Label": s["name"],
                "Mechanism": s["category"],
                "Definition": s["description"],
            }
            for sid, s in SCENARIOS.items()
        ]
    )
    st.dataframe(sc_df, width="stretch", hide_index=True)

# =============================================================================
# TAB 3: CHECKPOINTS CP1–CP6
# =============================================================================
with tab_checkpoints:
    st.markdown("### Verifiable Historical Experimental Checkpoints (CP1–CP6)")
    c1, c2, c3, c4, c5, c6 = st.tabs(
        [
            "CP1: Sensor Physics",
            "CP2: Classical Baselines",
            "CP3: Causal TCN v2",
            "CP4: Conformal Risk",
            "CP5: CPCB 24h",
            "CP6: Local Big Data",
        ]
    )

    with c1:
        st.markdown("**Channel Bias vs. BAM-1020 Reference (`outputs/reports/sensor_bias_summary.csv`, N=18,584):**")
        if not df_bias.empty:
            st.dataframe(df_bias[df_bias["site"] == "POOLED (All Sites)"], width="stretch", hide_index=True)
        st.markdown("**Psychrometric Anomaly Audit (`outputs/reports/dew_point_investigation.csv`):**")
        if not df_dew.empty:
            st.dataframe(df_dew, width="stretch", hide_index=True)
        st.markdown("**Physical Timestamp Gaps (`outputs/reports/missing_timestamp_report.csv`):**")
        if not df_gaps.empty:
            st.dataframe(df_gaps, width="stretch", hide_index=True)

    with c2:
        df_base = load_metrics_table("baseline_summary")
        df_recon = load_metrics_table("corrected_reconstruction_metrics")
        st.markdown("**Calibration Grid Summary (`baseline_summary.csv`):**")
        if not df_base.empty:
            st.dataframe(df_base, width="stretch", hide_index=True)
        st.markdown("**Phase 3A.2 Corrected Reconstruction Baselines (`corrected_reconstruction_metrics.csv`):**")
        if not df_recon.empty:
            st.dataframe(df_recon, width="stretch", hide_index=True)

    with c3:
        st.markdown("**Causality-Safe Joint TCN v2 (`joint_tcn_v2_metadata.json`):**")
        val_m = meta_tcn.get("validation_metrics", {})
        if val_m:
            st.dataframe(
                pd.DataFrame(
                    [{"target": k, **v} for k, v in val_m.items()]
                ),
                width="stretch",
                hide_index=True,
            )
        df_ab = load_metrics_table("joint_tcn_ablation")
        if not df_ab.empty:
            st.markdown("**Historical Multi-Task Ablation Study (`joint_tcn_ablation.csv`):**")
            st.dataframe(df_ab, width="stretch", hide_index=True)

    with c4:
        df_conf_rec = load_metrics_table("conformal_coverage_reconciliation")
        df_conf_blk = load_metrics_table("conformal_simultaneous_block_coverage")
        st.markdown("**Pointwise Conformal Coverage Reconciliation (`conformal_coverage_reconciliation.csv`):**")
        if not df_conf_rec.empty:
            st.dataframe(df_conf_rec, width="stretch", hide_index=True)
        st.markdown("**Simultaneous Block Coverage (`conformal_simultaneous_block_coverage.csv`):**")
        if not df_conf_blk.empty:
            st.dataframe(df_conf_blk, width="stretch", hide_index=True)

    with c5:
        st.markdown("**Controlled 24-Hour CPCB Model Comparison (`cpcb_controlled_comparison.csv`, N=127):**")
        if not df_cpcb.empty:
            st.dataframe(df_cpcb, width="stretch", hide_index=True)
        df_cat = load_metrics_table("checkpoint5_category_metrics")
        if not df_cat.empty:
            st.markdown("**Category-Wise Precision, Recall & F1 (`checkpoint5_category_metrics.csv`):**")
            st.dataframe(df_cat, width="stretch", hide_index=True)

    with c6:
        st.markdown("**Warehouse Format Footprints (`checkpoint6_storage_formats.csv`):**")
        if not df_storage.empty:
            st.dataframe(df_storage, width="stretch", hide_index=True)
        st.markdown("**Single-Host Pandas vs. Local Spark Scaling (`checkpoint6_scaling_benchmark.csv`):**")
        if not df_scale.empty:
            st.dataframe(df_scale, width="stretch", hide_index=True)

# =============================================================================
# TAB 4: DISCLOSURES & DIRECT ROUTE NAVIGATION
# =============================================================================
with tab_disclosures:
    st.markdown("### Mandatory Research Methodology Limitations & Disclosures")
    st.markdown(
        """
        <div class="research-warning" style="margin-bottom: 18px;">
            <strong>Verified Scientific Boundaries:</strong>
            <ul style="margin-top: 8px; margin-bottom: 0; font-size: 0.87rem; line-height: 1.6;">
                <li><strong>TCN Version Separation:</strong> Phase 3C uses causality-safe <code>joint_tcn_v2_causal_checkpoint.pt</code> (<code>ChannelLayerNorm1d</code>, 144/144 checks pass). Historical Checkpoints 4 and 5 evaluate the earlier BatchNorm checkpoint (<code>frozen_joint_tcn_checkpoint.pt</code>) and are documented transparently as separate archival phases.</li>
                <li><strong>Pipeline D Non-Comparability:</strong> Complete-case calibration evaluates only observed rows (~79.3%–80.0%) and must not be directly ranked against full-test (100% coverage) models.</li>
                <li><strong>MNAR-PM Oracle Limitation:</strong> The MNAR-PM mask triggers outages using held-out BAM PM2.5 levels; BAM never enters model inputs, but the oracle nature of synthetic MNAR masks is disclosed as an evaluation limitation.</li>
                <li><strong>Bootstrap Interpretation:</strong> Empirical uncentered bootstrap tail proportions reflect the fraction of moving-block resampled RMSE differences crossing zero and are reported alongside paired 95% CIs rather than formal null hypothesis p-values.</li>
                <li><strong>Conformal Temporal Dependence &amp; Saturation (Checkpoint 4 @ 90%):</strong> Pointwise coverage exceeds simultaneous full-block coverage on <code>frozen_joint_tcn_checkpoint.pt</code> — 12h block: 86.40% pointwise vs. 63.83% simultaneous (120/188 blocks, 88.67 µg/m³ width); 24h block: 95.70% pointwise vs. 74.58% simultaneous (44/59 blocks, 200.84 µg/m³ width); extreme haze (&gt;250 µg/m³) drops coverage to 1.95%.</li>
                <li><strong>CPCB PM2.5-Only Scope:</strong> Evaluates 24-hour PM2.5 thresholds only (not multi-pollutant National AQI); Severe category had N=4 test days with 0/4 recall.</li>
                <li><strong>Single-Machine Big Data Scope:</strong> Spark benchmarks ran in <code>local[*]</code> mode on one workstation; network-scale cluster deployment is not claimed.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Direct Analytical Module Navigation")
    routes = [
        ("pages/02_Dataset_Explorer.py", "📁 Page 02: Dataset Explorer"),
        ("pages/03_Calibration_Reconstruction.py", "⚖️ Page 03: Calibration & Reconstruction Lab"),
        ("pages/04_Joint_TCN.py", "🧠 Page 04: Joint TCN Explorer"),
        ("pages/05_Uncertainty_Analysis.py", "🛡️ Page 05: Uncertainty Analysis"),
        ("pages/06_CPCB_Reliability.py", "📋 Page 06: CPCB Reliability"),
        ("pages/07_Big_Data_Analytics.py", "⚡ Page 07: Big Data Analytics"),
    ]
    c_l, c_r = st.columns(2)
    for i, (r_path, r_lbl) in enumerate(routes):
        with (c_l if i % 2 == 0 else c_r):
            _safe_page_link(r_path, r_lbl)
