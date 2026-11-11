"""
AIRWISE — PM2.5 Research Analytics Platform
Main Entry Point, End-to-End Methodology Landing Page & Navigation Hub
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd
import streamlit as st

from dashboard.config import (
    APP_NAME,
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
    render_conformal_split_diagram,
    render_four_pipelines_diagram,
)
from dashboard.services.data_service import (
    load_phase3c_metrics,
    load_joint_tcn_v2_metadata,
    load_metrics_table,
)

# Page Configuration
st.set_page_config(
    page_title="AIRWISE — PM2.5 Research Analytics Platform",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Persistent Sidebar
filters = render_benchmark_sidebar(
    show_site_filter=True,
    show_date_filter=False,
    show_scenario_filter=False,
    show_pipeline_filter=False,
    show_population_filter=False,
    show_metric_filter=False,
    active_page="Home / Research Portal",
)

# Inject Dark Theme CSS
css_path = DASHBOARD_DIR / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Verifiable Research Summaries
df_overall = load_phase3c_metrics("overall")
df_missing = load_phase3c_metrics("missing_pm")
meta_tcn = load_joint_tcn_v2_metadata()
cpcb_ctrl = load_metrics_table("cpcb_controlled_comparison")
scale_df = load_metrics_table("checkpoint6_scaling")

b_overall = df_overall[df_overall["pipeline_code"] == "B"] if not df_overall.empty else pd.DataFrame()
b_rmse_min = float(b_overall["rmse"].min()) if not b_overall.empty else 36.26
b_rmse_max = float(b_overall["rmse"].max()) if not b_overall.empty else 41.64

c_mar_rh_row = (
    df_missing[(df_missing["pipeline_code"] == "C") & (df_missing["scenario_id"] == "MAR_RH")]
    if not df_missing.empty
    else pd.DataFrame()
)
c_mar_rh = float(c_mar_rh_row["rmse"].iloc[0]) if not c_mar_rh_row.empty else 43.05

best_epoch = int(meta_tcn.get("training_summary", {}).get("best_epoch", 6))
total_checks = int(meta_tcn.get("causality_verification", {}).get("total_checks", 144))

cpcb_tcn_row = (
    cpcb_ctrl[cpcb_ctrl["model_name"].str.contains("Joint TCN", na=False)]
    if not cpcb_ctrl.empty
    else pd.DataFrame()
)
cpcb_acc = float(cpcb_tcn_row["accuracy_pct"].iloc[0]) if not cpcb_tcn_row.empty else 72.44
cpcb_kappa = float(cpcb_tcn_row["cohen_kappa_quadratic"].iloc[0]) if not cpcb_tcn_row.empty else 0.9271

peak_spark_rps = float(scale_df["spark_throughput_rows_sec"].max()) if not scale_df.empty else 961275.2

# Hero Header
st.markdown(
    f"""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size: 2.35rem; font-weight: 800; color: {THEME['accent_teal']}; letter-spacing: -1px; margin-bottom: 4px;">
            🌬️ {APP_NAME}
        </div>
        <div style="font-size: 1.2rem; font-weight: 600; color: {THEME['text_primary']}; margin-bottom: 6px;">
            {APP_TITLE}
        </div>
        <div style="font-size: 0.94rem; color: {THEME['text_secondary']}; max-width: 960px; line-height: 1.55;">
            {APP_SUBTITLE} — Assessing Sequential versus Joint Deep Architectures across Nine Missingness Regimes in Urban and Rural India.
        </div>
        <div style="font-size: 0.80rem; color: {THEME['accent_cyan']}; margin-top: 6px; font-weight: 500;">
            <strong>Reference Dataset:</strong> {ACADEMIC_CITATION}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Executive KPI Row
render_kpi_row(
    [
        {
            "title": "Curated Observations",
            "value": f"{DATASET_SPLITS['total']:,}",
            "subtitle": "Delhi, Bangalore & Hamirpur",
            "delta": f"Test Split: N = {DATASET_SPLITS['test']:,} (20%)",
            "color": THEME["accent_teal"],
        },
        {
            "title": "Pipeline B (Full Test)",
            "value": f"{b_rmse_min:.2f}–{b_rmse_max:.2f}",
            "subtitle": "RMSE (µg/m³) across 9 Scenarios",
            "delta": "Calibrate → Impute (100% Coverage)",
            "color": THEME["accent_cyan"],
        },
        {
            "title": "Causal Joint TCN v2",
            "value": f"{c_mar_rh:.2f} µg/m³",
            "subtitle": "Lowest MAR-RH Missing-PM RMSE",
            "delta": f"Epoch {best_epoch} | {total_checks}/{total_checks} Causality Checks",
            "color": THEME["accent_purple"],
        },
        {
            "title": "24h CPCB PM2.5 Reliability",
            "value": f"{cpcb_acc:.2f}%",
            "subtitle": f"N = 127 Windows | Quad κ = {cpcb_kappa:.4f}",
            "delta": "PM2.5 Only (Not Full NAQI)",
            "color": THEME["accent_teal"],
        },
        {
            "title": "Local Spark Throughput",
            "value": f"{peak_spark_rps / 1e3:.1f}k rows/s",
            "subtitle": "50× Scale (local[8] Workstation)",
            "delta": "Parquet -73.0% | ORC -84.1%",
            "color": THEME["accent_amber"],
        },
    ]
)

st.markdown("---")

# End-to-End Research Workflow
st.subheader("🔄 End-to-End AIRWISE Research Architecture (7 Stages)")
render_research_workflow_diagram()

st.markdown("")

# Chronological Split & Central Research Finding
col_split, col_finding = st.columns([1.25, 0.95])
with col_split:
    st.subheader("📊 Authoritative 4-Way Chronological Partition")
    render_conformal_split_diagram()

with col_finding:
    st.subheader("🧭 Central Phase 3C Research Finding")
    st.info(
        '**Core Thesis:** *"No single reconstruction strategy is uniformly optimal across missingness regimes."*',
        icon="🔬",
    )
    st.markdown(
        f"""
        - **Full Test Population (`N = 3,720`):** Sequential **Pipeline B (Calibrate → Impute)** achieves the lowest overall RMSE across all 9 scenarios (`{b_rmse_min:.2f}–{b_rmse_max:.2f} µg/m³`).
        - **Missing-PM Subgroup (`m_t = 0`):** **Joint Causal TCN v2 (Pipeline C)** outperforms sequential pipelines on short/weather-driven gaps (`MAR-RH: {c_mar_rh:.2f} µg/m³`, `20% MCAR: 54.87`, `Block 1h: 52.44`, `Block 3h: 53.04`), but compounds on extended multi-day outages (`Block 24h` nominal up to `63h` merged: `81.86 µg/m³`).
        - **Complete-Case Reference (`Pipeline D`):** Evaluated strictly on observed complete cases (~`79.3%–80.0%` coverage) as a no-imputation ceiling; **never** ranked directly against `100%`-coverage models.
        """
    )

st.markdown("---")

# Four Pipelines & Nine Scenarios
st.subheader("🔀 Four Comparative Pipelines (Phase 3C)")
render_four_pipelines_diagram()

with st.expander("Explore the Nine Standardized Missingness Scenarios", expanded=False):
    sc_rows = [
        {
            "Scenario ID": sid,
            "Display Name": info["name"],
            "Mechanism Class": info["category"],
            "Protocol Description": info["description"],
        }
        for sid, info in SCENARIOS.items()
    ]
    st.dataframe(pd.DataFrame(sc_rows), width="stretch", hide_index=True)

st.markdown("---")

# Module Navigation Hub
st.subheader("🧭 Analytical Platform Modules (Click Any Module to Navigate)")

nav_modules = [
    (
        "pages/01_Research_Overview.py",
        "🔬 Module 01 • Research Overview",
        "Architecture, Splits & Evidence",
        "Complete end-to-end methodology, per-site chronological split bounds, 9-scenario protocol, and reconciled Checkpoint 1–6 & Phase 3C findings.",
        THEME["accent_teal"],
    ),
    (
        "pages/02_Dataset_Explorer.py",
        "📁 Module 02 • Dataset Explorer",
        "Raw Telemetry & Missingness Profiles",
        "Independent PM-channel availability, partial vs. complete PM vs. station outages, missingness heatmaps, distributions, correlations & CSV export.",
        THEME["accent_cyan"],
    ),
    (
        "pages/03_Calibration_Reconstruction.py",
        "⚖️ Module 03 • Calibration & Reconstruction Lab",
        "4-Pipeline × 9-Scenario Benchmark",
        "Side-by-side Pipelines A, B, C, and D across isolated Full, Missing-PM, and Common-Support populations with moving-block bootstrap CIs.",
        THEME["accent_teal"],
    ),
    (
        "pages/04_Joint_TCN.py",
        "🧠 Module 04 • Joint TCN Explorer",
        "Causal v2 Architecture & Outage Demo",
        "ChannelLayerNorm1d schema, 144 causality checks, 16-epoch multi-task loss curves, outage degradation, and frozen Epoch-6 synthetic stress demo.",
        THEME["accent_purple"],
    ),
    (
        "pages/05_Uncertainty_Analysis.py",
        "🛡️ Module 05 • Conformal Uncertainty",
        "90% / 95% Intervals & Block Coverage",
        "Site-wise coverage disparities (Delhi 51.07% vs Bangalore 97.58%), temporal block calibration, pointwise vs. simultaneous coverage & saturation failure.",
        THEME["accent_amber"],
    ),
    (
        "pages/06_CPCB_Reliability.py",
        "📋 Module 06 • CPCB Reliability",
        "24h PM2.5 Categories & Stability",
        "Interactive 6-class CPCB confusion matrix, category precision/recall/F1 (Severe 0/4 recall), 75% completeness rule, and stability screening.",
        THEME["accent_cyan"],
    ),
    (
        "pages/07_Big_Data_Analytics.py",
        "⚡ Module 07 • Big Data Analytics",
        "DuckDB Warehouse & Local Spark",
        "CSV vs. Parquet (-73.0%) vs. ORC (-84.1%), local[*] Spark scaling benchmarks (1×–50×), partition pruning, caching, and guarded read-only SQL.",
        THEME["accent_purple"],
    ),
]

for row_start in range(0, len(nav_modules), 2):
    cols = st.columns(2)
    for offset, col in enumerate(cols):
        idx = row_start + offset
        if idx >= len(nav_modules):
            break
        route, badge, title, desc, color = nav_modules[idx]
        with col:
            st.markdown(
                f"""
                <div class="airwise-card" style="border-top: 3px solid {color}; min-height: 138px;">
                    <div class="airwise-card-title" style="color: {color};">{badge}</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: {THEME['text_primary']}; margin-bottom: 6px;">{title}</div>
                    <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.48;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.page_link(route, label=f"Open {title} →", width="stretch")

st.markdown("---")

# Mandatory Scientific Disclosures
with st.expander("🔬 Authoritative Scientific Disclosures & Methodological Boundaries", expanded=True):
    st.markdown(
        """
        - **Causal TCN v2 vs. Historical Conformal Checkpoints:** Phase 3C benchmark uses causality-safe `joint_tcn_v2_causal_checkpoint.pt` (`ChannelLayerNorm1d`, 144/144 checks passed). Earlier Checkpoint 4 conformal and Checkpoint 5 CPCB runs remain pinned to the frozen historical BatchNorm checkpoint (`frozen_joint_tcn_checkpoint.pt`) and are never extrapolated across experiments.
        - **No Direct Ranking of Complete-Case Pipeline D:** Pipeline D evaluates only complete observations (`m_t = 1`); comparing its error directly to `100%`-coverage models conflates selection bias with predictive accuracy.
        - **Pointwise vs. Simultaneous Block Conformal Coverage (Checkpoint 4 @ 90%):** Under `frozen_joint_tcn_checkpoint.pt`, 12h block conformal achieves `86.40%` pointwise (`88.67 µg/m³` width) vs. `63.83%` simultaneous (`120/188` blocks), while 24h block conformal achieves `95.70%` pointwise (`200.84 µg/m³` width) vs. `74.58%` simultaneous (`44/59` blocks).
        - **CPCB Regulatory Scope:** Evaluates 24h PM2.5 concentration bands (`CUPS/82/2014-15`), not full multi-pollutant National AQI (`NAQI`). Only 4 test windows fall in Severe (`>250 µg/m³`), with `0/4` recall.
        - **Local Spark Scope:** Apache Spark benchmarks execute in single-host local mode (`local[*]`) rather than a multi-node network cluster; DuckDB provides read-only local Hive-view querying.
        - **Dataset Stationarity & Humidity Confounding:** High-RH vs. Low-RH error differences are confounded by monitoring city and season (Bangalore monsoon vs. Delhi winter stagnation).
        """
    )
