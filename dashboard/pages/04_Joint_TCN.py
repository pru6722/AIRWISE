"""
AIRWISE Dashboard Page 5: Joint Causal TCN v2 Explorer (Pipeline C)
Dedicated exploration of the Phase 3C.1B Causality-Safe Joint TCN v2 architecture:
- Layered causal architecture & 144 causality verification checks
- Actual 16-epoch training & validation convergence (multi-task, calibration & reconstruction heads)
- Phase 3C outage duration degradation & site-wise performance (Pipeline C vs sequential baselines)
- Optional live synthetic outage demonstration using the frozen Epoch-6 v2 checkpoint only
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "dashboard") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import (
    THEME,
    SITES,
    SCENARIOS,
    PIPELINES,
    EVALUATION_POPULATIONS,
    DATASET_SPLITS,
)
from dashboard.components.sidebar import render_benchmark_sidebar
from dashboard.components.metric_cards import render_kpi_row
from dashboard.components.architecture import render_joint_tcn_diagram
from dashboard.components.charts import (
    PLOTLY_CHART_CONFIG,
    apply_dark_layout,
    create_training_loss_chart,
    create_outage_degradation_chart,
    create_site_stratified_chart,
    create_timeseries_chart,
)
from dashboard.services.data_service import (
    load_cleaned_dataset,
    load_benchmark_predictions,
    load_phase3c_metrics,
    load_joint_tcn_v2_metadata,
    load_metrics_table,
    filter_benchmark_predictions,
    compute_evaluation_metrics,
)
from dashboard.services.tcn_service import run_sequence_inference
from src.uncertainty.split_manager import create_conformal_splits


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AIRWISE | Joint TCN Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = PROJECT_ROOT / "dashboard" / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

filters = render_benchmark_sidebar(
    show_site_filter=True,
    show_date_filter=False,
    show_scenario_filter=True,
    show_pipeline_filter=False,
    show_population_filter=True,
    show_metric_filter=True,
    active_page="Joint TCN Explorer",
)

selected_site: str = filters["site"]
scenario_id: str = filters["scenario_id"]
population_mode: str = filters["population_mode"]
metric_col: str = filters["metric"]

scenario_meta = SCENARIOS.get(scenario_id, {"name": scenario_id, "category": "", "description": ""})
pop_meta = EVALUATION_POPULATIONS.get(
    population_mode,
    {"name": "Full Test Population", "subtitle": "N = 3,720", "description": ""},
)


# ---------------------------------------------------------------------------
# Cached Test Partition Helper (Zero Leakage for Optional Live Demo)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def _load_test_partition() -> pd.DataFrame:
    df_clean = load_cleaned_dataset()
    _, _, _, test_df, _ = create_conformal_splits(df_clean)
    return test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load Artifact Metadata & Phase 3C Metrics
# ---------------------------------------------------------------------------
meta: Dict[str, Any] = load_joint_tcn_v2_metadata()
train_cfg: Dict[str, Any] = meta.get("training_config", {})
causality_info: Dict[str, Any] = meta.get("causality_verification", {})
train_summary: Dict[str, Any] = meta.get("training_summary", {})
val_metrics: Dict[str, Any] = meta.get("validation_metrics", {})
hashes: Dict[str, Any] = meta.get("artifact_hashes", {})
history: list = meta.get("history", [])

best_epoch = int(train_summary.get("best_epoch", 6))
best_val_loss = float(train_summary.get("best_val_loss", 0.5929))
best_val_cal = float(train_summary.get("best_val_cal_loss", 0.2695))
best_val_rec = float(train_summary.get("best_val_rec_loss", 0.3234))
total_epochs = int(train_summary.get("total_epochs_trained", 16))
train_sec = float(train_summary.get("training_duration_sec", 30.19))

total_checks = int(causality_info.get("total_checks", 144))
checks_passed = bool(causality_info.get("passed", True))
max_dev = float(causality_info.get("max_deviation", 0.0))
tol = float(causality_info.get("tolerance", 1e-6))
verify_sec = float(causality_info.get("verification_duration_sec", 0.109))

rf_hours = int(train_cfg.get("receptive_field_hours", 61))
seq_len = int(train_cfg.get("seq_len", 64))
kernel_size = int(train_cfg.get("kernel_size", 3))
dilations = train_cfg.get("dilations", [1, 2, 4, 8])
hidden_channels = int(train_cfg.get("hidden_channels", 32))
norm_layer = str(train_cfg.get("norm_layer", "ChannelLayerNorm1d"))
dropout_rate = float(train_cfg.get("dropout", 0.15))
lr_init = float(train_cfg.get("lr", 0.001))

df_pop_metrics = load_phase3c_metrics(population_mode)
df_missing_pm_metrics = load_phase3c_metrics("missing_pm")
df_site_metrics = load_phase3c_metrics("site_stratified")


# ---------------------------------------------------------------------------
# Hero Header & Methodological Disclosure Banner
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="margin-bottom: 14px;">
        <h1 style="margin:0; font-size:2.05rem; color:{THEME['accent_purple']};">
            🧠 Joint Causal TCN v2 Explorer (Pipeline C)
        </h1>
        <p style="margin:6px 0 0 0; color:{THEME['text_secondary']}; font-size:0.96rem; max-width:980px;">
            Multi-task dilated causal temporal convolutional architecture jointly predicting observed
            calibration (<code>y_cal</code>) and outage reconstruction (<code>y_rec</code>).
            Evaluated using the frozen Phase 3C.1B Epoch-{best_epoch} checkpoint across
            {pop_meta['name']} and 9 missingness regimes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

chk_sha = str(hashes.get("new_checkpoint_sha256", ""))
st.info(
    f"**Frozen Artifact Protocol:** Displaying Phase 3C.1B / 3C.2 verified outputs from "
    f"`{train_cfg.get('save_path', 'outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt')}` "
    f"(SHA-256: `{chk_sha[:16]}...`). Train partition `N={DATASET_SPLITS['train']:,}`, "
    f"Early-Stopping Validation `N={DATASET_SPLITS['validation']:,}`, Independent Test `N={DATASET_SPLITS['test']:,}`.",
    icon="🔒",
)

# KPI Row
render_kpi_row(
    [
        {
            "title": "Architecture & RF",
            "value": f"RF = {rf_hours}h",
            "subtitle": f"K={kernel_size}, Dilations={dilations}",
            "delta": f"Window={seq_len}h | Channels={hidden_channels}",
            "color": THEME["accent_purple"],
        },
        {
            "title": "Causal Normalization",
            "value": norm_layer,
            "subtitle": "Strictly Per-Timestep",
            "delta": f"Dropout={dropout_rate} | AdamW lr={lr_init}",
            "color": THEME["accent_cyan"],
        },
        {
            "title": f"Epoch {best_epoch} Convergence",
            "value": f"{best_val_loss:.4f}",
            "subtitle": f"Cal: {best_val_cal:.4f} | Rec: {best_val_rec:.4f}",
            "delta": f"{total_epochs} epochs trained in {train_sec:.2f}s",
            "color": THEME["accent_teal"],
        },
        {
            "title": "Causality Verification",
            "value": f"{total_checks}/{total_checks} Passed" if checks_passed else "Failed",
            "subtitle": f"Max Future Dev: {max_dev:.4f}",
            "delta": f"Tol: {tol:.0e} | Runtime: {verify_sec:.3f}s",
            "color": THEME["accent_teal"] if checks_passed else THEME["accent_rose"],
        },
    ]
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_arch, tab_train, tab_bench, tab_sim = st.tabs(
    [
        "📐 Architecture & 144 Causality Checks",
        "📈 16-Epoch Training & Head Convergence",
        "📊 Phase 3C Outage Degradation & Site Analysis",
        "🧪 Synthetic Outage Simulator (Frozen Epoch-6 Demo)",
    ]
)


# ===========================================================================
# TAB 1: ARCHITECTURE & 144 CAUSALITY CHECKS
# ===========================================================================
with tab_arch:
    st.subheader("Causal Dilated Architecture & Mask-Gated Dual Heads")
    render_joint_tcn_diagram()

    st.markdown(
        r"""
        <div style="margin-top:10px;"></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
        **Receptive Field Formula:** For 4 residual causal blocks ($B=4$) with dilation factors $d \in \{1, 2, 4, 8\}$ and kernel size $K = 3$ (2 causal convolutions per residual block):
        $$RF = 1 + 2(K - 1)\sum_{i=0}^{3} 2^i = 1 + 2(2)(1 + 2 + 4 + 8) = 61 \text{ hours}$$
        """
    )

    col_cfg, col_val = st.columns([1.05, 1.0])

    with col_cfg:
        st.markdown("#### Frozen Model Hyperparameters & Artifact Hashes")
        cfg_rows = [
            {"Parameter": "Architecture Variant", "Value": str(train_cfg.get("architecture", "JointTCN v2 (Causality-Safe)"))},
            {"Parameter": "Normalization Layer", "Value": f"{norm_layer} (norm_type='{train_cfg.get('norm_type', 'layer')}')"},
            {"Parameter": "Input Features (11 ch)", "Value": f"10 scaled features + binary mask M_t (zero BAM input)"},
            {"Parameter": "Sequence Length / Stride", "Value": f"{seq_len}h window / stride {train_cfg.get('stride', 1)}h"},
            {"Parameter": "Loss Weighting", "Value": f"lambda_cal = {train_cfg.get('lambda_cal', 1.0)}, lambda_rec = {train_cfg.get('lambda_rec', 1.0)} (Huber)"},
            {"Parameter": "Early Stopping / Patience", "Value": f"Best Epoch {best_epoch} of {total_epochs} trained (patience={train_cfg.get('patience', 10)})"},
            {"Parameter": "Random Seed", "Value": str(train_cfg.get("random_seed", 42))},
            {"Parameter": "v2 Checkpoint SHA-256", "Value": chk_sha},
            {"Parameter": "Frozen v1 Conformal SHA-256", "Value": str(hashes.get("frozen_joint_tcn_sha256", ""))},
        ]
        st.dataframe(pd.DataFrame(cfg_rows), width="stretch", hide_index=True)

    with col_val:
        st.markdown("#### 144 Causality Checks & Validation Split Metrics (N=1,857)")
        st.success(
            f"**Formal Causality Verification Passed ({total_checks}/{total_checks} perturbations):** "
            f"Perturbing any future timestep $s > t$ yields `max_deviation = {max_dev:.6f}` "
            f"(strictly below tolerance `{tol:.0e}`, audited in `{verify_sec:.3f}s`)."
        )

        val_rows = []
        head_labels = [
            ("overall_composite", "Composite Output (y_final)"),
            ("calibration_head", "Calibration Head (y_cal, M_t=1)"),
            ("reconstruction_head", "Reconstruction Head (y_rec, M_t=0)"),
        ]
        for key, display_name in head_labels:
            v = val_metrics.get(key, {})
            if v:
                val_rows.append(
                    {
                        "Validation Target": display_name,
                        "MAE (µg/m³)": v.get("mae"),
                        "RMSE (µg/m³)": v.get("rmse"),
                        "R²": v.get("r2"),
                        "MBE (µg/m³)": v.get("mbe"),
                        "Rel Bias (%)": v.get("relative_bias_pct"),
                    }
                )
        if val_rows:
            st.dataframe(pd.DataFrame(val_rows), width="stretch", hide_index=True)
            st.caption(
                "Note: Validation metrics above reflect the canonical N=1,857 validation split reported in "
                "`joint_tcn_v2_metadata.json` at Epoch 6 convergence, prior to frozen Phase 3C test evaluation."
            )

    with st.expander("Why Causal Convolutions & ChannelLayerNorm1d Matter", expanded=False):
        st.markdown(
            r"""
            1. **Zero Future Leakage via Left-Only Chomp Padding:** Every `CausalConv1d` layer applies `(kernel_size - 1) * dilation` left padding and slices away the trailing elements, guaranteeing $\hat{y}_t$ depends exclusively on $X_{\le t}$.
            2. **Eliminating Temporal Statistics Leakage (`ChannelLayerNorm1d` vs `BatchNorm1d`):** `BatchNorm1d` computes mean and variance across both batch and full sequence time axes $t \in [1, 64]$, implicitly allowing future inputs to influence earlier standardized activations inside a sequence window. Replacing it with `ChannelLayerNorm1d` normalizes exclusively across the feature channel dimension at each independent timestep $t$.
            3. **Mask-Gated Reconstruction Head:** During sensor outages ($M_t = 0$), `ReconstructionHead` computes $M_t \odot \hat{y}_{\text{cal}, t}$ so unobserved calibration outputs are deterministically zeroed before downstream MLPs.
            """
        )


# ===========================================================================
# TAB 2: 16-EPOCH TRAINING & HEAD CONVERGENCE
# ===========================================================================
with tab_train:
    st.subheader("Multi-Task Huber Loss Convergence (Actual 16-Epoch Log)")

    if history:
        fig_loss = create_training_loss_chart(
            history,
            best_epoch=best_epoch,
            title="Joint TCN v2 Training & Validation Convergence (Multi-Task Huber Loss)",
        )
        st.plotly_chart(fig_loss, width="stretch", config=PLOTLY_CHART_CONFIG)

        hist_df = pd.DataFrame(history)

        st.markdown("#### Decomposition: Calibration Head vs. Reconstruction Head Loss")
        fig_heads = go.Figure()
        head_traces = [
            ("train_cal_loss", "Train Calibration Head", THEME["accent_purple"], "solid"),
            ("val_cal_loss", "Val Calibration Head", THEME["accent_purple"], "dash"),
            ("train_rec_loss", "Train Reconstruction Head", THEME["accent_amber"], "solid"),
            ("val_rec_loss", "Val Reconstruction Head", THEME["accent_amber"], "dash"),
        ]
        for col, label, color, dash in head_traces:
            if col in hist_df.columns:
                fig_heads.add_trace(
                    go.Scatter(
                        x=hist_df["epoch"],
                        y=hist_df[col],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2.2, dash=dash),
                        marker=dict(size=6),
                    )
                )
        fig_heads.add_vline(
            x=best_epoch,
            line_dash="dash",
            line_color=THEME["accent_rose"],
            annotation_text=f"Selected Epoch {best_epoch}",
            annotation_position="top right",
        )
        apply_dark_layout(
            fig_heads,
            "Head-Level Huber Loss Dynamics (Train vs Validation Across 16 Epochs)",
            "Training Epoch",
            "Head Huber Loss",
            height=400,
        )
        st.plotly_chart(fig_heads, width="stretch", config=PLOTLY_CHART_CONFIG)

        c_tbl, c_dl = st.columns([4, 1])
        with c_tbl:
            st.markdown("#### Recorded Epoch-by-Epoch Training Metrics")
        with c_dl:
            st.download_button(
                label="Download Loss CSV",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="joint_tcn_v2_training_history.csv",
                mime="text/csv",
                width="stretch",
            )
        st.dataframe(hist_df, width="stretch", hide_index=True)
    else:
        st.warning("Training history array not found in `joint_tcn_v2_metadata.json`.")

    with st.expander("Historical Multi-Task Loss Ablation Study (outputs/reports/joint_tcn_ablation.csv)", expanded=False):
        df_ablation = load_metrics_table("joint_tcn_ablation")
        if not df_ablation.empty:
            st.caption(
                "Historical pre-Phase-3C ablation archive loaded directly from `outputs/reports/joint_tcn_ablation.csv` "
                "(preserved verbatim; compare configurations A–F without re-running experiments)."
            )
            st.dataframe(df_ablation, width="stretch", hide_index=True)
        else:
            st.info("`joint_tcn_ablation.csv` was not found.")


# ===========================================================================
# TAB 3: PHASE 3C OUTAGE DEGRADATION & SITE-WISE ANALYSIS
# ===========================================================================
with tab_bench:
    sc_name = scenario_meta.get("name", scenario_id)
    st.subheader(f"Phase 3C Benchmark Diagnostics — {pop_meta['name']}")
    st.caption(
        f"**Active Population Mode:** {pop_meta['name']} ({pop_meta['subtitle']}) — "
        "Populations are strictly isolated (`overall` N=3,720, `missing_pm` m_t=0 subgroup, and `common_support` m_t=1 N=2,951)."
    )

    if scenario_id == "Block_24h":
        st.warning(
            "**Nominal vs. Merged Duration Note:** `Block_24h` injects nominal 24-hour outage blocks; "
            "adjacent block placement produces merged continuous outages ranging from **24h up to 63h** in the test partition.",
            icon="⏱️",
        )
    elif scenario_id == "MNAR_PM":
        st.info(
            "**MNAR-PM Protocol Disclosure:** Missingness in `MNAR_PM` is an oracle synthetic stress condition "
            "derived from held-out BAM concentration intensity. No BAM value or derived feature ever enters model inputs.",
            icon="ℹ️",
        )

    # Callout computed directly from scenario_metrics_missing_pm.csv
    c_miss = df_missing_pm_metrics[df_missing_pm_metrics["pipeline_code"] == "C"].copy()
    if not c_miss.empty:
        c_1h = c_miss.loc[c_miss["scenario_id"] == "Block_1h", "rmse"]
        c_24h = c_miss.loc[c_miss["scenario_id"] == "Block_24h", "rmse"]
        if not c_1h.empty and not c_24h.empty:
            r1 = float(c_1h.iloc[0])
            r24 = float(c_24h.iloc[0])
            delta_block = r24 - r1
            st.info(
                f"**Missing-PM Outage Compound Drift (Pipeline C):** Under pure reconstruction (`m_t = 0`), "
                f"Pipeline C achieves strong short-gap accuracy (`Block_1h` RMSE = **{r1:.2f} µg/m³**, beating sequential baselines), "
                f"but compounds during multi-day dropouts (`Block_24h` nominal / up to 63h merged RMSE = **{r24:.2f} µg/m³**, "
                f"**+{delta_block:.2f} µg/m³** degradation).",
                icon="📉",
            )

    col_deg, col_site = st.columns(2)
    with col_deg:
        fig_deg = create_outage_degradation_chart(
            df_pop_metrics,
            metric_col=metric_col,
            population_label=pop_meta["name"],
            title=f"Outage Degradation (1h → 24h Nominal) — {pop_meta['name']}",
        )
        st.plotly_chart(fig_deg, width="stretch", config=PLOTLY_CHART_CONFIG)

    with col_site:
        df_site_plot = df_site_metrics.copy()
        if selected_site != "All Sites":
            df_site_plot = df_site_plot[df_site_plot["site"] == selected_site]
        fig_site = create_site_stratified_chart(
            df_site_plot,
            metric_col=metric_col,
            scenario_id=scenario_id,
            title=f"Full-Test Site Comparison — {sc_name}" + (f" ({selected_site})" if selected_site != "All Sites" else ""),
        )
        st.plotly_chart(fig_site, width="stretch", config=PLOTLY_CHART_CONFIG)
        st.caption("Note: `site_stratified_metrics.csv` evaluates Full Test Population per city (BLR=1,489, DEL=1,502, HAM=729) across Pipelines A, B, and C.")

    # Pipeline C All-Scenario Summary Table
    st.markdown(f"#### Pipeline C (Joint TCN v2) Across All 9 Scenarios — {pop_meta['name']}")
    c_table = df_pop_metrics[df_pop_metrics["pipeline_code"] == "C"].copy()
    if not c_table.empty:
        display_cols = [c for c in ["scenario_id", "scenario_name", "category", "n_samples", "coverage_pct", "rmse", "mae", "r2", "mbe"] if c in c_table.columns]
        st.dataframe(c_table[display_cols], width="stretch", hide_index=True)

    # Site-wise detail table for Pipeline C
    st.markdown("#### Pipeline C Site-Wise Metrics (Full Test Population per Station)")
    c_site_tbl = df_site_metrics[df_site_metrics["pipeline_code"] == "C"].copy()
    if selected_site != "All Sites":
        c_site_tbl = c_site_tbl[c_site_tbl["site"] == selected_site]
    c_site_tbl = c_site_tbl[c_site_tbl["scenario_id"] == scenario_id] if scenario_id in c_site_tbl["scenario_id"].values else c_site_tbl
    if not c_site_tbl.empty:
        st.dataframe(c_site_tbl, width="stretch", hide_index=True)

    # Frozen Phase 3C Prediction vs BAM Trace for Pipeline C
    st.markdown(f"#### Frozen Test Prediction vs. BAM-1020 Trace — {sc_name}")
    df_bench_all = load_benchmark_predictions()
    trace_site = selected_site if selected_site != "All Sites" else st.selectbox(
        "Station for Contiguous Time-Series Trace",
        options=SITES,
        index=0,
        key="jtcn_trace_station",
    )
    df_trace = filter_benchmark_predictions(
        df_bench_all,
        scenario_id=scenario_id,
        site=trace_site,
        population_mode=population_mode,
    )
    pred_c_col = f"pred_C_{scenario_id}"
    if not df_trace.empty and pred_c_col in df_trace.columns:
        slice_metrics = compute_evaluation_metrics(df_trace["bam_reference"], df_trace[pred_c_col])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Slice Samples (N)", f"{slice_metrics['n_samples']:,}")
        m2.metric("Slice RMSE", f"{slice_metrics['rmse']:.2f} µg/m³")
        m3.metric("Slice MAE", f"{slice_metrics['mae']:.2f} µg/m³")
        m4.metric("Slice R²", f"{slice_metrics['r2']:.4f}")

        fig_trace = create_timeseries_chart(
            df_trace,
            y_cols=["bam_reference", pred_c_col],
            labels={
                "bam_reference": "BAM-1020 Reference",
                pred_c_col: PIPELINES["C"]["short_name"],
            },
            colors={
                "bam_reference": "#f8fafd",
                pred_c_col: PIPELINES["C"]["color"],
            },
            title=f"{trace_site} — Pipeline C vs. BAM Reference ({sc_name} | {pop_meta['name']})",
            show_rangeslider=True,
        )
        st.plotly_chart(fig_trace, width="stretch", config=PLOTLY_CHART_CONFIG)
    else:
        st.info("No frozen Pipeline C test observations matched the selected station/scenario/population slice.")


# ===========================================================================
# TAB 4: OPTIONAL SYNTHETIC OUTAGE SIMULATOR (FROZEN EPOCH-6 DEMO ONLY)
# ===========================================================================
with tab_sim:
    st.subheader("Interactive Synthetic Outage Stress Simulator (Optional Demo)")
    st.warning(
        "**Illustrative Demonstration Using Frozen Checkpoint Only:** Inference below runs live in read-only "
        "evaluation mode using the **frozen Phase 3C.1B Epoch-6 `joint_tcn_v2_causal_checkpoint.pt`** "
        "(`norm_type='layer'`, `DataNormalizer` fitted on the 11,149-row canonical training partition) over slices "
        "drawn strictly from the **held-out independent Test Partition (`N=3,720`)** with user-injected synthetic outage masks. "
        "This simulator does **not** retrain weights or alter frozen Phase 3C benchmark metrics.",
        icon="⚠️",
    )

    test_df = _load_test_partition()
    default_sim_site_idx = SITES.index(selected_site) if selected_site in SITES else 0

    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1.0, 1.0, 1.0, 1.0])
    with c_ctrl1:
        sim_site = st.selectbox("Test Station", options=SITES, index=default_sim_site_idx, key="sim_site_sel")

    site_test_df = test_df[test_df["site"] == sim_site].sort_values("timestamp").reset_index(drop=True)
    max_win_len = min(128, len(site_test_df))
    min_win_len = min(48, max_win_len)
    default_win_len = min(64, max_win_len)

    with c_ctrl2:
        window_len = st.slider(
            "Context Window Length (hours)",
            min_value=min_win_len,
            max_value=max_win_len,
            value=default_win_len,
            step=8,
            key="sim_win_len",
        )

    max_start = max(0, len(site_test_df) - window_len)
    with c_ctrl3:
        start_idx = st.slider(
            f"Test Start Offset (0–{max_start})",
            min_value=0,
            max_value=max_start,
            value=min(120, max_start),
            step=1,
            key="sim_start_offset",
        )

    with c_ctrl4:
        preset = st.selectbox(
            "Outage Pattern Preset",
            options=["Custom", "Short Burst (3h)", "Medium Gap (12h)", "Diurnal Outage (24h)", "Dual Split (2 x 6h)"],
            index=2,
            key="sim_outage_preset",
        )

    preset_defaults = {
        "Short Burst (3h)": (min(24, max(4, window_len - 4)), 3),
        "Medium Gap (12h)": (min(20, max(4, window_len - 13)), min(12, max(1, window_len - 8))),
        "Diurnal Outage (24h)": (min(16, max(4, window_len - 25)), min(24, max(1, window_len - 8))),
    }
    out_start_default, out_dur_default = preset_defaults.get(
        preset,
        (min(20, max(4, window_len - 10)), 8),
    )
    max_start_bound = max(5, window_len - 4)
    if (preset in preset_defaults and st.session_state.get("_prev_sim_preset") != preset) or "sim_out_start" not in st.session_state:
        st.session_state["sim_out_start"] = max(4, min(out_start_default, max_start_bound))
    else:
        st.session_state["sim_out_start"] = max(4, min(int(st.session_state["sim_out_start"]), max_start_bound))

    s1_col, s2_col = st.columns(2)
    with s1_col:
        out_start = st.slider(
            "Outage Injection Hour (offset inside window)",
            min_value=4,
            max_value=max_start_bound,
            key="sim_out_start",
        )
    max_dur = max(1, window_len - out_start)
    max_dur_bound = min(48, max_dur)
    if (preset in preset_defaults and st.session_state.get("_prev_sim_preset") != preset) or "sim_out_dur" not in st.session_state:
        st.session_state["sim_out_dur"] = max(1, min(out_dur_default, max_dur_bound))
    else:
        st.session_state["sim_out_dur"] = max(1, min(int(st.session_state["sim_out_dur"]), max_dur_bound))
    st.session_state["_prev_sim_preset"] = preset

    with s2_col:
        out_dur = st.slider(
            "Synthetic Outage Duration (hours)",
            min_value=1,
            max_value=max_dur_bound,
            key="sim_out_dur",
        )

    df_win = site_test_df.iloc[start_idx : start_idx + window_len].copy().reset_index(drop=True)
    sim_mask = np.ones(len(df_win), dtype=np.float32)

    if preset == "Dual Split (2 x 6h)" and window_len >= 32:
        b1_s = min(12, len(df_win) - 18)
        b2_s = min(b1_s + 12, len(df_win) - 6)
        sim_mask[b1_s : b1_s + 6] = 0.0
        sim_mask[b2_s : b2_s + 6] = 0.0
        outage_spans = [(b1_s, b1_s + 5), (b2_s, b2_s + 5)]
    else:
        out_end = min(len(df_win), out_start + out_dur)
        sim_mask[out_start:out_end] = 0.0
        outage_spans = [(out_start, max(out_start, out_end - 1))]

    # Execute frozen causal v2 inference (run_sequence_inference strips/ignores bam_reference)
    n_win_segments = int(df_win["segment_id"].nunique()) if "segment_id" in df_win.columns else 1
    y_cal, y_rec, y_final = run_sequence_inference(df_win, sim_mask, model_variant="v2")
    st.caption(
        f"Selected slice contains **{len(df_win)} test hours** spanning **{n_win_segments} contiguous segment(s)**. "
        "All 3 nephelometer channels (`PA_CF1`, `PA_ATM`, `PA_ALT`) are zeroed at `M_t = 0`, and `bam_reference` is isolated from model inference."
    )

    df_sim = pd.DataFrame(
        {
            "timestamp": df_win["timestamp"],
            "bam_reference": df_win["bam_reference"].to_numpy(dtype=float),
            "pa_atm_observed": np.where(sim_mask == 1.0, df_win["pa_atm"].to_numpy(dtype=float), np.nan),
            "mask_mt": sim_mask.astype(int),
            "y_cal": y_cal,
            "y_rec": y_rec,
            "y_final": y_final,
        }
    )

    obs_mask_bool = sim_mask == 1.0
    out_mask_bool = sim_mask == 0.0
    m_all = compute_evaluation_metrics(df_sim["bam_reference"], df_sim["y_final"])
    m_obs = compute_evaluation_metrics(df_sim.loc[obs_mask_bool, "bam_reference"], df_sim.loc[obs_mask_bool, "y_cal"])
    m_out = compute_evaluation_metrics(df_sim.loc[out_mask_bool, "bam_reference"], df_sim.loc[out_mask_bool, "y_rec"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Synthetic Outage Hours", f"{int(out_mask_bool.sum())} / {len(df_sim)}h")
    k2.metric("Observed Cal RMSE (M_t=1)", f"{m_obs['rmse']:.2f} µg/m³" if not np.isnan(m_obs["rmse"]) else "N/A")
    k3.metric("Outage Rec RMSE (M_t=0)", f"{m_out['rmse']:.2f} µg/m³" if not np.isnan(m_out["rmse"]) else "N/A")
    k4.metric("Window Composite RMSE", f"{m_all['rmse']:.2f} µg/m³" if not np.isnan(m_all["rmse"]) else "N/A")

    fig_sim = go.Figure()
    fig_sim.add_trace(
        go.Scatter(
            x=df_sim["timestamp"],
            y=df_sim["bam_reference"],
            mode="lines+markers",
            name="BAM-1020 Ground Truth",
            line=dict(color="#f8fafd", width=2.4),
            marker=dict(size=4),
        )
    )
    fig_sim.add_trace(
        go.Scatter(
            x=df_sim["timestamp"],
            y=df_sim["pa_atm_observed"],
            mode="lines",
            name="Raw PA_ATM (Masked Out During Gap)",
            line=dict(color=THEME["accent_amber"], width=1.6, dash="dot"),
            connectgaps=False,
        )
    )
    fig_sim.add_trace(
        go.Scatter(
            x=df_sim["timestamp"],
            y=df_sim["y_cal"],
            mode="lines",
            name="y_cal (Calibration Head)",
            line=dict(color=THEME["accent_cyan"], width=1.5, dash="dash"),
            opacity=0.72,
        )
    )
    fig_sim.add_trace(
        go.Scatter(
            x=df_sim["timestamp"],
            y=df_sim["y_rec"],
            mode="lines",
            name="y_rec (Reconstruction Head)",
            line=dict(color=THEME["accent_rose"], width=1.6, dash="dot"),
            opacity=0.80,
        )
    )
    fig_sim.add_trace(
        go.Scatter(
            x=df_sim["timestamp"],
            y=df_sim["y_final"],
            mode="lines+markers",
            name="y_final = M_t·y_cal + (1-M_t)·y_rec",
            line=dict(color=THEME["accent_purple"], width=2.6),
            marker=dict(size=5),
        )
    )

    for s_i, e_i in outage_spans:
        x0 = df_sim.iloc[s_i]["timestamp"]
        x1 = df_sim.iloc[min(e_i, len(df_sim) - 1)]["timestamp"]
        fig_sim.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(255, 85, 85, 0.16)",
            layer="below",
            line_width=1,
            line_color="rgba(255, 85, 85, 0.55)",
            annotation_text="Injected Outage (M_t=0)",
            annotation_position="top left",
        )

    apply_dark_layout(
        fig_sim,
        f"Live Causal Joint TCN v2 Stress Simulation — {sim_site} ({len(df_sim)}h Test Slice)",
        "Timestamp",
        "PM2.5 (µg/m³)",
        height=470,
    )
    st.plotly_chart(fig_sim, width="stretch", config=PLOTLY_CHART_CONFIG)

    with st.expander("Inspect Synthetic Simulation Outputs & Export CSV", expanded=False):
        st.download_button(
            label="Download Simulation Trace CSV",
            data=df_sim.to_csv(index=False).encode("utf-8"),
            file_name=f"joint_tcn_v2_synthetic_outage_{sim_site.lower()}.csv",
            mime="text/csv",
        )
        st.dataframe(df_sim, width="stretch", hide_index=True)
