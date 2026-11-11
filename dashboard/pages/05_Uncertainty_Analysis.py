"""
AIRWISE Dashboard: Uncertainty Quantification & Conformal Calibration Analysis
Visualizes split conformal dual-head intervals, site/regime conditioning,
pointwise vs. simultaneous temporal block coverage, and extreme-event breakdown.

Checkpoint Provenance:
All conformal uncertainty artifacts on this page evaluate the historical
Checkpoint 4 / Phase 3A frozen Joint TCN (`frozen_joint_tcn_checkpoint.pt`)
over the authoritative 60/10/10/20 split (Train 11,149 | Val 1,857 |
Conformal Cal 1,858 | Test 3,720).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
for p in [str(PROJECT_ROOT), str(DASHBOARD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import THEME, SITES
from dashboard.components.sidebar import render_benchmark_sidebar
from dashboard.components.metric_cards import render_kpi_row
from dashboard.components.charts import (
    PLOTLY_CHART_CONFIG,
    apply_dark_layout,
    create_uncertainty_ribbon_chart,
)
from dashboard.services.data_service import (
    load_predictions,
    load_metrics_table,
    filter_dataset,
)
from dashboard.services.uncertainty_service import load_conformal_quantiles

# 1. Page Setup
st.set_page_config(
    page_title="AIRWISE — Conformal Uncertainty Analysis",
    layout="wide",
    page_icon="🛡️",
)

css_path = DASHBOARD_DIR / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Shared Navigation & Filtering Sidebar
filters = render_benchmark_sidebar(
    show_site_filter=True,
    show_date_filter=True,
    show_scenario_filter=False,
    show_pipeline_filter=False,
    show_population_filter=False,
    show_metric_filter=False,
    active_page="Conformal Uncertainty",
)
selected_site = filters["site"]
start_date = filters["start_date"]
end_date = filters["end_date"]

# 3. Load Frozen Conformal Artifacts
df_conf = load_predictions("conformal")
quantiles = load_conformal_quantiles()

df_overall = load_metrics_table("conformal_coverage")
df_site = load_metrics_table("conformal_site_metrics")
df_humidity = load_metrics_table("conformal_humidity_metrics")
df_pollution = load_metrics_table("conformal_pollution_metrics")
df_block = load_metrics_table("conformal_block_comparison")
df_reconcil = load_metrics_table("conformal_coverage_reconciliation")
df_simul = load_metrics_table("conformal_simultaneous_block_coverage")
df_extreme = load_metrics_table("conformal_extreme_pollution")

# 4. Page Header & Scientific Provenance Disclosure
st.title("🛡️ Conformal Uncertainty Quantification & Reliability")
st.markdown(
    "Distribution-free uncertainty analysis evaluated on the **3,720-hour test set** "
    "(calibrated on **1,858 chronological conformal hours**). Compares dual-head "
    "ordinary split conformal intervals against regime-conditioned and temporal-block "
    "calibrations, and audits failure modes under temporal dependence and severe pollution."
)

st.info(
    "**Checkpoint Provenance & Key Methodological Limitations:**  \n"
    "1. **Model Checkpoint:** Evaluated strictly on the historical frozen Joint TCN "
    "(`frozen_joint_tcn_checkpoint.pt`, RMSE `48.30 µg/m³`, MBE `-8.84 µg/m³`), separate "
    "from Phase 3C's causal Joint TCN v2 (`joint_tcn_v2_causal_checkpoint.pt`).  \n"
    "2. **Non-Exchangeability:** High hourly lag-1 residual autocorrelation ($r_1 > 0.65$) "
    "violates the i.i.d. exchangeability assumption of ordinary split conformal, pulling "
    "90% marginal test coverage down to **77.02%** (`width 51.40 µg/m³`).  \n"
    "3. **Pointwise vs. Simultaneous Block Guarantees (Checkpoint 4 @ 90% Nominal):** "
    "Both 12h and 24h simultaneous rates come from the same Checkpoint 4 experiment "
    "(`conformal_simultaneous_block_coverage.csv`): **12h block** (`N_cal = 105`, width `88.67 µg/m³`) "
    "achieves **86.40% pointwise** vs. **63.83% simultaneous** (`120 / 188` blocks); "
    "**24h block** (`N_cal = 32`, width `200.84 µg/m³`) achieves **95.70% pointwise** vs. "
    "**74.58% simultaneous** (`44 / 59` blocks)."
)

# 5. Executive KPI Row from Frozen Tables
ord_row = (
    df_overall[df_overall["method"] == "Ordinary Split (Dual Head)"].iloc[0]
    if not df_overall.empty and (df_overall["method"] == "Ordinary Split (Dual Head)").any()
    else None
)
b12_row = (
    df_overall[df_overall["method"] == "Temporal Block Conformal (12h)"].iloc[0]
    if not df_overall.empty and (df_overall["method"] == "Temporal Block Conformal (12h)").any()
    else None
)
sim12_90 = (
    df_simul[(df_simul["block_size_hours"] == 12) & (df_simul["nominal_coverage_pct"] == 90)].iloc[0]
    if not df_simul.empty
    else None
)
sim24_90 = (
    df_simul[(df_simul["block_size_hours"] == 24) & (df_simul["nominal_coverage_pct"] == 90)].iloc[0]
    if not df_simul.empty
    else None
)
ext_row = (
    df_extreme[df_extreme["bam_band"].str.strip().str.startswith(">", na=False)].iloc[0]
    if not df_extreme.empty and df_extreme["bam_band"].str.strip().str.startswith(">", na=False).any()
    else None
)

render_kpi_row([
    {
        "title": "Ordinary Split (Dual Head)",
        "value": f"{ord_row['cov_90']:.2f}% / {ord_row['cov_95']:.2f}%" if ord_row is not None else "N/A",
        "subtitle": (
            f"Mean width: {ord_row['mean_width_90']:.2f} / {ord_row['mean_width_95']:.2f} µg/m³"
            if ord_row is not None else ""
        ),
        "delta": "Target: 90% / 95% (Pointwise)",
        "color": THEME["accent_amber"],
    },
    {
        "title": "Temporal Block 12h (Pointwise)",
        "value": f"{b12_row['cov_90']:.2f}% / {b12_row['cov_95']:.2f}%" if b12_row is not None else "N/A",
        "subtitle": (
            f"Width: {b12_row['mean_width_90']:.2f} / {b12_row['mean_width_95']:.2f} µg/m³"
            if b12_row is not None else ""
        ),
        "delta": "n_calib_blocks = 105",
        "color": THEME["accent_teal"],
    },
    {
        "title": "Simultaneous Block @ 90%",
        "value": (
            f"12h: {sim12_90['simultaneous_block_coverage_pct']:.2f}% | "
            f"24h: {sim24_90['simultaneous_block_coverage_pct']:.2f}%"
            if sim12_90 is not None and sim24_90 is not None
            else "N/A"
        ),
        "subtitle": (
            f"12h: {int(sim12_90['covered_blocks_count'])}/{int(sim12_90['n_test_blocks'])} | "
            f"24h: {int(sim24_90['covered_blocks_count'])}/{int(sim24_90['n_test_blocks'])} full blocks"
            if sim12_90 is not None and sim24_90 is not None
            else ""
        ),
        "delta": "Pointwise: 86.40% (12h) / 95.70% (24h)",
        "color": THEME["accent_cyan"],
    },
    {
        "title": "Extreme Band (>250 µg/m³)",
        "value": f"{ext_row['cov_90']:.2f}% / {ext_row['cov_95']:.2f}%" if ext_row is not None else "N/A",
        "subtitle": (
            f"Winkler-90: {ext_row['winkler_score_90']:.1f} | MBE: {ext_row['point_mbe']:.1f}"
            if ext_row is not None else ""
        ),
        "delta": f"n = {int(ext_row['sample_count'])} Delhi saturation hours" if ext_row is not None else "",
        "color": THEME["accent_rose"],
    },
])

st.markdown("---")

# 6. Five Interactive Tabs
tab_ts, tab_site, tab_env, tab_block, tab_ext = st.tabs([
    "📈 90% vs 95% Interval Explorer",
    "📍 Site Coverage Disparity",
    "🌦️ Humidity & Pollution Conditioning",
    "🧱 Block Calibration & Simultaneous Guarantee",
    "🚨 Extreme Pollution Breakdown",
])

# ---------------------------------------------------------------------------
# TAB 1: Dual-Head Time Series & 90% vs 95% Interval Explorer
# ---------------------------------------------------------------------------
with tab_ts:
    st.subheader("Dual-Head & Temporal Block Interval Explorer")

    df_slice = filter_dataset(
        df_conf,
        site=selected_site,
        start_date=start_date,
        end_date=end_date,
    )

    c_ctrl1, c_ctrl2 = st.columns(2)
    with c_ctrl1:
        cov_display_mode = st.radio(
            "Interval Display Level",
            options=["90% Interval", "95% Interval", "Overlay Both (90% + 95%)"],
            index=2,
            horizontal=True,
            key="unc_cov_mode",
        )
    with c_ctrl2:
        cal_method_options = {
            "Ordinary Split (Dual Head: Cal vs Rec)": "dual_head",
            "Temporal Block (6h: ±36.21 / ±45.31)": "block_6h",
            "Temporal Block (12h: ±44.34 / ±100.42)": "block_12h",
            "Temporal Block (24h: ±100.42 / ±141.68)": "block_24h",
        }
        cal_method_label = st.selectbox(
            "Uncertainty Calibration Basis",
            options=list(cal_method_options.keys()),
            index=0,
            key="unc_method_mode",
        )
        cal_method_key = cal_method_options[cal_method_label]

    if df_slice.empty:
        st.warning("No conformal test predictions match the active site and date filters.")
    else:
        df_view = df_slice.copy()
        if cal_method_key == "dual_head":
            df_view["view_lower_90"] = df_view["lower_90"]
            df_view["view_upper_90"] = df_view["upper_90"]
            df_view["view_lower_95"] = df_view["lower_95"]
            df_view["view_upper_95"] = df_view["upper_95"]
            method_note = (
                f"Dual-head calibration applies **Calibrated Head** ($q_{{90}}={quantiles.get('cal_head_q90', 24.50):.2f}$, "
                f"$q_{{95}}={quantiles.get('cal_head_q95', 29.16):.2f}$) on observed PM hours and wider "
                f"**Reconstruction Head** ($q_{{90}}={quantiles.get('rec_head_q90', 30.30):.2f}$, "
                f"$q_{{95}}={quantiles.get('rec_head_q95', 36.73):.2f}$) during PM outage hours."
            )
        else:
            q90_val = float(quantiles.get(f"{cal_method_key}_q90", 44.34))
            q95_val = float(quantiles.get(f"{cal_method_key}_q95", 100.42))
            df_view["view_lower_90"] = df_view["point_prediction"] - q90_val
            df_view["view_upper_90"] = df_view["point_prediction"] + q90_val
            df_view["view_lower_95"] = df_view["point_prediction"] - q95_val
            df_view["view_upper_95"] = df_view["point_prediction"] + q95_val
            method_note = (
                f"Temporal block calibration uses horizon-level block-maximum quantiles "
                f"($q_{{90}}=\\pm{q90_val:.2f}$, $q_{{95}}=\\pm{q95_val:.2f}$ µg/m³) computed "
                f"over contiguous conformal calibration blocks."
            )

        st.caption(method_note)

        # Compute exact slice coverage and widths
        in_90 = (df_view["true_bam"] >= df_view["view_lower_90"]) & (df_view["true_bam"] <= df_view["view_upper_90"])
        in_95 = (df_view["true_bam"] >= df_view["view_lower_95"]) & (df_view["true_bam"] <= df_view["view_upper_95"])
        w_90 = (df_view["view_upper_90"] - df_view["view_lower_90"]).mean()
        w_95 = (df_view["view_upper_95"] - df_view["view_lower_95"]).mean()
        cov_90_pct = float(in_90.mean() * 100.0)
        cov_95_pct = float(in_95.mean() * 100.0)
        n_slice = len(df_view)
        n_rec_slice = int((df_view["prediction_head"] == "Reconstruction").sum())

        render_kpi_row([
            {
                "title": "Filtered Slice Hours",
                "value": f"{n_slice:,}",
                "subtitle": f"Observed/Cal: {n_slice - n_rec_slice:,} | Outage/Rec: {n_rec_slice:,}",
                "color": THEME["accent_teal"],
            },
            {
                "title": "Slice 90% Coverage",
                "value": f"{cov_90_pct:.2f}%",
                "subtitle": f"Mean width: {w_90:.2f} µg/m³",
                "delta": f"Error vs 90%: {cov_90_pct - 90.0:+.2f} pp",
                "color": THEME["accent_cyan"],
            },
            {
                "title": "Slice 95% Coverage",
                "value": f"{cov_95_pct:.2f}%",
                "subtitle": f"Mean width: {w_95:.2f} µg/m³",
                "delta": f"Error vs 95%: {cov_95_pct - 95.0:+.2f} pp",
                "color": THEME["accent_purple"],
            },
            {
                "title": "Dual-Head q90 / q95",
                "value": f"{quantiles.get('cal_head_q90', 24.5):.2f} / {quantiles.get('rec_head_q90', 30.3):.2f}",
                "subtitle": (
                    f"Cal q95={quantiles.get('cal_head_q95', 29.16):.2f} | "
                    f"Rec q95={quantiles.get('rec_head_q95', 36.73):.2f}"
                ),
                "color": THEME["accent_amber"],
            },
        ])

        if selected_site == "All Sites":
            plot_site = st.selectbox(
                "Time Series Site View (Single Station Prevents Overlapping Site Timelines)",
                options=SITES,
                index=0,
                key="unc_plot_site_single",
            )
            df_plot = df_view[df_view["site"] == plot_site].sort_values("timestamp")
        else:
            plot_site = selected_site
            df_plot = df_view.sort_values("timestamp")

        if cov_display_mode == "90% Interval":
            fig_ts = create_uncertainty_ribbon_chart(
                df_plot,
                y_pred_col="point_prediction",
                lower_col="view_lower_90",
                upper_col="view_upper_90",
                ref_col="true_bam",
                title=f"{plot_site} — {cal_method_label} (90% Prediction Interval)",
                coverage_label="90% Conformal Interval",
            )
        elif cov_display_mode == "95% Interval":
            fig_ts = create_uncertainty_ribbon_chart(
                df_plot,
                y_pred_col="point_prediction",
                lower_col="view_lower_95",
                upper_col="view_upper_95",
                ref_col="true_bam",
                title=f"{plot_site} — {cal_method_label} (95% Prediction Interval)",
                coverage_label="95% Conformal Interval",
            )
        else:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["view_upper_95"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_ts.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["view_lower_95"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(189, 147, 249, 0.16)",
                name="95% Conformal Band",
            ))
            fig_ts.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["view_upper_90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_ts.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["view_lower_90"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 210, 196, 0.25)",
                name="90% Conformal Band",
            ))
            fig_ts.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["true_bam"],
                mode="lines",
                name="Reference BAM",
                line=dict(color=THEME["text_primary"], width=1.4),
            ))
            fig_ts.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["point_prediction"],
                mode="lines",
                name="Point Prediction",
                line=dict(color=THEME["accent_teal"], width=1.8),
            ))
            outage_pts = df_plot[df_plot["prediction_head"] == "Reconstruction"]
            if not outage_pts.empty:
                fig_ts.add_trace(go.Scatter(
                    x=outage_pts["timestamp"],
                    y=outage_pts["point_prediction"],
                    mode="markers",
                    name="Reconstruction Head Active (Outage)",
                    marker=dict(color=THEME["accent_amber"], size=5, symbol="diamond"),
                ))
            apply_dark_layout(
                fig_ts,
                f"{plot_site} — {cal_method_label} (90% + 95% Nested Bands)",
                x_title="Timestamp (UTC/Local)",
                y_title="PM2.5 (µg/m³)",
                height=460,
            )

        st.plotly_chart(fig_ts, width="stretch", config=PLOTLY_CHART_CONFIG)

        with st.expander("Dual-Head Interval Partition & Downloadable Test Slice", expanded=False):
            rec_mask = df_view["prediction_head"] == "Reconstruction"
            dual_summary = pd.DataFrame([
                {
                    "Prediction Head": "Calibration Head (PM Observed)",
                    "Test Hours": int((~rec_mask).sum()),
                    "90% Cov (%)": round(float(in_90[~rec_mask].mean() * 100.0), 2) if (~rec_mask).any() else np.nan,
                    "95% Cov (%)": round(float(in_95[~rec_mask].mean() * 100.0), 2) if (~rec_mask).any() else np.nan,
                    "Mean Width 90": round(float((df_view.loc[~rec_mask, "view_upper_90"] - df_view.loc[~rec_mask, "view_lower_90"]).mean()), 2) if (~rec_mask).any() else np.nan,
                    "Mean Width 95": round(float((df_view.loc[~rec_mask, "view_upper_95"] - df_view.loc[~rec_mask, "view_lower_95"]).mean()), 2) if (~rec_mask).any() else np.nan,
                },
                {
                    "Prediction Head": "Reconstruction Head (PM Outage)",
                    "Test Hours": int(rec_mask.sum()),
                    "90% Cov (%)": round(float(in_90[rec_mask].mean() * 100.0), 2) if rec_mask.any() else np.nan,
                    "95% Cov (%)": round(float(in_95[rec_mask].mean() * 100.0), 2) if rec_mask.any() else np.nan,
                    "Mean Width 90": round(float((df_view.loc[rec_mask, "view_upper_90"] - df_view.loc[rec_mask, "view_lower_90"]).mean()), 2) if rec_mask.any() else np.nan,
                    "Mean Width 95": round(float((df_view.loc[rec_mask, "view_upper_95"] - df_view.loc[rec_mask, "view_lower_95"]).mean()), 2) if rec_mask.any() else np.nan,
                },
            ])
            st.dataframe(dual_summary, width="stretch", hide_index=True)

            csv_bytes = df_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Filtered Conformal Intervals CSV",
                data=csv_bytes,
                file_name=f"airwise_conformal_{selected_site.lower().replace(' ', '_')}_{cal_method_key}.csv",
                mime="text/csv",
                key="dl_conformal_slice",
            )

# ---------------------------------------------------------------------------
# TAB 2: Site-Wise Spatial Coverage Disparity
# ---------------------------------------------------------------------------
with tab_site:
    st.subheader("Spatial Coverage Disparity Across Monitoring Stations (90% Nominal)")
    st.markdown(
        "Conformal intervals calibrated globally across sites mask dramatic spatial heterogeneity. "
        "**Bangalore** (clean tropical urban, RMSE 19.10) is overcovered across all methods, whereas "
        "**Delhi** (winter biomass/fog pollution, RMSE 71.79, MBE -28.25 µg/m³) suffers persistent undercoverage."
    )

    if not df_site.empty:
        methods_order = df_site["conformal_method"].drop_duplicates().tolist()
        fig_site = go.Figure()
        palette = [
            THEME["accent_teal"],
            THEME["accent_amber"],
            THEME["accent_cyan"],
            THEME["accent_purple"],
            THEME["accent_rose"],
        ]
        for idx, m_name in enumerate(methods_order):
            sub = df_site[df_site["conformal_method"] == m_name]
            fig_site.add_trace(go.Bar(
                name=m_name,
                x=sub["site"],
                y=sub["empirical_coverage"],
                text=sub["empirical_coverage"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker_color=palette[idx % len(palette)],
            ))
        fig_site.add_hline(
            y=90.0,
            line_dash="dash",
            line_color=THEME["text_primary"],
            annotation_text="90% Target Coverage",
            annotation_position="top right",
        )
        fig_site.update_layout(barmode="group")
        apply_dark_layout(
            fig_site,
            "Site-Stratified 90% Empirical Coverage by Conformal Method",
            x_title="Monitoring Station",
            y_title="Empirical Coverage (%)",
            height=450,
            hovermode="x unified",
        )
        fig_site.update_yaxes(range=[0, 112])
        st.plotly_chart(fig_site, width="stretch", config=PLOTLY_CHART_CONFIG)

        st.dataframe(df_site, width="stretch", hide_index=True)
        st.warning(
            "**Delhi Structural Undercoverage Alert:** Under **Ordinary Split**, Delhi reaches only "
            "**51.07%** coverage (`Winkler = 532.07`, `N = 1,502`). Even **Temporal Block Conformal (12h)** "
            "reaches only **68.77%** at Delhi while overcovering Hamirpur (**96.57%**) and Bangalore (**99.19%**)."
        )
    else:
        st.info("Site-level conformal metrics file could not be loaded.")

# ---------------------------------------------------------------------------
# TAB 3: Humidity & Predicted-Pollution Conditioning
# ---------------------------------------------------------------------------
with tab_env:
    st.subheader("Conditional Stratification: Humidity & Predicted-Pollution Regimes")
    st.markdown(
        "Conditional Mondrian partitioning attempts to adapt interval widths by local "
        "environmental regime, yet exposes two critical failure mechanisms: **cross-site meteorological confounding** "
        "and **calibration sample depletion**."
    )

    col_hum, col_pol = st.columns(2)

    with col_hum:
        st.markdown("#### 💧 Relative Humidity Conditioning (90% Target)")
        if not df_humidity.empty:
            hum_methods = df_humidity["conformal_method"].drop_duplicates().tolist()
            fig_hum = go.Figure()
            for idx, m_name in enumerate(hum_methods):
                sub = df_humidity[df_humidity["conformal_method"] == m_name]
                fig_hum.add_trace(go.Bar(
                    name=m_name,
                    x=sub["humidity_regime"],
                    y=sub["empirical_coverage"],
                    marker_color=[
                        THEME["accent_teal"], THEME["accent_amber"], THEME["accent_cyan"],
                        THEME["accent_purple"], THEME["accent_rose"],
                    ][idx % 5],
                ))
            fig_hum.add_hline(y=90.0, line_dash="dash", line_color=THEME["text_primary"])
            fig_hum.update_layout(barmode="group")
            apply_dark_layout(
                fig_hum,
                "Coverage by RH Regime (Dry/Moderate Conflates Delhi Winter)",
                x_title="Humidity Regime",
                y_title="90% Coverage (%)",
                height=390,
                hovermode="x unified",
            )
            fig_hum.update_yaxes(range=[0, 110])
            st.plotly_chart(fig_hum, width="stretch", config=PLOTLY_CHART_CONFIG)
            st.caption(
                "**Physical Confounding Finding:** Medium RH (`40–65% RH`, `n = 1,993`) exhibits the highest "
                "error (`RMSE = 58.96 µg/m³`, `MBE = -15.36`) and lowest coverage (`70.35%` Ordinary Split), "
                "whereas High RH (`>65% RH`, `n = 572`) achieves `92.66%` (`RMSE = 17.68`). Moderate RH in this dataset "
                "disproportionately overlaps Delhi daytime/transition winter conditions."
            )
        else:
            st.info("Humidity metrics CSV unavailable.")

    with col_pol:
        st.markdown("#### 🌫️ Predicted Pollution Conditioning (90% Target)")
        if not df_pollution.empty:
            pol_methods = df_pollution["conformal_method"].drop_duplicates().tolist()
            fig_pol = go.Figure()
            for idx, m_name in enumerate(pol_methods):
                sub = df_pollution[df_pollution["conformal_method"] == m_name]
                fig_pol.add_trace(go.Bar(
                    name=m_name,
                    x=sub["predicted_pollution_regime"],
                    y=sub["empirical_coverage"],
                    marker_color=[
                        THEME["accent_teal"], THEME["accent_amber"], THEME["accent_cyan"],
                        THEME["accent_purple"], THEME["accent_rose"],
                    ][idx % 5],
                ))
            fig_pol.add_hline(y=90.0, line_dash="dash", line_color=THEME["text_primary"])
            fig_pol.update_layout(barmode="group")
            apply_dark_layout(
                fig_pol,
                "Coverage Collapse in Severe (>120 µg/m³) Predictions",
                x_title="Predicted Pollution Regime",
                y_title="90% Coverage (%)",
                height=390,
                hovermode="x unified",
            )
            fig_pol.update_yaxes(range=[0, 110])
            st.plotly_chart(fig_pol, width="stretch", config=PLOTLY_CHART_CONFIG)
            st.caption(
                "**Calibration Sample Depletion (`N_cal = 7`):** Predicted-Pollution-Conditioned improves "
                "`Moderate (60–120 µg/m³)` coverage from `68.59%` to `83.31%` (`width 78.47`). However, in "
                "`Severe (>120 µg/m³)`, only **7 calibration samples** fell into the Severe bin, causing "
                "conditioning to collapse to **34.31%** coverage (`RMSE = 92.68`, `MBE = -36.15`)."
            )
        else:
            st.info("Pollution regime metrics CSV unavailable.")

    with st.expander("View Complete Stratified Metrics Tables (Humidity & Pollution)", expanded=False):
        if not df_humidity.empty:
            st.markdown("**Humidity-Stratified Test Performance (`conformal_humidity_metrics.csv`)**")
            st.dataframe(df_humidity, width="stretch", hide_index=True)
        if not df_pollution.empty:
            st.markdown("**Predicted-Pollution-Stratified Test Performance (`conformal_pollution_metrics.csv`)**")
            st.dataframe(df_pollution, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4: Temporal Block Calibration & Pointwise vs Simultaneous Guarantee
# ---------------------------------------------------------------------------
with tab_block:
    st.subheader("Temporal Block Conformal Calibration & Simultaneous Guarantees")
    st.markdown(
        "Replacing single-hour nonconformity scores with block-maximum residuals across contiguous "
        "**6h, 12h, and 24h** intervals mitigates short-range autocorrelation—but at the cost of fewer "
        "effective calibration blocks (`1,858 -> 265 -> 105 -> 32`) and wider fixed bands."
    )

    c_left, c_right = st.columns(2)

    with c_left:
        if not df_reconcil.empty and not df_simul.empty:
            rec_90 = df_reconcil[
                (df_reconcil["nominal_coverage_pct"] == 90)
                & (df_reconcil["method"].str.contains("Block", na=False))
            ].copy()
            rec_90["horizon"] = rec_90["method"].str.extract(r"(\d+h)")[0]
            sim_90 = df_simul[df_simul["nominal_coverage_pct"] == 90].copy()
            sim_90["horizon"] = sim_90["block_size_hours"].astype(str) + "h"
            merged_90 = pd.merge(rec_90, sim_90, on="horizon", how="inner")

            rec_95 = df_reconcil[
                (df_reconcil["nominal_coverage_pct"] == 95)
                & (df_reconcil["method"].str.contains("Block", na=False))
            ].copy()
            rec_95["horizon"] = rec_95["method"].str.extract(r"(\d+h)")[0]
            sim_95 = df_simul[df_simul["nominal_coverage_pct"] == 95].copy()
            sim_95["horizon"] = sim_95["block_size_hours"].astype(str) + "h"
            merged_95 = pd.merge(rec_95, sim_95, on="horizon", how="inner")

            fig_pvs = go.Figure()
            fig_pvs.add_trace(go.Bar(
                name="Pointwise Hourly (90% Nominal)",
                x=merged_90["horizon"],
                y=merged_90["empirical_coverage_pct"],
                text=merged_90["empirical_coverage_pct"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker_color=THEME["accent_teal"],
            ))
            fig_pvs.add_trace(go.Bar(
                name="Simultaneous Full-Block (90% Nominal)",
                x=merged_90["horizon"],
                y=merged_90["simultaneous_block_coverage_pct"],
                text=merged_90["simultaneous_block_coverage_pct"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker_color=THEME["accent_amber"],
            ))
            fig_pvs.add_trace(go.Scatter(
                name="Pointwise Hourly (95% Nominal)",
                x=merged_95["horizon"],
                y=merged_95["empirical_coverage_pct"],
                mode="lines+markers",
                line=dict(color=THEME["accent_cyan"], width=2.5),
            ))
            fig_pvs.add_trace(go.Scatter(
                name="Simultaneous Full-Block (95% Nominal)",
                x=merged_95["horizon"],
                y=merged_95["simultaneous_block_coverage_pct"],
                mode="lines+markers",
                line=dict(color=THEME["accent_purple"], width=2.5, dash="dot"),
            ))
            fig_pvs.add_hline(y=90.0, line_dash="dash", line_color=THEME["text_secondary"])
            fig_pvs.update_layout(barmode="group")
            apply_dark_layout(
                fig_pvs,
                "Pointwise vs. Simultaneous Full-Block Coverage",
                x_title="Block Horizon",
                y_title="Coverage (%)",
                height=420,
                hovermode="x unified",
            )
            fig_pvs.update_yaxes(range=[0, 112])
            st.plotly_chart(fig_pvs, width="stretch", config=PLOTLY_CHART_CONFIG)

    with c_right:
        if not df_block.empty:
            fig_blk = go.Figure()
            fig_blk.add_trace(go.Scatter(
                x=df_block["block_duration_hours"],
                y=df_block["mean_interval_width_90"],
                mode="lines+markers+text",
                name="Mean Width 90% (µg/m³)",
                text=df_block["n_calib_blocks"].map(lambda n: f"n_cal={n}"),
                textposition="top center",
                line=dict(color=THEME["accent_cyan"], width=2.5),
                marker=dict(size=9),
            ))
            fig_blk.add_trace(go.Scatter(
                x=df_block["block_duration_hours"],
                y=df_block["winkler_score_90"],
                mode="lines+markers",
                name="Winkler Score 90% (lower=better)",
                line=dict(color=THEME["accent_amber"], width=2.5, dash="dot"),
                marker=dict(size=9),
            ))
            apply_dark_layout(
                fig_blk,
                "Interval Width & Winkler Score vs Calibration Block Count",
                x_title="Block Duration (Hours)",
                y_title="µg/m³",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_blk, width="stretch", config=PLOTLY_CHART_CONFIG)

    st.markdown("**Overall Comparison (`conformal_coverage.csv` & `conformal_simultaneous_block_coverage.csv`)**")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.dataframe(df_overall, width="stretch", hide_index=True)
    with col_t2:
        st.dataframe(df_simul, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5: Extreme Pollution Saturation Breakdown
# ---------------------------------------------------------------------------
with tab_ext:
    st.subheader("Extreme Haze & Sensor Saturation Failure Mode (Delhi Test Hours)")
    st.markdown(
        "During winter severe smog events (reference BAM `> 250 µg/m³`, `n = 257` hours), PurpleAir plantower "
        "optical scattering saturates and systematically underpredicts PM2.5 (`MBE = -111.16 µg/m³`), "
        "causing split conformal intervals (`width ~ 51.62 µg/m³`) to miss virtually every extreme peak."
    )

    if not df_extreme.empty:
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            fig_ecov = go.Figure()
            fig_ecov.add_trace(go.Bar(
                name="90% Empirical Coverage",
                x=df_extreme["bam_band"],
                y=df_extreme["cov_90"],
                text=df_extreme["cov_90"].map(lambda v: f"{v:.2f}%"),
                textposition="outside",
                marker_color=THEME["accent_teal"],
            ))
            fig_ecov.add_trace(go.Bar(
                name="95% Empirical Coverage",
                x=df_extreme["bam_band"],
                y=df_extreme["cov_95"],
                text=df_extreme["cov_95"].map(lambda v: f"{v:.2f}%"),
                textposition="outside",
                marker_color=THEME["accent_purple"],
            ))
            fig_ecov.add_hline(y=90.0, line_dash="dash", line_color=THEME["accent_teal"])
            fig_ecov.add_hline(y=95.0, line_dash="dot", line_color=THEME["accent_purple"])
            fig_ecov.update_layout(barmode="group")
            apply_dark_layout(
                fig_ecov,
                "Coverage Collapse Above 120 µg/m³ BAM Reference",
                x_title="Reference BAM Band",
                y_title="Empirical Coverage (%)",
                height=400,
                hovermode="x unified",
            )
            fig_ecov.update_yaxes(range=[0, 108])
            st.plotly_chart(fig_ecov, width="stretch", config=PLOTLY_CHART_CONFIG)

        with c_e2:
            fig_ewink = go.Figure()
            fig_ewink.add_trace(go.Bar(
                name="Winkler Score (90%)",
                x=df_extreme["bam_band"],
                y=df_extreme["winkler_score_90"],
                text=df_extreme["winkler_score_90"].map(lambda v: f"{v:.1f}"),
                textposition="outside",
                marker_color=THEME["accent_rose"],
            ))
            fig_ewink.add_trace(go.Bar(
                name="Point RMSE (µg/m³)",
                x=df_extreme["bam_band"],
                y=df_extreme["point_rmse"],
                text=df_extreme["point_rmse"].map(lambda v: f"{v:.1f}"),
                textposition="outside",
                marker_color=THEME["accent_amber"],
            ))
            fig_ewink.update_layout(barmode="group")
            apply_dark_layout(
                fig_ewink,
                "Winkler Penalty & RMSE Escalation Under Optical Saturation",
                x_title="Reference BAM Band",
                y_title="Score / µg/m³",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig_ewink, width="stretch", config=PLOTLY_CHART_CONFIG)

        st.dataframe(df_extreme, width="stretch", hide_index=True)
    else:
        st.info("Extreme pollution conformal breakdown CSV unavailable.")
