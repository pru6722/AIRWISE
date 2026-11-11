"""
AIRWISE Dashboard: CPCB PM2.5 Regulatory Category & Boundary Reliability
Evaluates 24-hour CPCB PM2.5 compliance classification (CUPS/82/2014-15),
completeness requirements (>=18/24 hours), category confusion matrices,
boundary-proximity stability, and severe-episode failure analysis.

Checkpoint Provenance:
All CPCB regulatory tables and 24-hour window assessments on this page
evaluate the historical Checkpoint 5 / Phase 3A controlled frozen Joint TCN
(`frozen_joint_tcn_checkpoint.pt`) across 127 valid 24-hour test windows.
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

from dashboard.config import THEME, CPCB_CATEGORIES, CPCB_BREAKPOINTS, CPCB_COLORS
from dashboard.components.sidebar import render_benchmark_sidebar
from dashboard.components.metric_cards import render_kpi_row
from dashboard.components.charts import (
    PLOTLY_CHART_CONFIG,
    apply_dark_layout,
    create_cpcb_confusion_matrix,
)
from dashboard.services.data_service import load_predictions, load_metrics_table
from dashboard.services.cpcb_service import evaluate_cpcb_status
from dashboard.utils.formatting import (
    get_cpcb_badge_html,
    get_stability_badge_html,
    format_pm25,
)

# 1. Page Setup
st.set_page_config(
    page_title="AIRWISE — CPCB Regulatory Reliability",
    layout="wide",
    page_icon="🏛️",
)

css_path = DASHBOARD_DIR / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Shared Sidebar Filter Controls
filters = render_benchmark_sidebar(
    show_site_filter=True,
    show_date_filter=True,
    show_scenario_filter=False,
    show_pipeline_filter=False,
    show_population_filter=False,
    show_metric_filter=False,
    active_page="CPCB Reliability",
)
selected_site = filters["site"]
start_date = filters["start_date"]
end_date = filters["end_date"]

# 3. Load Frozen Checkpoint 5 & Phase 3A Controlled Artifacts
df_24h = load_predictions("cpcb_stability_assessment")
df_cat = load_metrics_table("checkpoint5_category_metrics")
df_site_rel = load_metrics_table("checkpoint5_site_reliability")
df_stab = load_metrics_table("checkpoint5_category_stability")
df_bound = load_metrics_table("checkpoint5_boundary_analysis")
df_24h_point = load_metrics_table("checkpoint5_point_metrics")
df_24h_cov = load_metrics_table("checkpoint5_coverage_metrics")
df_ctrl = load_metrics_table("cpcb_controlled_comparison")

# 4. Page Header & Authoritative Scope Disclosures
st.title("🏛️ CPCB PM2.5 Regulatory Band & Completeness Reliability")
st.markdown(
    "Evaluates 24-hour non-overlapping test windows (`N = 127` valid windows satisfying "
    "CPCB `>= 18/24` hourly completeness, `>= 75%`) across official CPCB PM2.5 breakpoint bands "
    "(`CUPS/82/2014-15`)."
)

st.warning(
    "**Regulatory Scope & Critical Failure Disclosures:**  \n"
    "1. **PM2.5-Only Evaluation (Not Full Multi-Pollutant NAQI):** AIRWISE evaluates CPCB 24-hour "
    "PM2.5 concentration bands (`Good` `<=30`, `Satisfactory` `30–60`, `Moderate` `60–90`, `Poor` `90–120`, "
    "`Very Poor` `120–250`, `Severe` `>250 µg/m³`). Official National Air Quality Index (`NAQI`) requires "
    "8 pollutants (`PM10`, `NO2`, `SO2`, `CO`, `O3`, `NH3`, `Pb`, `PM2.5`) and taking the dominant sub-index.  \n"
    "2. **Severe Category (`>250 µg/m³`) Complete Recall Failure:** Out of `127` test windows, only **4 windows** "
    "are truly Severe (all in Delhi), and the calibrated model recalls **0 / 4** (`Precision = 0.0%`, "
    "`Recall = 0.0%`, `F1 = 0.0%`, `24h MBE = -74.99 µg/m³`, `24h 90% Coverage = 0.0%`).  \n"
    "3. **Checkpoint Provenance:** Historical Checkpoint 5 / Phase 3A controlled evaluation of "
    "`frozen_joint_tcn_checkpoint.pt` (separate from Phase 3C Joint TCN v2)."
)

# 5. Executive KPI Row from Frozen Tables
pooled_pt = (
    df_24h_point[df_24h_point["subset"] == "Pooled Test"].iloc[0]
    if not df_24h_point.empty and (df_24h_point["subset"] == "Pooled Test").any()
    else None
)
tcn_ctrl = (
    df_ctrl[df_ctrl["model_name"].str.contains("Calibrated", na=False)].iloc[0]
    if not df_ctrl.empty
    else None
)
sev_row = (
    df_cat[df_cat["cpcb_category"] == "Severe"].iloc[0]
    if not df_cat.empty and (df_cat["cpcb_category"] == "Severe").any()
    else None
)
stab_row = df_stab.iloc[0] if not df_stab.empty else None

render_kpi_row([
    {
        "title": "24h Calibrated Point Accuracy",
        "value": f"{tcn_ctrl['accuracy_pct']:.2f}%" if tcn_ctrl is not None else "N/A",
        "subtitle": (
            f"Macro F1: {tcn_ctrl['macro_f1_pct']:.2f}% | Exact: {int(tcn_ctrl['exact_matches'])}/{int(tcn_ctrl['eligible_windows'])}"
            if tcn_ctrl is not None else ""
        ),
        "delta": (
            f"vs Raw ATM: 66.14% | Raw CF=1: 39.37%"
        ),
        "color": THEME["accent_teal"],
    },
    {
        "title": "Pooled 24h RMSE / R²",
        "value": f"{pooled_pt['rmse']:.2f} / {pooled_pt['r2']:.4f}" if pooled_pt is not None else "N/A",
        "subtitle": (
            f"MAE: {pooled_pt['mae']:.2f} | MBE: {pooled_pt['mbe']:.2f} µg/m³"
            if pooled_pt is not None else ""
        ),
        "delta": "N = 127 CPCB-eligible windows",
        "color": THEME["accent_cyan"],
    },
    {
        "title": "Severe Category (>250 µg/m³)",
        "value": f"0 / {int(sev_row['support'])} Recall" if sev_row is not None else "0 / 4 Recall",
        "subtitle": "Precision: 0.0% | Recall: 0.0% | F1: 0.0%",
        "delta": "Severe saturation bias (MBE -74.99 µg/m³)",
        "color": THEME["accent_rose"],
    },
    {
        "title": "Stability Heuristic Gain",
        "value": f"+{stab_row['stability_accuracy_gain']:.2f} pp" if stab_row is not None else "N/A",
        "subtitle": (
            f"Stable (n={int(stab_row['n_stable_windows'])}): {stab_row['point_accuracy_stable']:.2f}% vs "
            f"Ambiguous (n={int(stab_row['n_ambiguous_windows'])}): {stab_row['point_accuracy_ambiguous']:.2f}%"
            if stab_row is not None else ""
        ),
        "delta": "Operational boundary heuristic",
        "color": THEME["accent_amber"],
    },
])

st.markdown("---")

# 6. Helper for filtering 24h Assessment Frame cleanly
def _filter_24h_windows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sub = df.copy()
    if selected_site != "All Sites" and "site" in sub.columns:
        sub = sub[sub["site"] == selected_site]
    if "start_time" in sub.columns:
        ts = pd.to_datetime(sub["start_time"], errors="coerce")
        if start_date is not None:
            sub = sub[ts >= pd.Timestamp(start_date)]
            ts = pd.to_datetime(sub["start_time"], errors="coerce")
        if end_date is not None:
            sub = sub[ts <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return sub


df_24h_filtered = _filter_24h_windows(df_24h)


def _compute_confusion_and_prf(df: pd.DataFrame, pred_col: str, ref_col: str = "cat_ref"):
    valid = df.dropna(subset=[ref_col, pred_col]) if not df.empty else df
    cm = np.zeros((len(CPCB_CATEGORIES), len(CPCB_CATEGORIES)), dtype=int)
    if not valid.empty:
        ct = pd.crosstab(valid[ref_col], valid[pred_col]).reindex(
            index=CPCB_CATEGORIES, columns=CPCB_CATEGORIES, fill_value=0
        )
        cm = ct.to_numpy(dtype=int)

    rows = []
    for i, cat in enumerate(CPCB_CATEGORIES):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        pred_total = int(cm[:, i].sum())
        prec = (tp / pred_total * 100.0) if pred_total > 0 else 0.0
        rec = (tp / support * 100.0) if support > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({
            "CPCB Category": cat,
            "Support (True)": support,
            "Predicted Count": pred_total,
            "Exact Match": tp,
            "Precision (%)": round(prec, 2),
            "Recall (%)": round(rec, 2),
            "F1 Score (%)": round(f1, 2),
        })
    acc = (float(np.trace(cm)) / float(cm.sum()) * 100.0) if cm.sum() > 0 else 0.0
    return cm, pd.DataFrame(rows), acc, int(cm.sum())


# 7. Five Analytical Tabs
tab_overview, tab_cm, tab_sites, tab_stab, tab_sim = st.tabs([
    "📆 24h Windows & Completeness Audit",
    "🧩 Confusion Matrix & Per-Class Metrics",
    "📍 Site-Wise Reliability & Lag-1",
    "⚖️ Stability & Boundary Proximity",
    "🧪 Single-Window Decision Simulator",
])

# ---------------------------------------------------------------------------
# TAB 1: 24-Hour Windows & CPCB Completeness Audit
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("24-Hour Mean Agreement & Regulatory Completeness (>= 18/24 Hours)")

    if df_24h_filtered.empty:
        st.warning("No 24-hour assessment windows match the active site and date filters.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig_24 = go.Figure()
            for cat in CPCB_CATEGORIES:
                sub_c = df_24h_filtered[df_24h_filtered["cat_ref"] == cat]
                if sub_c.empty:
                    continue
                fig_24.add_trace(go.Scatter(
                    x=sub_c["ref_pm25_24h"],
                    y=sub_c["pred_pm25_24h"],
                    mode="markers",
                    name=f"Ref: {cat} (n={len(sub_c)})",
                    marker=dict(color=CPCB_COLORS.get(cat, THEME["accent_teal"]), size=9, opacity=0.85),
                    text=sub_c.apply(
                        lambda r: (
                            f"{r['site']} | {str(r['start_time'])[:10]}<br>"
                            f"Ref: {r['ref_pm25_24h']:.1f} ({r['cat_ref']})<br>"
                            f"Pred: {r['pred_pm25_24h']:.1f} ({r['cat_point']})<br>"
                            f"Valid hrs: {int(r['n_valid_hours'])}/24 ({r['completeness_pct']:.1f}%)"
                        ),
                        axis=1,
                    ),
                    hoverinfo="text",
                ))
            max_xy = float(max(df_24h_filtered["ref_pm25_24h"].max(), df_24h_filtered["pred_pm25_24h"].max(), 60.0)) * 1.06
            fig_24.add_trace(go.Scatter(
                x=[0.0, max_xy],
                y=[0.0, max_xy],
                mode="lines",
                name="1:1 Reference Agreement",
                line=dict(color=THEME["text_secondary"], dash="dash", width=1.4),
            ))
            for b_val in [30.0, 60.0, 90.0, 120.0, 250.0]:
                if b_val < max_xy:
                    fig_24.add_vline(x=b_val, line_dash="dot", line_color="rgba(148,163,184,0.3)")
                    fig_24.add_hline(y=b_val, line_dash="dot", line_color="rgba(148,163,184,0.3)")
            apply_dark_layout(
                fig_24,
                f"24h BAM vs Calibrated Joint TCN (n={len(df_24h_filtered)} Windows)",
                x_title="24h Reference BAM PM2.5 (µg/m³)",
                y_title="24h Calibrated Prediction (µg/m³)",
                height=440,
                hovermode="closest",
            )
            st.plotly_chart(fig_24, width="stretch", config=PLOTLY_CHART_CONFIG)

        with c2:
            hrs_counts = (
                df_24h_filtered["n_valid_hours"]
                .value_counts()
                .sort_index()
            )
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=[f"{int(h)}h" for h in hrs_counts.index],
                y=hrs_counts.values,
                text=hrs_counts.values,
                textposition="outside",
                marker_color=THEME["accent_cyan"],
                name="Windows",
            ))
            apply_dark_layout(
                fig_comp,
                "24h Completeness Eligibility Distribution (>=18h Valid Rule)",
                x_title="Valid Hours Inside 24h Window",
                y_title="Window Count",
                height=440,
                hovermode="x",
            )
            st.plotly_chart(fig_comp, width="stretch", config=PLOTLY_CHART_CONFIG)

        comp_summary = (
            df_24h_filtered.groupby("site", as_index=False)
            .agg(
                eligible_windows=("window_id", "count"),
                min_valid_hours=("n_valid_hours", "min"),
                mean_completeness_pct=("completeness_pct", "mean"),
                max_consec_gap_hours=("max_consecutive_missing_hours", "max"),
                mean_abs_residual=("abs_residual_24h", "mean"),
            )
        )
        comp_summary["mean_completeness_pct"] = comp_summary["mean_completeness_pct"].round(2)
        comp_summary["mean_abs_residual"] = comp_summary["mean_abs_residual"].round(2)
        st.dataframe(comp_summary, width="stretch", hide_index=True)

        csv_bytes = df_24h_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Filtered 24h CPCB Window Assessment CSV",
            data=csv_bytes,
            file_name=f"airwise_cpcb_24h_{selected_site.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="dl_cpcb_24h_slice",
        )

# ---------------------------------------------------------------------------
# TAB 2: Interactive Confusion Matrix & Controlled Model Comparison
# ---------------------------------------------------------------------------
with tab_cm:
    st.subheader("Interactive CPCB Band Confusion Matrix & Per-Class Performance")

    ctrl_m1, ctrl_m2 = st.columns(2)
    with ctrl_m1:
        cm_model_map = {
            "Calibrated Joint TCN (Checkpoint 5)": "cat_point",
            "Raw PurpleAir (ATM Factory Calibration)": "raw_atm_category",
            "Raw PurpleAir (CF=1 Standard Ratio)": "raw_cf1_category",
        }
        cm_model_label = st.selectbox(
            "Select Classification Model",
            options=list(cm_model_map.keys()),
            index=0,
            key="cpcb_cm_model_sel",
        )
        cm_pred_col = cm_model_map[cm_model_label]

    with ctrl_m2:
        cm_norm_mode = st.radio(
            "Matrix Values",
            options=["Raw Window Counts", "Row-Normalized Recall (%)"],
            index=0,
            horizontal=True,
            key="cpcb_cm_norm",
        )

    if df_24h_filtered.empty:
        st.warning("No 24h windows available under active filters.")
    else:
        cm_raw, df_prf, slice_acc, slice_n = _compute_confusion_and_prf(df_24h_filtered, cm_pred_col)

        if cm_norm_mode == "Row-Normalized Recall (%)":
            row_sums = cm_raw.sum(axis=1, keepdims=True)
            cm_display = np.where(row_sums > 0, np.round(cm_raw / np.maximum(row_sums, 1) * 100.0, 1), 0.0)
        else:
            cm_display = cm_raw

        col_cm_l, col_cm_r = st.columns([1.2, 1.0])
        with col_cm_l:
            fig_cm = create_cpcb_confusion_matrix(
                cm_display,
                CPCB_CATEGORIES,
                title=f"{cm_model_label} — {selected_site} (Acc {slice_acc:.2f}%, n={slice_n})",
            )
            if cm_norm_mode == "Row-Normalized Recall (%)":
                fig_cm.update_traces(texttemplate="%{z:.1f}%")
                fig_cm.update_coloraxes(colorbar_title="Recall (%)")
            st.plotly_chart(fig_cm, width="stretch", config=PLOTLY_CHART_CONFIG)

        with col_cm_r:
            fig_prf = go.Figure()
            fig_prf.add_trace(go.Bar(
                name="Precision (%)",
                x=df_prf["CPCB Category"],
                y=df_prf["Precision (%)"],
                marker_color=THEME["accent_teal"],
            ))
            fig_prf.add_trace(go.Bar(
                name="Recall (%)",
                x=df_prf["CPCB Category"],
                y=df_prf["Recall (%)"],
                marker_color=THEME["accent_amber"],
            ))
            fig_prf.add_trace(go.Bar(
                name="F1 Score (%)",
                x=df_prf["CPCB Category"],
                y=df_prf["F1 Score (%)"],
                marker_color=THEME["accent_purple"],
            ))
            fig_prf.update_layout(barmode="group")
            apply_dark_layout(
                fig_prf,
                f"Category Precision / Recall / F1 ({cm_model_label})",
                x_title="CPCB PM2.5 Band",
                y_title="Score (%)",
                height=440,
                hovermode="x unified",
            )
            fig_prf.update_yaxes(range=[0, 108])
            st.plotly_chart(fig_prf, width="stretch", config=PLOTLY_CHART_CONFIG)

        st.markdown("**Live Filtered Slice Per-Category Breakdown**")
        st.dataframe(df_prf, width="stretch", hide_index=True)

        st.markdown("**Authoritative Controlled Three-Model Comparison (`cpcb_controlled_comparison.csv`)**")
        if not df_ctrl.empty:
            st.dataframe(df_ctrl, width="stretch", hide_index=True)
            st.info(
                "**Kappa vs. Exact Accuracy Nuance (`Raw ATM` vs `Calibrated Joint TCN`):** Calibrated "
                "Joint TCN achieves the highest exact category accuracy (**72.44%**, `92/127` exact vs `84/127`) "
                "and Macro F1 (**58.96%** vs `55.76%`). However, `Raw ATM` yields slightly higher quadratic Cohen's "
                "κ (`0.9392` vs `0.9271`) because Joint TCN exhibits **2 large errors** (>1 category jump under "
                "Delhi saturation) that receive quadratic distance weight `4` while `Raw ATM` had `0` multi-band jumps."
            )

        with st.expander("Authoritative Frozen Category Metrics (`checkpoint5_category_metrics.csv`)"):
            st.dataframe(df_cat, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3: Site-Wise Reliability & Lag-1 Autocorrelation
# ---------------------------------------------------------------------------
with tab_sites:
    st.subheader("Station-Level Regulatory Accuracy, Interval Coverage & Autocorrelation")

    if not df_site_rel.empty:
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Bar(
                name="Category Accuracy (%)",
                x=df_site_rel["site"],
                y=df_site_rel["category_accuracy"],
                text=df_site_rel["category_accuracy"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker_color=THEME["accent_teal"],
            ))
            fig_sr.add_trace(go.Bar(
                name="24h Conformal 90% Cov (%)",
                x=df_site_rel["site"],
                y=df_site_rel["cov_90"],
                text=df_site_rel["cov_90"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker_color=THEME["accent_amber"],
            ))
            fig_sr.add_hline(y=90.0, line_dash="dash", line_color=THEME["text_primary"])
            fig_sr.update_layout(barmode="group")
            apply_dark_layout(
                fig_sr,
                "Site Category Accuracy vs 24h Interval Coverage",
                x_title="Monitoring Site",
                y_title="Percentage (%)",
                height=400,
                hovermode="x unified",
            )
            fig_sr.update_yaxes(range=[0, 112])
            st.plotly_chart(fig_sr, width="stretch", config=PLOTLY_CHART_CONFIG)

        with c_s2:
            fig_ac = go.Figure()
            fig_ac.add_trace(go.Bar(
                name="24h Residual Lag-1 Autocorrelation",
                x=df_site_rel["site"],
                y=df_site_rel["residual_lag1_autocorr"],
                text=df_site_rel["residual_lag1_autocorr"].map(lambda v: f"{v:.3f}"),
                textposition="outside",
                marker_color=THEME["accent_purple"],
            ))
            fig_ac.add_trace(go.Bar(
                name="Quadratic Cohen's Kappa",
                x=df_site_rel["site"],
                y=df_site_rel["cohen_kappa_quadratic"],
                text=df_site_rel["cohen_kappa_quadratic"].map(lambda v: f"{v:.3f}"),
                textposition="outside",
                marker_color=THEME["accent_cyan"],
            ))
            fig_ac.update_layout(barmode="group")
            apply_dark_layout(
                fig_ac,
                "24h Residual Persistence vs Ordinal Agreement (Kappa)",
                x_title="Monitoring Site",
                y_title="Correlation / Kappa",
                height=400,
                hovermode="x unified",
            )
            fig_ac.update_yaxes(range=[0, 1.15])
            st.plotly_chart(fig_ac, width="stretch", config=PLOTLY_CHART_CONFIG)

        st.markdown("**Site-Level Reliability (`checkpoint5_site_reliability.csv`)**")
        st.dataframe(df_site_rel, width="stretch", hide_index=True)

        col_pt, col_cv = st.columns(2)
        with col_pt:
            st.markdown("**24h Point Metrics by Subset (`checkpoint5_24h_point_metrics.csv`)**")
            st.dataframe(df_24h_point, width="stretch", hide_index=True)
        with col_cv:
            st.markdown("**24h Coverage Metrics by Subset (`checkpoint5_24h_coverage_metrics.csv`)**")
            st.dataframe(df_24h_cov, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4: Uncertainty-Aware Stability & Boundary Proximity
# ---------------------------------------------------------------------------
with tab_stab:
    st.subheader("Uncertainty-Aware Category Stability vs Boundary Proximity")
    st.markdown(
        "A 24-hour prediction is labeled **Stable** (`24/127 = 18.9%`) when its entire 90% conformal interval "
        "`[lower_90, upper_90]` (`width = 30.59 µg/m³`) lies strictly inside a single CPCB band, and "
        "**Ambiguous** (`103/127 = 81.1%`) when it straddles one or more regulatory cutoffs (`30, 60, 90, 120, 250`)."
    )
    st.caption(
        "**Operational Heuristic Disclaimer:** Stability flags are an uncertainty-aware operational triage rule, "
        "not a calibrated conformal classification probability guarantee. Observed `+13.43 pp` accuracy difference "
        "(`83.33%` stable vs `69.90%` ambiguous) reflects boundary geometry rather than formal hypothesis testing."
    )

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        if not df_stab.empty:
            sr = df_stab.iloc[0]
            fig_st = go.Figure()
            fig_st.add_trace(go.Bar(
                x=["All Windows (n=127)", f"Stable (n={int(sr['n_stable_windows'])})", f"Ambiguous (n={int(sr['n_ambiguous_windows'])})"],
                y=[sr["point_accuracy_overall"], sr["point_accuracy_stable"], sr["point_accuracy_ambiguous"]],
                text=[
                    f"{sr['point_accuracy_overall']:.2f}%",
                    f"{sr['point_accuracy_stable']:.2f}%",
                    f"{sr['point_accuracy_ambiguous']:.2f}%",
                ],
                textposition="outside",
                marker_color=[THEME["accent_cyan"], THEME["accent_teal"], THEME["accent_amber"]],
            ))
            apply_dark_layout(
                fig_st,
                f"Point Category Accuracy by Stability Flag (+{sr['stability_accuracy_gain']:.2f} pp)",
                x_title="Decision Stability Cohort",
                y_title="Category Accuracy (%)",
                height=390,
                hovermode="x",
            )
            fig_st.update_yaxes(range=[0, 105])
            st.plotly_chart(fig_st, width="stretch", config=PLOTLY_CHART_CONFIG)

    with c_b2:
        if not df_bound.empty:
            fig_bd = go.Figure()
            fig_bd.add_trace(go.Bar(
                name="Point Category Accuracy (%)",
                x=df_bound["distance_tier"],
                y=df_bound["point_accuracy"],
                text=df_bound["point_accuracy"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker_color=THEME["accent_teal"],
            ))
            fig_bd.add_trace(go.Bar(
                name="Ref Compatible w/ Interval (%)",
                x=df_bound["distance_tier"],
                y=df_bound["pct_ref_compatible"],
                text=df_bound["pct_ref_compatible"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker_color=THEME["accent_purple"],
            ))
            fig_bd.update_layout(barmode="group")
            apply_dark_layout(
                fig_bd,
                "Boundary-Distance Sensitivity of Regulatory Accuracy",
                x_title="Distance to Nearest CPCB Cutoff",
                y_title="Percentage (%)",
                height=390,
                hovermode="x unified",
            )
            fig_bd.update_yaxes(range=[0, 110])
            st.plotly_chart(fig_bd, width="stretch", config=PLOTLY_CHART_CONFIG)

    st.dataframe(df_stab, width="stretch", hide_index=True)
    st.dataframe(df_bound, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5: Illustrative Single-Window CPCB Threshold Simulator
# ---------------------------------------------------------------------------
with tab_sim:
    st.subheader("Illustrative 24-Hour CPCB Cutoff & Stability Evaluator")
    st.caption(
        "Interactive illustrative tool applying `evaluate_cpcb_status()` over CPCB `CUPS/82/2014-15` PM2.5 breakpoints. "
        "This decision-support aid does not modify or replace frozen research benchmark tables."
    )

    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        sim_point = st.slider(
            "24h Point Estimate (µg/m³)",
            min_value=0.0,
            max_value=450.0,
            value=82.5,
            step=0.5,
            key="cpcb_sim_point",
        )
    with cs2:
        sim_half = st.slider(
            "Conformal 90% Half-Width (±µg/m³)",
            min_value=5.0,
            max_value=120.0,
            value=15.3,
            step=0.5,
            key="cpcb_sim_half",
        )
    with cs3:
        include_ref = st.checkbox("Supply Reference BAM Value", value=False, key="cpcb_sim_ref_chk")
        sim_ref = (
            st.number_input("Reference 24h BAM (µg/m³)", min_value=0.0, max_value=500.0, value=92.0, step=1.0, key="cpcb_sim_ref")
            if include_ref else None
        )

    sim_lower = max(0.0, sim_point - sim_half)
    sim_upper = sim_point + sim_half
    res = evaluate_cpcb_status(sim_point, sim_lower, sim_upper, ref_value=sim_ref)

    badge_point = get_cpcb_badge_html(res["cat_point"])
    badge_stab = get_stability_badge_html(bool(res["is_stable"]))

    st.markdown(
        f"**Point Estimate:** `{format_pm25(sim_point)}` -> {badge_point} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**90% Band:** `[{format_pm25(sim_lower)}, {format_pm25(sim_upper)}]` -> {badge_stab}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- **Covered Categories:** {', '.join(res['covered_categories'])}  \n"
        f"- **Nearest Cutoff:** `{res['nearest_boundary_value']:.1f} µg/m³` (distance `{res['distance_to_nearest_boundary']:.1f} µg/m³`)  \n"
        f"- **Crossed Breakpoints:** `{res['crossed_boundaries'] if res['crossed_boundaries'] else 'None (Stable inside band)'}`"
        + (
            f"  \n- **Reference BAM Match:** `{res['cat_ref']}` | Point Match: `{res['is_point_correct']}` | Interval Compatible: `{res['is_ref_compatible']}`"
            if include_ref and res.get("cat_ref") not in (None, "Unknown") else ""
        )
    )
