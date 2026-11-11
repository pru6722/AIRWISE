"""
AIRWISE Dashboard Page 2: Dataset Explorer & Sensor Diagnostics
Inspects the 18,584-hour un-imputed dataset & raw Campmier Excel archive:
- Independent channel availability & 3-level outage taxonomy (Partial PM, Complete PM, Station Outage)
- Interactive natural gap heatmaps & Phase 3C scenario mask heatmaps
- Concentration distributions, multi-channel scatter, Pearson correlation matrix
- Psychrometric Td > T audit & filtered CSV downloads preserving raw measurements
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import THEME, SITES, SCENARIOS
from dashboard.components.sidebar import render_benchmark_sidebar
from dashboard.components.metric_cards import render_kpi_row
from dashboard.components.charts import (
    PLOTLY_CHART_CONFIG,
    apply_dark_layout,
    create_timeseries_chart,
    create_scatter_comparison,
    create_channel_availability_chart,
    create_correlation_heatmap,
)
from dashboard.services.data_service import (
    load_cleaned_dataset,
    load_benchmark_predictions,
    load_metrics_table,
    get_raw_sheet_names,
    load_raw_sheet,
    filter_dataset,
)

# Page Config & CSS
st.set_page_config(page_title="AIRWISE — Dataset Explorer", layout="wide", page_icon="📁")

css_path = DASHBOARD_DIR / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

filters = render_benchmark_sidebar(
    show_site_filter=True,
    show_date_filter=True,
    show_scenario_filter=True,
    show_pipeline_filter=False,
    show_population_filter=False,
    show_metric_filter=False,
    active_page="Dataset Explorer",
)

selected_site: str = filters["site"]
start_date = filters["start_date"]
end_date = filters["end_date"]
scenario_id: str = filters["scenario_id"]
scenario_meta = SCENARIOS.get(scenario_id, {"name": scenario_id})

# Title
st.title("📁 Dataset Explorer & Multi-Channel Sensor Diagnostics")
st.markdown(
    """
    Explore the curated empirical dataset from **Campmier et al. (2023)** (*Atmospheric Measurement Techniques*, 16, 4357–4374 |
    Dryad `doi:10.6078/D1RQ70`, CC0 1.0). All raw and curated timestamps preserve original measurements without silent imputation.
    """
)

# Load Data
df_all = load_cleaned_dataset()
df_filtered = filter_dataset(df_all, site=selected_site, start_date=start_date, end_date=end_date)
df_bench = load_benchmark_predictions()
df_gap_report = load_metrics_table("missing_timestamp_report")
df_bias_summary = load_metrics_table("sensor_bias_summary")
df_dew_audit = load_metrics_table("dew_point_investigation")
df_audit_stats = load_metrics_table("dataset_audit")

total_obs = len(df_filtered)
bam_mean = float(df_filtered["bam_reference"].mean()) if total_obs > 0 else 0.0
bam_max = float(df_filtered["bam_reference"].max()) if total_obs > 0 else 0.0
n_segments = int(df_filtered["segment_id"].nunique()) if total_obs > 0 else 0
n_gaps = int((df_filtered["time_diff_hours"] > 1.05).sum()) if total_obs > 0 else 0

render_kpi_row(
    [
        {
            "title": "Filtered Observations",
            "value": f"{total_obs:,}",
            "subtitle": f"Site: {selected_site}",
            "delta": "Preserves Raw Un-Imputed Rows",
            "color": THEME["accent_teal"],
        },
        {
            "title": "Mean BAM-1020",
            "value": f"{bam_mean:.1f} µg/m³",
            "subtitle": f"Peak BAM: {bam_max:.1f} µg/m³",
            "color": THEME["accent_cyan"],
        },
        {
            "title": "Contiguous Segments",
            "value": str(n_segments),
            "subtitle": f"{n_gaps:,} Timestamp Gaps (>1.05h)",
            "color": THEME["accent_amber"],
        },
        {
            "title": "Psychrometric Check",
            "value": "100.0% Td > T",
            "subtitle": "dew_point_raw Quarantined",
            "delta": "Stored for Audit, Excluded from Models",
            "color": THEME["accent_rose"],
        },
    ]
)

st.markdown("---")

tab_ts, tab_missing, tab_dist_corr, tab_raw_audit, tab_export = st.tabs(
    [
        "📈 Time-Series & Channel Availability",
        "⏳ Missingness Heatmaps & Taxonomy",
        "📊 Distributions & Correlations",
        "🔬 Raw Measurements & Td Audit",
        "📋 Filtered Data Export",
    ]
)

# =============================================================================
# TAB 1: TIME-SERIES & INDEPENDENT CHANNEL AVAILABILITY
# =============================================================================
with tab_ts:
    st.markdown("### Independent Channel Availability Profile (Curated Measurements)")
    st.caption(
        "Computes independent observation availability (%) per PurpleAir nephelometer channel (`pa_cf1`, `pa_atm`, `pa_alt`), "
        "meteorological covariates (`relative_humidity`, `temperature`), and target monitor (`bam_reference`) across the active slice. "
        "Notice: curated `cleaned_data.parquet` contains complete retained hours (100% column presence where station is online), "
        "while physical field outages manifest as missing clock hours in the Missingness tab."
    )
    if total_obs > 0:
        fig_avail = create_channel_availability_chart(
            df_filtered,
            title=f"Independent Channel Availability ({selected_site}, N = {total_obs:,} Retained Hours)",
        )
        st.plotly_chart(fig_avail, width="stretch", config=PLOTLY_CHART_CONFIG)

    st.markdown("### Multichannel Concentration Time-Series vs. BAM Reference")
    channel_options = st.multiselect(
        "Select Visible Concentration Channels:",
        options=["bam_reference", "pa_atm", "pa_cf1", "pa_alt"],
        default=["bam_reference", "pa_atm", "pa_cf1"],
        format_func=lambda x: {
            "bam_reference": "BAM-1020 Reference Monitor (Target)",
            "pa_atm": "PurpleAir ATM (Atmospheric)",
            "pa_cf1": "PurpleAir CF=1 (Factory)",
            "pa_alt": "PurpleAir ALT (Alternate)",
        }.get(x, x),
    )

    if total_obs > 0 and channel_options:
        fig_ts = create_timeseries_chart(
            df_filtered,
            y_cols=channel_options,
            labels={
                "bam_reference": "BAM-1020 Reference",
                "pa_atm": "PA_ATM",
                "pa_cf1": "PA_CF1",
                "pa_alt": "PA_ALT",
            },
            colors={
                "bam_reference": THEME["accent_cyan"],
                "pa_atm": THEME["accent_teal"],
                "pa_cf1": THEME["accent_rose"],
                "pa_alt": THEME["accent_amber"],
            },
            title=f"Raw Sensor vs. BAM Time Series — {selected_site}",
            show_rangeslider=True,
        )
        st.plotly_chart(fig_ts, width="stretch", config=PLOTLY_CHART_CONFIG)
    else:
        st.info("Select at least one channel and a non-empty date/site filter.")

    st.markdown("### Ambient Meteorology & Confounding Notice")
    st.info(
        "**Observed Association vs. Causality:** High RH (`>65%`) in this archive occurs primarily during monsoon periods "
        "in Bangalore (`~20 µg/m³`), whereas low RH (`<40%`) predominates during Delhi winter inversions (`>200 µg/m³`). "
        "Humidity and baseline concentration regimes are naturally confounded across sites and seasons.",
        icon="ℹ️",
    )
    if total_obs > 0:
        c_rh, c_tp = st.columns(2)
        with c_rh:
            fig_rh = go.Figure(
                go.Scatter(
                    x=df_filtered["timestamp"],
                    y=df_filtered["relative_humidity"],
                    mode="lines",
                    name="RH (%)",
                    line=dict(color=THEME["accent_teal"], width=1.3),
                    connectgaps=False,
                )
            )
            apply_dark_layout(fig_rh, "Relative Humidity (%)", "Timestamp", "RH (%)", height=340)
            st.plotly_chart(fig_rh, width="stretch", config=PLOTLY_CHART_CONFIG)
        with c_tp:
            fig_tp = go.Figure(
                go.Scatter(
                    x=df_filtered["timestamp"],
                    y=df_filtered["temperature"],
                    mode="lines",
                    name="Temperature (°C)",
                    line=dict(color=THEME["accent_amber"], width=1.3),
                    connectgaps=False,
                )
            )
            apply_dark_layout(fig_tp, "Ambient Dry-Bulb Temperature (°C)", "Timestamp", "Temperature (°C)", height=340)
            st.plotly_chart(fig_tp, width="stretch", config=PLOTLY_CHART_CONFIG)

# =============================================================================
# TAB 2: MISSINGNESS HEATMAPS & 3-TIER OUTAGE TAXONOMY
# =============================================================================
with tab_missing:
    st.markdown("### Missingness Taxonomy: Partial PM vs. Complete PM vs. Complete Station Outage")
    taxonomy_df = pd.DataFrame(
        [
            {
                "Outage Regime": "1. Partial PM-Channel Missingness",
                "Channel Mask State": "1 or 2 of {PA_CF1, PA_ATM, PA_ALT} missing; remaining PM + Meteo observed",
                "Presence in Dataset": "Supported structurally in Pipeline A/B multi-channel imputers (preserved without discarding observed PM channels).",
                "Model Handling": "Observed PM channels strictly retained (Sequential Pipelines); Phase 3C.2 benchmark simulates synchronized complete PM dropout.",
            },
            {
                "Outage Regime": "2. Complete PM Dropout (m_t^PM = 0)",
                "Channel Mask State": "All 3 PM channels {PA_CF1, PA_ATM, PA_ALT} missing; RH, Temp & cyclical time observed",
                "Presence in Dataset": "Evaluated systematically across all 9 Phase 3C scenarios on N=3,720 test hours (`benchmark_predictions.parquet`).",
                "Model Handling": "Pipeline A imputes raw PM from meteo; Pipeline B reconstructs calibrated PM from meteo/lags; Pipeline C masks PM inputs & invokes Reconstruction Head; Pipeline D abstains (NaN).",
            },
            {
                "Outage Regime": "3. Complete Station Outage (Blackout)",
                "Channel Mask State": "All PM and meteorological telemetry offline simultaneously",
                "Presence in Dataset": "Observed naturally in raw Campmier timestamps wherever consecutive clock steps exceed 1.05h (Delhi 40.82%, Hamirpur 48.65%, Bangalore 23.64% missing clock hours).",
                "Model Handling": "Natural gaps create new `segment_id` boundaries (no cross-gap interpolation or sequence bridging); sequential fallbacks use training medians.",
            },
        ]
    )
    st.dataframe(taxonomy_df, width="stretch", hide_index=True)

    st.markdown("#### Natural Timestamp Completeness Heatmap (Calendar Month × Hour-of-Day)")
    if total_obs > 0:
        hm_site = selected_site if selected_site != "All Sites" else st.selectbox(
            "Select Station for Hourly Completeness Heatmap",
            options=SITES,
            index=0,
            key="de_hm_site",
        )
        df_site_hm = df_filtered[df_filtered["site"] == hm_site].copy()
        if not df_site_hm.empty:
            t_min = df_site_hm["timestamp"].min().floor("h")
            t_max = df_site_hm["timestamp"].max().floor("h")
            full_hours = pd.date_range(t_min, t_max, freq="1h")
            obs_hours_set = set(df_site_hm["timestamp"].dt.floor("h"))
            grid_df = pd.DataFrame({"hour_ts": full_hours})
            grid_df["observed"] = grid_df["hour_ts"].isin(obs_hours_set).astype(float)
            grid_df["year_month"] = grid_df["hour_ts"].dt.strftime("%Y-%m")
            grid_df["hour_of_day"] = grid_df["hour_ts"].dt.hour

            piv = grid_df.pivot_table(
                index="year_month",
                columns="hour_of_day",
                values="observed",
                aggfunc="mean",
            ) * 100.0

            fig_nat_hm = px.imshow(
                piv,
                labels=dict(x="Hour of Day (UTC/Local Archive)", y="Year-Month", color="Observed (%)"),
                color_continuous_scale=[[0.0, "#1e293b"], [0.5, "#ffb86c"], [1.0, "#00d2c4"]],
                zmin=0,
                zmax=100,
                aspect="auto",
                title=f"Natural Station Continuity Heatmap — {hm_site} (% Expected Hours Observed)",
            )
            apply_dark_layout(fig_nat_hm, f"Natural Station Completeness Heatmap — {hm_site}", "Hour of Day", "Month", height=420, hovermode="closest")
            st.plotly_chart(fig_nat_hm, width="stretch", config=PLOTLY_CHART_CONFIG)

        gaps = df_filtered[df_filtered["time_diff_hours"] > 1.05]
        c_g1, c_g2 = st.columns([1.2, 1.0])
        with c_g1:
            if not gaps.empty:
                fig_gaps = px.histogram(
                    gaps,
                    x="time_diff_hours",
                    color="site",
                    nbins=30,
                    title=f"Natural Station Outage Durations (> 1.05h) — {len(gaps):,} Discrete Gaps",
                    labels={"time_diff_hours": "Gap Duration (Hours)", "site": "Station"},
                    color_discrete_sequence=[THEME["accent_amber"], THEME["accent_cyan"], THEME["accent_teal"]],
                )
                apply_dark_layout(fig_gaps, "Distribution of Natural Gap Durations", "Outage Length (Hours)", "Gap Count", height=360)
                st.plotly_chart(fig_gaps, width="stretch", config=PLOTLY_CHART_CONFIG)
            else:
                st.info("No gaps > 1.05 hours detected in the current selection.")
        with c_g2:
            if not df_gap_report.empty:
                st.markdown("**Authoritative Station Timestamp Gap Audit (`missing_timestamp_report.csv`):**")
                st.dataframe(df_gap_report, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown(f"#### Phase 3C Synthetic Scenario Mask Matrix Heatmap — {scenario_meta.get('name', scenario_id)}")
    mask_col = f"mask_{scenario_id}"
    if not df_bench.empty and mask_col in df_bench.columns:
        bench_site = selected_site if selected_site != "All Sites" else st.selectbox(
            "Select Test Station for Scenario Mask Inspection",
            options=SITES,
            index=0,
            key="de_mask_site",
        )
        sub_b = df_bench[df_bench["site"] == bench_site].sort_values("timestamp").reset_index(drop=True)
        if not sub_b.empty:
            m_pm = sub_b[mask_col].to_numpy(dtype=float)  # 1 = observed, 0 = Complete PM dropout
            n_slice = min(240, len(sub_b))
            m_slice = m_pm[:n_slice]
            ts_slice = sub_b["timestamp"].iloc[:n_slice].dt.strftime("%Y-%m-%d %H:%M").tolist()

            # Construct channel availability matrix matching Phase 3C protocol:
            # All 3 PM channels share m_t^PM in Complete PM Dropout; RH/Temp remain observed (1.0)
            mat = np.vstack([m_slice, m_slice, m_slice, np.ones(n_slice), np.ones(n_slice)])
            ch_names = ["PA_CF1", "PA_ATM", "PA_ALT", "Rel. Humidity", "Temperature"]
            fig_m = px.imshow(
                mat,
                x=ts_slice,
                y=ch_names,
                color_continuous_scale=[[0.0, "#ff5555"], [1.0, "#00d2c4"]],
                zmin=0.0,
                zmax=1.0,
                aspect="auto",
                title=f"{bench_site} First {n_slice} Test Hours — Mask `{scenario_id}` (Teal=Observed 1, Rose=PM Missing 0)",
            )
            apply_dark_layout(fig_m, f"Phase 3C Channel Mask Matrix ({scenario_meta.get('name', scenario_id)} — {bench_site})", "Test Timestamp", "Telemetry Channel", height=320, hovermode="closest")
            st.plotly_chart(fig_m, width="stretch", config=PLOTLY_CHART_CONFIG)

            n_miss = int((~sub_b[mask_col]).sum())
            pct_miss = (n_miss / len(sub_b)) * 100.0
            st.caption(
                f"{bench_site} test split ({len(sub_b):,} hours): **{n_miss:,} Complete PM Dropout hours ({pct_miss:.2f}%)** "
                f"under `{scenario_id}`, with ambient meteorology (`RH`, `Temperature`) remaining observed."
            )

# =============================================================================
# TAB 3: DISTRIBUTIONS, SCATTER & CORRELATIONS
# =============================================================================
with tab_dist_corr:
    st.markdown("### Channel Distributions & Multi-City Comparisons")
    if total_obs > 0:
        dist_col = st.selectbox(
            "Inspect Channel Distribution:",
            options=["bam_reference", "pa_atm", "pa_cf1", "pa_alt", "relative_humidity", "temperature", "dew_point_raw"],
            index=0,
            key="de_dist_col",
        )
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            fig_hist = px.histogram(
                df_filtered,
                x=dist_col,
                color="site" if selected_site == "All Sites" else None,
                nbins=45,
                marginal="box",
                title=f"Distribution of `{dist_col}` ({selected_site})",
                color_discrete_sequence=[THEME["accent_cyan"], THEME["accent_teal"], THEME["accent_amber"]],
            )
            apply_dark_layout(fig_hist, f"Distribution — {dist_col}", dist_col, "Hour Count", height=400)
            st.plotly_chart(fig_hist, width="stretch", config=PLOTLY_CHART_CONFIG)
        with c_d2:
            fig_box = px.box(
                df_filtered,
                x="site",
                y=dist_col,
                color="site",
                title=f"Site-Wise Spread of `{dist_col}`",
                color_discrete_sequence=[THEME["accent_cyan"], THEME["accent_teal"], THEME["accent_amber"]],
            )
            apply_dark_layout(fig_box, f"Site Boxplot — {dist_col}", "Monitoring Station", dist_col, height=400, hovermode="closest")
            st.plotly_chart(fig_box, width="stretch", config=PLOTLY_CHART_CONFIG)

    st.markdown("### Raw Nephelometer vs. BAM-1020 Reference Agreement")
    if total_obs > 0:
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.plotly_chart(
                create_scatter_comparison(
                    df_filtered,
                    x_col="pa_atm",
                    y_col="bam_reference",
                    x_title="Raw PA_ATM (µg/m³)",
                    y_title="BAM Reference PM2.5 (µg/m³)",
                    color_col="site" if selected_site == "All Sites" else None,
                    title="PA_ATM vs. BAM-1020",
                ),
                width="stretch",
                config=PLOTLY_CHART_CONFIG,
            )
        with c_s2:
            st.plotly_chart(
                create_scatter_comparison(
                    df_filtered,
                    x_col="pa_cf1",
                    y_col="bam_reference",
                    x_title="Raw PA_CF1 (µg/m³)",
                    y_title="BAM Reference PM2.5 (µg/m³)",
                    color_col="site" if selected_site == "All Sites" else None,
                    title="PA_CF1 vs. BAM-1020 (Systematic Positive Bias)",
                ),
                width="stretch",
                config=PLOTLY_CHART_CONFIG,
            )

    if not df_bias_summary.empty:
        st.markdown("#### Multi-Channel Sensor Bias Audit Summary (`sensor_bias_summary.csv`, N=18,584)")
        st.dataframe(df_bias_summary, width="stretch", hide_index=True)

    st.markdown("### Pearson Correlation Analysis")
    corr_defaults = ["bam_reference", "pa_atm", "pa_cf1", "pa_alt", "relative_humidity", "temperature"]
    corr_choices = st.multiselect(
        "Select Variables for Pearson Correlation Heatmap:",
        options=corr_defaults + ["dew_point_raw", "time_diff_hours"],
        default=corr_defaults,
        key="de_corr_cols",
    )
    if total_obs > 0 and len(corr_choices) >= 2:
        st.plotly_chart(
            create_correlation_heatmap(
                df_filtered,
                corr_choices,
                title=f"Pearson Correlation Matrix — {selected_site} (N = {total_obs:,})",
            ),
            width="stretch",
            config=PLOTLY_CHART_CONFIG,
        )

# =============================================================================
# TAB 4: PSYCHROMETRIC AUDIT & RAW MEASUREMENTS
# =============================================================================
with tab_raw_audit:
    st.markdown("### Psychrometric Anomaly Audit (`dew_point_raw` Quarantine)")
    if not df_dew_audit.empty:
        st.dataframe(df_dew_audit, width="stretch", hide_index=True)
    st.warning(
        "**Empirical Finding:** `dew_point_raw` exceeds ambient dry-bulb temperature (`Td > T`) in **100.0% of records (18,584 / 18,584)** "
        "because raw PurpleAir dew point values were logged on a Fahrenheit scale (`mean ~62°F ≈ 16.7°C` vs Magnus-formula C). "
        "The raw field is preserved in `cleaned_data.parquet` for audit reproducibility and strictly excluded from all modeling pipelines.",
        icon="⚠️",
    )

    if not df_audit_stats.empty:
        st.markdown("#### Full-Archive Statistical Profile (`dataset_audit_report.csv`, N=18,584)")
        st.dataframe(df_audit_stats, width="stretch", hide_index=True)

    st.markdown("### Preserved Raw Excel Workbook Inspection (`Campmier_Manuscript_Data.xlsx`)")
    sheets = get_raw_sheet_names()
    chosen_sheet = st.selectbox("Select Original Workbook Sheet:", options=sheets, key="de_raw_sheet")
    raw_preview_df = load_raw_sheet(chosen_sheet, nrows=100)
    st.caption(f"Previewing first 100 unchanged rows from sheet `{chosen_sheet}` (Dryad CC0 1.0 raw deposit).")
    st.dataframe(raw_preview_df, width="stretch", hide_index=True)

    raw_full_df = load_raw_sheet(chosen_sheet)
    st.download_button(
        label=f"📥 Download Preserved Raw Sheet `{chosen_sheet}` ({len(raw_full_df):,} rows)",
        data=raw_full_df.to_csv(index=False).encode("utf-8"),
        file_name=f"airwise_raw_campmier_{chosen_sheet.lower()}.csv",
        mime="text/csv",
    )

# =============================================================================
# TAB 5: FILTERED CURATED DATA EXPORT
# =============================================================================
with tab_export:
    st.markdown("### Filtered Curated Observation Table (Un-Imputed Raw Measurements Preserved)")
    st.caption(
        f"Showing first 250 of **{total_obs:,}** curated records for `{selected_site}` "
        f"({start_date} to {end_date}). Null count in retained columns: {int(df_filtered.isna().sum().sum()) if total_obs > 0 else 0}."
    )
    st.dataframe(df_filtered.head(250), width="stretch", hide_index=True)
    if total_obs > 0:
        st.download_button(
            label=f"📥 Download Filtered Curated Data ({total_obs:,} rows)",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"airwise_curated_{selected_site.lower().replace(' ', '_')}_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
