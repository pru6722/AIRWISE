"""
AIRWISE Dashboard: Interactive Plotly Charting Library
Dark-themed, publication-grade Plotly visualizations with interactive hover tooltips,
zoom/pan/range selection, PNG download export, and strict preservation of research data.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import THEME, CPCB_COLORS, CPCB_BREAKPOINTS, PIPELINES, SCENARIOS

PLOTLY_CHART_CONFIG: Dict[str, Any] = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["autoScale2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "airwise_research_chart",
        "height": 700,
        "width": 1200,
        "scale": 2,
    },
}


def apply_dark_layout(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: Optional[int] = None,
    hovermode: str = "x unified",
) -> go.Figure:
    """Applies standardized dark academic theme styling to a Plotly figure."""
    layout_kwargs: Dict[str, Any] = dict(
        title=dict(
            text=title,
            font=dict(color=THEME["text_primary"], size=15, family="sans-serif"),
            x=0.01,
        ),
        paper_bgcolor=THEME["bg_main"],
        plot_bgcolor="#0e1626",
        font=dict(color=THEME["text_secondary"], size=12),
        margin=dict(l=55, r=30, t=55, b=50),
        xaxis=dict(
            title=x_title,
            gridcolor=THEME["border"],
            zerolinecolor=THEME["border"],
            showline=True,
            linecolor=THEME["border"],
        ),
        yaxis=dict(
            title=y_title,
            gridcolor=THEME["border"],
            zerolinecolor=THEME["border"],
            showline=True,
            linecolor=THEME["border"],
        ),
        legend=dict(
            bgcolor="rgba(19, 28, 46, 0.85)",
            bordercolor=THEME["border"],
            borderwidth=1,
            font=dict(color=THEME["text_primary"], size=11),
        ),
        hoverlabel=dict(
            bgcolor="#131c2e",
            bordercolor="#00d2c4",
            font=dict(color="#f8fafd", size=12),
        ),
        hovermode=hovermode,
    )
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    return fig


def _empty_state_figure(title: str, message: str = "No observations match the selected filters.") -> go.Figure:
    """Creates a clean dark-themed empty-state chart instead of failing on empty slices."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=THEME["text_secondary"], size=13),
    )
    return apply_dark_layout(fig, title, "", "", hovermode="closest")


def create_timeseries_chart(
    df: pd.DataFrame,
    y_cols: List[str],
    labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    title: str = "Time-Series Concentration Profile",
    y_title: str = "PM2.5 (µg/m³)",
    show_rangeslider: bool = False,
) -> go.Figure:
    """Creates a multi-line interactive time-series chart with optional range slider."""
    if df.empty or "timestamp" not in df.columns:
        return _empty_state_figure(title)

    fig = go.Figure()
    labels = labels or {}
    colors = colors or {}
    default_palette = [
        THEME["accent_teal"],
        THEME["accent_cyan"],
        THEME["accent_purple"],
        THEME["accent_amber"],
        THEME["accent_rose"],
    ]

    added = 0
    for i, col in enumerate(y_cols):
        if col in df.columns:
            name = labels.get(col, col)
            color = colors.get(col, default_palette[i % len(default_palette)])
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[col],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.6),
                    connectgaps=False,
                    hovertemplate=f"<b>{name}</b>: %{{y:.2f}}<extra></extra>",
                )
            )
            added += 1

    if added == 0:
        return _empty_state_figure(title, "Selected channels are unavailable in this slice.")

    apply_dark_layout(fig, title, "Timestamp", y_title)
    if show_rangeslider:
        fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.06)
    return fig


def create_scatter_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    color_col: Optional[str] = None,
    title: str = "Sensor vs Reference Scatter",
) -> go.Figure:
    """Creates an interactive scatter plot with 1:1 diagonal agreement reference."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return _empty_state_figure(title)

    valid_df = df.dropna(subset=[x_col, y_col])
    if valid_df.empty:
        return _empty_state_figure(title, "No valid paired observations in selected slice.")

    if color_col and color_col in valid_df.columns:
        fig = px.scatter(
            valid_df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_sequence=[THEME["accent_teal"], THEME["accent_cyan"], THEME["accent_amber"]],
            opacity=0.65,
            hover_data=[c for c in ["timestamp", "site"] if c in valid_df.columns],
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=valid_df[x_col],
                y=valid_df[y_col],
                mode="markers",
                marker=dict(color=THEME["accent_teal"], size=5, opacity=0.55),
                name="Observations",
            )
        )

    max_val = float(max(valid_df[x_col].max(), valid_df[y_col].max(), 50.0))
    fig.add_trace(
        go.Scatter(
            x=[0.0, max_val],
            y=[0.0, max_val],
            mode="lines",
            name="1:1 Perfect Agreement",
            line=dict(color=THEME["accent_rose"], width=1.6, dash="dash"),
        )
    )

    return apply_dark_layout(fig, title, x_title, y_title, hovermode="closest")


def create_uncertainty_ribbon_chart(
    df: pd.DataFrame,
    y_pred_col: str,
    lower_col: str,
    upper_col: str,
    ref_col: Optional[str] = "bam_reference",
    title: str = "Conformal Uncertainty Prediction Interval",
    coverage_label: str = "Conformal Interval (90%)",
) -> go.Figure:
    """Renders point predictions with shaded conformal prediction interval band."""
    if df.empty or y_pred_col not in df.columns or lower_col not in df.columns or upper_col not in df.columns:
        return _empty_state_figure(title)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[upper_col],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[lower_col],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 210, 196, 0.18)",
            name=coverage_label,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[y_pred_col],
            mode="lines",
            name="Model Prediction",
            line=dict(color=THEME["accent_teal"], width=2),
        )
    )

    if ref_col and ref_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[ref_col],
                mode="lines",
                name="BAM Reference Target",
                line=dict(color=THEME["accent_cyan"], width=1.5, dash="dot"),
            )
        )

    return apply_dark_layout(fig, title, "Timestamp", "PM2.5 Concentration (µg/m³)")


def create_cpcb_confusion_matrix(
    cm: np.ndarray,
    categories: List[str],
    title: str = "CPCB Category Confusion Matrix",
) -> go.Figure:
    """Creates an annotated confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        x=categories,
        y=categories,
        color_continuous_scale="Teal",
        labels=dict(x="Predicted CPCB Category", y="True Reference CPCB Category", color="Count"),
        text_auto=True,
    )
    apply_dark_layout(fig, title, "Predicted CPCB Category", "True Reference CPCB Category", hovermode="closest")
    fig.update_coloraxes(colorbar_tickfont=dict(color=THEME["text_secondary"]))
    return fig


# =============================================================================
# Phase 3C Reusable Comparative & Diagnostic Charts
# =============================================================================

def create_pipeline_comparison_bar(
    df_metrics: pd.DataFrame,
    metric_col: str = "rmse",
    scenario_id: Optional[str] = None,
    population_label: str = "Full Test Population",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Compares Pipelines A, B, C, and D for a single scenario or across all 9 scenarios.
    Preserves explicit Pipeline D eligibility annotations when coverage < 100%.
    """
    if df_metrics.empty or metric_col not in df_metrics.columns:
        return _empty_state_figure(title or "Four-Pipeline Comparison")

    metric_label_map = {
        "rmse": "RMSE (µg/m³)",
        "mae": "MAE (µg/m³)",
        "r2": "R² Score",
        "mbe": "Mean Bias Error (µg/m³)",
    }
    y_label = metric_label_map.get(metric_col, metric_col.upper())

    df_plot = df_metrics.copy()
    if scenario_id:
        df_plot = df_plot[df_plot["scenario_id"] == scenario_id]
        sc_name = SCENARIOS.get(scenario_id, {}).get("name", scenario_id)
        chart_title = title or f"Four-Pipeline {y_label} Comparison — {sc_name} ({population_label})"
    else:
        chart_title = title or f"Four-Pipeline {y_label} Across Missingness Regimes ({population_label})"

    if df_plot.empty:
        return _empty_state_figure(chart_title)

    fig = go.Figure()
    for code in ["A", "B", "C", "D"]:
        sub = df_plot[df_plot["pipeline_code"] == code].copy()
        if sub.empty:
            continue
        p_info = PIPELINES.get(code, {})
        p_name = p_info.get("short_name", f"Pipeline {code}")
        p_color = p_info.get("color", THEME["accent_teal"])

        if scenario_id:
            x_vals = [p_name]
            y_vals = sub[metric_col].tolist()
        else:
            x_vals = [SCENARIOS.get(sid, {}).get("name", sid) for sid in sub["scenario_id"]]
            y_vals = sub[metric_col].tolist()

        cov_vals = sub["coverage_pct"].tolist() if "coverage_pct" in sub.columns else [100.0] * len(sub)
        n_vals = sub["n_samples"].tolist() if "n_samples" in sub.columns else [0] * len(sub)

        custom_hover = [
            f"<b>{p_name}</b><br>{y_label}: {y:.2f}<br>N Evaluated: {n:,}<br>Coverage: {cov:.1f}%"
            if pd.notna(y)
            else f"<b>{p_name}</b><br>No predictions on missing intervals (N={n:,})"
            for y, n, cov in zip(y_vals, n_vals, cov_vals)
        ]

        text_vals = [
            f"{y:.2f}" if pd.notna(y) else "N/A"
            for y in y_vals
        ]

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=[y if pd.notna(y) else 0.0 for y in y_vals],
                name=p_name,
                marker_color=p_color,
                text=text_vals if scenario_id else None,
                textposition="outside" if scenario_id else None,
                hovertext=custom_hover,
                hoverinfo="text",
            )
        )

    x_title = "Pipeline Architecture" if scenario_id else "Missingness Scenario"
    apply_dark_layout(fig, chart_title, x_title, y_label, hovermode="closest")
    fig.update_layout(barmode="group")
    return fig


def create_outage_degradation_chart(
    df_metrics: pd.DataFrame,
    metric_col: str = "rmse",
    population_label: str = "Missing-PM Subgroup",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plots error trajectories across nominal block-generation durations (1h, 3h, 6h, 12h, 24h).
    Explicitly labels nominal block-generation parameter vs merged continuous outages.
    """
    block_order = ["Block_1h", "Block_3h", "Block_6h", "Block_12h", "Block_24h"]
    block_labels = {
        "Block_1h": "1h (Nominal)",
        "Block_3h": "3h (Nominal)",
        "Block_6h": "6h (Nominal)",
        "Block_12h": "12h (Nominal)",
        "Block_24h": "24h (Nominal, up to 63h merged)",
    }
    chart_title = title or f"Outage Duration Degradation Trajectory ({population_label})"

    if df_metrics.empty or metric_col not in df_metrics.columns:
        return _empty_state_figure(chart_title)

    sub_df = df_metrics[df_metrics["scenario_id"].isin(block_order)].copy()
    if sub_df.empty:
        return _empty_state_figure(chart_title)

    sub_df["order_idx"] = sub_df["scenario_id"].map({sid: i for i, sid in enumerate(block_order)})
    sub_df = sub_df.sort_values("order_idx")

    fig = go.Figure()
    for code in ["A", "B", "C", "D"]:
        p_df = sub_df[sub_df["pipeline_code"] == code].dropna(subset=[metric_col])
        if p_df.empty:
            continue
        p_info = PIPELINES.get(code, {})
        fig.add_trace(
            go.Scatter(
                x=[block_labels[sid] for sid in p_df["scenario_id"]],
                y=p_df[metric_col],
                mode="lines+markers",
                name=p_info.get("short_name", f"Pipeline {code}"),
                line=dict(color=p_info.get("color", THEME["accent_teal"]), width=2.4),
                marker=dict(size=8),
                hovertemplate=(
                    f"<b>{p_info.get('short_name', code)}</b><br>"
                    "Duration: %{x}<br>"
                    f"{metric_col.upper()}: %{{y:.2f}}<extra></extra>"
                ),
            )
        )

    y_label = "RMSE (µg/m³)" if metric_col == "rmse" else ("MAE (µg/m³)" if metric_col == "mae" else "R²")
    return apply_dark_layout(fig, chart_title, "Nominal Block-Generation Duration", y_label, hovermode="x unified")


def create_residual_distribution_chart(
    df: pd.DataFrame,
    ref_col: str,
    pred_col: str,
    pipeline_name: str = "Model",
    title: Optional[str] = None,
) -> go.Figure:
    """Renders residual histogram (Prediction - Reference) with zero-bias and mean bias markers."""
    chart_title = title or f"Residual Error Distribution — {pipeline_name}"
    if df.empty or ref_col not in df.columns or pred_col not in df.columns:
        return _empty_state_figure(chart_title)

    valid = df[[ref_col, pred_col]].dropna()
    if valid.empty:
        return _empty_state_figure(chart_title, "No valid prediction-reference pairs in slice.")

    residuals = valid[pred_col] - valid[ref_col]
    mean_res = float(residuals.mean())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=50,
            name="Residuals (Pred - BAM)",
            marker=dict(color=THEME["accent_teal"], line=dict(color=THEME["bg_main"], width=0.5)),
            opacity=0.8,
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color=THEME["text_primary"], annotation_text="Zero Bias (0.0)")
    fig.add_vline(
        x=mean_res,
        line_dash="dot",
        line_color=THEME["accent_amber"],
        annotation_text=f"Mean Bias: {mean_res:+.2f}",
        annotation_position="top left" if mean_res < 0 else "top right",
    )

    return apply_dark_layout(fig, chart_title, "Residual Error (Predicted - BAM, µg/m³)", "Observation Count", hovermode="x")


def create_residual_vs_reference_chart(
    df: pd.DataFrame,
    ref_col: str,
    pred_col: str,
    color_col: Optional[str] = "site",
    title: str = "Residuals vs. Ground-Truth BAM Concentration",
) -> go.Figure:
    """Plots residual dispersion against BAM concentrations to diagnose high-PM bias/saturation."""
    if df.empty or ref_col not in df.columns or pred_col not in df.columns:
        return _empty_state_figure(title)

    valid = df.dropna(subset=[ref_col, pred_col]).copy()
    if valid.empty:
        return _empty_state_figure(title, "No valid predictions to plot residuals.")

    valid["_residual"] = valid[pred_col] - valid[ref_col]

    if color_col and color_col in valid.columns:
        fig = px.scatter(
            valid,
            x=ref_col,
            y="_residual",
            color=color_col,
            color_discrete_sequence=[THEME["accent_teal"], THEME["accent_cyan"], THEME["accent_amber"]],
            opacity=0.6,
            hover_data=[c for c in ["timestamp", "site"] if c in valid.columns],
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=valid[ref_col],
                y=valid["_residual"],
                mode="markers",
                marker=dict(color=THEME["accent_teal"], size=5, opacity=0.55),
                name="Residuals",
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_color=THEME["accent_rose"], annotation_text="Zero Error")
    return apply_dark_layout(
        fig,
        title,
        "BAM Reference Concentration (µg/m³)",
        "Prediction Residual (Pred - BAM, µg/m³)",
        hovermode="closest",
    )


def create_bootstrap_forest_chart(
    df_boot: pd.DataFrame,
    mode: str = "scenarios",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Visualizes paired moving-block bootstrap 95% confidence intervals separately from
    empirical uncentered bootstrap tail proportions (never labeled as p-values).
    mode='scenarios': Forest plot across benchmark scenarios (canonical b=24h).
    mode='sensitivity': Block-length sensitivity sweep across b in {12, 24, 48, 72} hours.
    """
    if df_boot.empty:
        return _empty_state_figure(title or "Bootstrap Confidence Intervals")

    if mode == "sensitivity":
        chart_title = title or "Moving-Block Bootstrap Sensitivity Sweep (b ∈ {12, 24, 48, 72}h)"
        fig = go.Figure()
        scenarios_in_df = df_boot["scenario_id"].unique().tolist()
        palette = [THEME["accent_teal"], THEME["accent_cyan"], THEME["accent_amber"]]

        for i, sid in enumerate(scenarios_in_df):
            sub = df_boot[df_boot["scenario_id"] == sid].sort_values("block_length_hours")
            sc_label = SCENARIOS.get(sid, {}).get("name", sid)
            color = palette[i % len(palette)]
            tail_col = "p_value" if "p_value" in sub.columns else "p_value_rmse"

            # Upper/lower CI band without assuming point estimate sits inside percentile CI
            fig.add_trace(
                go.Scatter(
                    x=sub["block_length_hours"],
                    y=sub["ci_upper_95"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["block_length_hours"],
                    y=sub["ci_lower_95"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba(0, 210, 196, 0.12)" if i == 0 else "rgba(0, 168, 255, 0.10)",
                    name=f"{sc_label} (95% CI Band)",
                )
            )
            hover_text = [
                f"<b>{sc_label} (b={b}h)</b><br>"
                f"Point ΔRMSE: {d:+.3f} µg/m³<br>"
                f"95% CI: [{lo:+.3f}, {hi:+.3f}]<br>"
                f"Empirical Tail Proportion (p̂_tail): {pt:.3f}"
                for b, d, lo, hi, pt in zip(
                    sub["block_length_hours"],
                    sub["delta_rmse"],
                    sub["ci_lower_95"],
                    sub["ci_upper_95"],
                    sub[tail_col] if tail_col in sub.columns else [np.nan] * len(sub),
                )
            ]
            fig.add_trace(
                go.Scatter(
                    x=sub["block_length_hours"],
                    y=sub["delta_rmse"],
                    mode="lines+markers",
                    name=f"{sc_label} (Point ΔRMSE)",
                    line=dict(color=color, width=2.2),
                    marker=dict(size=7),
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            )

        fig.add_hline(y=0.0, line_dash="dash", line_color=THEME["accent_rose"], annotation_text="Zero Difference (ΔRMSE = 0)")
        apply_dark_layout(fig, chart_title, "Bootstrap Block Length (Hours)", "ΔRMSE: Joint TCN v2 (C) minus Baseline (µg/m³)", hovermode="closest")
        fig.update_xaxes(tickvals=[12, 24, 48, 72])
        return fig

    # Forest plot for canonical scenario comparisons
    chart_title = title or "Paired Moving-Block Bootstrap 95% Confidence Intervals (b = 24h)"
    df_f = df_boot.copy().reset_index(drop=True)
    df_f["label"] = [
        f"{SCENARIOS.get(r['scenario_id'], {}).get('name', r['scenario_id'])} | {r['comparison']}"
        for _, r in df_f.iterrows()
    ]
    tail_col = "p_value_rmse" if "p_value_rmse" in df_f.columns else "p_value"

    fig = go.Figure()
    for idx, row in df_f.iterrows():
        excludes_zero = (row["ci_lower_95"] > 0) or (row["ci_upper_95"] < 0)
        seg_color = THEME["accent_amber"] if excludes_zero else THEME["accent_teal"]
        fig.add_trace(
            go.Scatter(
                x=[row["ci_lower_95"], row["ci_upper_95"]],
                y=[row["label"], row["label"]],
                mode="lines",
                line=dict(color=seg_color, width=3.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    hover_points = [
        f"<b>{r['label']}</b><br>"
        f"Point ΔRMSE: {r['delta_rmse']:+.3f} µg/m³<br>"
        f"95% Paired CI: [{r['ci_lower_95']:+.3f}, {r['ci_upper_95']:+.3f}]<br>"
        f"95% CI Excludes Zero: {bool((r['ci_lower_95'] > 0) or (r['ci_upper_95'] < 0))}<br>"
        f"Empirical Tail Proportion (p̂_tail): {r[tail_col]:.3f}"
        for _, r in df_f.iterrows()
    ]
    fig.add_trace(
        go.Scatter(
            x=df_f["delta_rmse"],
            y=df_f["label"],
            mode="markers",
            name="Point Estimate ΔRMSE",
            marker=dict(color=THEME["text_primary"], size=8, symbol="diamond", line=dict(color=THEME["accent_cyan"], width=1.5)),
            hovertext=hover_points,
            hoverinfo="text",
        )
    )

    fig.add_vline(x=0.0, line_dash="dash", line_color=THEME["accent_rose"], annotation_text="Parity (ΔRMSE = 0)")
    fig_height = max(420, 26 * len(df_f) + 120)
    return apply_dark_layout(
        fig,
        chart_title,
        "ΔRMSE (µg/m³) [Negative favors Joint TCN v2; Positive favors Baseline]",
        "",
        height=fig_height,
        hovermode="closest",
    )


def create_training_loss_chart(
    history: List[Dict[str, Any]],
    best_epoch: int = 6,
    title: str = "Joint TCN v2 Training & Validation Convergence (Multi-Task Huber Loss)",
) -> go.Figure:
    """Plots epoch-by-epoch training and validation loss curves from joint_tcn_v2_metadata.json."""
    if not history:
        return _empty_state_figure(title, "Training history logs unavailable.")

    hist_df = pd.DataFrame(history)
    fig = go.Figure()

    traces = [
        ("train_loss", "Train Total Loss", THEME["accent_cyan"], "solid"),
        ("val_loss", "Validation Total Loss", THEME["accent_teal"], "solid"),
        ("val_cal_loss", "Val Calibration Head Loss", THEME["accent_purple"], "dot"),
        ("val_rec_loss", "Val Reconstruction Head Loss", THEME["accent_amber"], "dot"),
    ]
    for col, label, color, dash in traces:
        if col in hist_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist_df["epoch"],
                    y=hist_df[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2.0, dash=dash),
                    marker=dict(size=6),
                )
            )

    fig.add_vline(
        x=best_epoch,
        line_dash="dash",
        line_color=THEME["accent_rose"],
        annotation_text=f"Best Checkpoint (Epoch {best_epoch})",
        annotation_position="top right",
    )
    return apply_dark_layout(fig, title, "Training Epoch", "Multi-Task Huber Loss")


def create_site_stratified_chart(
    df_site: pd.DataFrame,
    metric_col: str = "rmse",
    scenario_id: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Creates grouped bar chart stratified across Bangalore, Delhi, and Hamirpur."""
    if df_site.empty or metric_col not in df_site.columns:
        return _empty_state_figure(title or "Site-Stratified Performance")

    df_plot = df_site.copy()
    if scenario_id:
        df_plot = df_plot[df_plot["scenario_id"] == scenario_id]
        sc_name = SCENARIOS.get(scenario_id, {}).get("name", scenario_id)
        chart_title = title or f"Site-Stratified {metric_col.upper()} — {sc_name}"
    else:
        chart_title = title or f"Site-Stratified {metric_col.upper()} Across Cities"

    if df_plot.empty:
        return _empty_state_figure(chart_title)

    fig = go.Figure()
    for code in ["A", "B", "C", "D"]:
        sub = df_plot[df_plot["pipeline_code"] == code]
        if sub.empty:
            continue
        p_info = PIPELINES.get(code, {})
        # If multiple scenarios remain, aggregate mean across scenarios for overview
        grouped = sub.groupby("site", as_index=False)[metric_col].mean()
        fig.add_trace(
            go.Bar(
                x=grouped["site"],
                y=grouped[metric_col],
                name=p_info.get("short_name", f"Pipeline {code}"),
                marker_color=p_info.get("color", THEME["accent_teal"]),
                text=[f"{v:.2f}" for v in grouped[metric_col]],
                textposition="outside",
            )
        )

    apply_dark_layout(
        fig,
        chart_title,
        "Monitoring Station",
        "RMSE (µg/m³)" if metric_col == "rmse" else metric_col.upper(),
        hovermode="closest",
    )
    fig.update_layout(barmode="group")
    return fig


def create_channel_availability_chart(
    df: pd.DataFrame,
    channels: Optional[List[str]] = None,
    title: str = "Independent Channel Availability Profile",
) -> go.Figure:
    """Computes and plots independent observation availability % for each sensor and meteorological channel."""
    target_channels = channels or ["pa_cf1", "pa_atm", "pa_alt", "relative_humidity", "temperature", "bam_reference"]
    present_channels = [c for c in target_channels if c in df.columns]

    if df.empty or not present_channels:
        return _empty_state_figure(title)

    n_total = len(df)
    rows = []
    label_map = {
        "pa_cf1": "PurpleAir CF1",
        "pa_atm": "PurpleAir ATM",
        "pa_alt": "PurpleAir ALT",
        "relative_humidity": "Relative Humidity (%)",
        "temperature": "Temperature (°C)",
        "bam_reference": "BAM-1020 Reference",
    }

    for c in present_channels:
        non_null = int(df[c].notna().sum())
        pct = (non_null / n_total) * 100.0
        rows.append({
            "channel": label_map.get(c, c),
            "availability_pct": pct,
            "missing_count": n_total - non_null,
            "observed_count": non_null,
        })

    avail_df = pd.DataFrame(rows)
    fig = go.Figure(
        go.Bar(
            x=avail_df["availability_pct"],
            y=avail_df["channel"],
            orientation="h",
            marker_color=THEME["accent_teal"],
            text=[f"{p:.2f}% ({obs:,}/{n_total:,})" for p, obs in zip(avail_df["availability_pct"], avail_df["observed_count"])],
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>Available: %{x:.2f}%<extra></extra>",
        )
    )
    apply_dark_layout(fig, title, "Observed Availability (%)", "Channel", hovermode="y unified")
    fig.update_xaxes(range=[0, 105])
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    cols: List[str],
    title: str = "Sensor & Meteorological Pearson Correlation Matrix",
) -> go.Figure:
    """Creates an annotated dark-mode correlation heatmap across selected columns."""
    valid_cols = [c for c in cols if c in df.columns]
    if df.empty or len(valid_cols) < 2:
        return _empty_state_figure(title, "Insufficient numeric columns for correlation.")

    corr = df[valid_cols].corr().round(3)
    fig = px.imshow(
        corr,
        x=valid_cols,
        y=valid_cols,
        color_continuous_scale="Tealrose",
        zmin=-1.0,
        zmax=1.0,
        text_auto=True,
    )
    apply_dark_layout(fig, title, "", "", hovermode="closest")
    fig.update_coloraxes(colorbar_tickfont=dict(color=THEME["text_secondary"]))
    return fig


def create_storage_comparison_chart(
    df_storage: pd.DataFrame,
    title: str = "Columnar Storage Footprint Comparison (CSV vs Parquet vs ORC)",
) -> go.Figure:
    """Creates an interactive format size comparison bar chart with reduction annotations."""
    if df_storage.empty or "format" not in df_storage.columns or "size_mb" not in df_storage.columns:
        return _empty_state_figure(title)

    color_map = {
        "CSV": THEME["accent_rose"],
        "PARQUET": THEME["accent_cyan"],
        "ORC": THEME["accent_teal"],
    }
    colors = [color_map.get(str(fmt).upper(), THEME["accent_purple"]) for fmt in df_storage["format"]]

    fig = go.Figure(
        go.Bar(
            x=df_storage["format"],
            y=df_storage["size_mb"],
            marker_color=colors,
            text=[f"{sz:.3f} MB" for sz in df_storage["size_mb"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Disk Footprint: %{y:.3f} MB<extra></extra>",
        )
    )
    return apply_dark_layout(fig, title, "Storage Format", "Serialized Size (MB)", hovermode="closest")
