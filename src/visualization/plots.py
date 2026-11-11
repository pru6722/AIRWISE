"""
Module: src.visualization.plots
Generates publication-quality figures for dataset audit, sensor bias, physical consistency,
and temporal continuity.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


# Publication style settings (Nature / AMT style)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_raw_vs_bam_timeseries(
    df: pd.DataFrame,
    site_name: str,
    output_path: Union[str, Path],
    window_days: int = 14,
    start_date: Optional[str] = None,
) -> Path:
    """
    Plots a high-resolution time-series trace comparing raw PurpleAir channels
    (PA_CF1, PA_ATM, PA_ALT) with the reference BAM.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    site_df = df[df["site"] == site_name].sort_values("timestamp").reset_index(drop=True)

    if start_date is not None:
        mask = site_df["timestamp"] >= pd.to_datetime(start_date)
        plot_df = site_df[mask].head(window_days * 24)
    else:
        # Default to a representative 14-day window in the middle
        mid_idx = len(site_df) // 2
        plot_df = site_df.iloc[mid_idx : mid_idx + (window_days * 24)]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(plot_df["timestamp"], plot_df["bam_reference"], label="Reference BAM (Ground Truth)", color="#111111", linewidth=2.0, zorder=5)
    ax.plot(plot_df["timestamp"], plot_df["pa_cf1"], label="PurpleAir CF=1", color="#D95F02", alpha=0.85, linestyle="-")
    ax.plot(plot_df["timestamp"], plot_df["pa_atm"], label="PurpleAir ATM", color="#7570B3", alpha=0.85, linestyle="--")
    ax.plot(plot_df["timestamp"], plot_df["pa_alt"], label="PurpleAir ALT", color="#1B9E77", alpha=0.85, linestyle=":")

    ax.set_title(f"Raw Low-Cost Sensor Channels vs. Reference BAM — {site_name} (Representative 14-Day Window)")
    ax.set_xlabel("Date and Time (IST)")
    ax.set_ylabel(r"$\mathrm{PM}_{2.5}$ Concentration ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_sensor_vs_bam_scatter(
    site_dfs: Dict[str, pd.DataFrame],
    output_path: Union[str, Path],
) -> Path:
    """
    Multi-panel scatter plot of PA_CF1, PA_ATM, PA_ALT vs BAM across all three sites.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sites = list(site_dfs.keys())
    channels = [("pa_cf1", "PA_CF1", "#D95F02"), ("pa_atm", "PA_ATM", "#7570B3"), ("pa_alt", "PA_ALT", "#1B9E77")]

    fig, axes = plt.subplots(len(sites), 3, figsize=(15, 4.5 * len(sites)), sharex="row", sharey="row")

    for row_idx, site in enumerate(sites):
        df = site_dfs[site]
        bam = df["bam_reference"].values
        max_val = max(bam.max(), df["pa_cf1"].max(), df["pa_atm"].max()) * 1.05

        for col_idx, (ch_col, ch_label, color) in enumerate(channels):
            ax = axes[row_idx, col_idx]
            ch_vals = df[ch_col].values

            # Scatter
            ax.scatter(bam, ch_vals, color=color, alpha=0.25, s=12, edgecolors="none")

            # 1:1 Identity line
            ax.plot([0, max_val], [0, max_val], color="#333333", linestyle="--", linewidth=1.5, label="1:1 Identity")

            # Linear regression line
            slope, intercept = np.polyfit(bam, ch_vals, 1)
            corr = np.corrcoef(bam, ch_vals)[0, 1]
            ax.plot([0, max_val], [intercept, slope * max_val + intercept], color="#D73027", linewidth=1.8, label=f"Fit: y={slope:.2f}x+{intercept:.1f}")

            mae = np.mean(np.abs(ch_vals - bam))
            rmse = np.sqrt(np.mean((ch_vals - bam) ** 2))
            ax.text(
                0.05, 0.92,
                f"$R^2$ = {corr**2:.3f}\nMAE = {mae:.1f} $\\mu$g/m$^3$\nRMSE = {rmse:.1f} $\\mu$g/m$^3$",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
                fontsize=8.5,
            )

            ax.set_title(f"{site} — {ch_label} vs. BAM")
            if row_idx == len(sites) - 1:
                ax.set_xlabel(r"BAM Reference ($\mu\mathrm{g}/\mathrm{m}^3$)")
            if col_idx == 0:
                ax.set_ylabel(r"PurpleAir Reading ($\mu\mathrm{g}/\mathrm{m}^3$)")
            ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("Collocated PurpleAir PM2.5 Channels vs. Reference BAM Across Study Sites", y=1.005, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_dew_point_physics_investigation(
    site_dfs: Dict[str, pd.DataFrame],
    output_path: Union[str, Path],
) -> Path:
    """
    Plots temperature vs dew point, demonstrating the physical violation (Td > T) in Celsius
    and the Fahrenheit unit-consistency hypothesis.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    sites = list(site_dfs.keys())

    for idx, site in enumerate(sites):
        ax = axes[idx]
        df = site_dfs[site]
        t = df["temperature"].values
        td_raw = df["dew_point_raw"].values
        td_converted = (td_raw - 32.0) * (5.0 / 9.0)

        # Plot raw dew point vs temperature
        ax.scatter(t, td_raw, color="#E41A1C", alpha=0.3, s=14, label=r"Raw $T_d$ (Labeled as °C)")
        # Plot converted dew point vs temperature
        ax.scatter(t, td_converted, color="#377EB8", alpha=0.3, s=14, label=r"Converted $(T_d - 32)\times\frac{5}{9}$")

        # 1:1 Physical saturation boundary line: T_d = T
        t_range = np.linspace(5, 50, 100)
        ax.plot(t_range, t_range, color="#111111", linestyle="--", linewidth=1.8, label=r"Physical Limit $T_d = T$")

        # Shaded physically impossible region (Td > T)
        ax.fill_between(t_range, t_range, 85, color="#FEE0D2", alpha=0.4, label="Physically Impossible Region ($T_d > T$)")

        ax.set_title(f"{site}: Dew Point vs. Temperature")
        ax.set_xlabel("Ambient Temperature (°C)")
        if idx == 0:
            ax.set_ylabel("Dew Point Value")
        ax.set_xlim(5, 50)
        ax.set_ylim(0, 85)
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Physical Consistency Audit: Verification of PA_DewPoint Unit Inconsistency", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_temporal_coverage_and_gaps(
    site_dfs: Dict[str, pd.DataFrame],
    output_path: Union[str, Path],
) -> Path:
    """
    Plots the timeline coverage of observations and distribution of time gap durations.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 1.2]})

    # Top: Timeline presence stripes
    site_names = list(site_dfs.keys())
    y_ticks = []
    y_labels = []

    for i, site in enumerate(site_names):
        df = site_dfs[site].sort_values("timestamp")
        t = df["timestamp"]
        y_pos = i
        y_ticks.append(y_pos)
        y_labels.append(site)
        ax1.scatter(t, [y_pos] * len(t), s=1.5, color=f"C{i}", alpha=0.6, marker="|")

    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.set_title("Temporal Observation Coverage Across Measurement Campaigns (2018–2021)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.set_ylabel("Study Site")

    # Bottom: Gap distribution by site
    gap_data = []
    for site in site_names:
        df = site_dfs[site].sort_values("timestamp")
        diff_h = (df["timestamp"].diff().dt.total_seconds() / 3600.0).dropna()
        gaps = diff_h[diff_h > 1.0] - 1.0  # missing hours
        for g in gaps:
            gap_data.append({"site": site, "missing_hours": g})

    gap_df = pd.DataFrame(gap_data)
    # Clip for plotting histogram up to 48 hours
    gap_df["missing_hours_clipped"] = np.clip(gap_df["missing_hours"], 1, 48)

    sns.histplot(
        data=gap_df,
        x="missing_hours_clipped",
        hue="site",
        multiple="dodge",
        bins=np.arange(0.5, 49.5, 2),
        ax=ax2,
        palette="Set2",
    )
    ax2.set_title("Distribution of Empirical Missing Hourly Gaps (Gaps ≤ 48 Hours)")
    ax2.set_xlabel("Consecutive Missing Hours in Gap Event")
    ax2.set_ylabel("Number of Gap Events")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_site_distributions(
    combined_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Boxplot / violin plot comparing PM2.5, Relative Humidity, and Temperature distributions
    across Delhi, Hamirpur, and Bangalore.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # PM2.5 distribution
    sns.boxplot(data=combined_df, x="site", y="bam_reference", ax=axes[0], palette="Blues_d", showfliers=False)
    axes[0].set_title(r"Reference $\mathrm{PM}_{2.5}$ Distribution")
    axes[0].set_ylabel(r"BAM $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$)")
    axes[0].set_xlabel("Site")

    # Relative Humidity distribution
    sns.boxplot(data=combined_df, x="site", y="relative_humidity", ax=axes[1], palette="Greens_d", showfliers=False)
    axes[1].set_title("Relative Humidity Distribution")
    axes[1].set_ylabel("Relative Humidity (%)")
    axes[1].set_xlabel("Site")

    # Temperature distribution
    sns.boxplot(data=combined_df, x="site", y="temperature", ax=axes[2], palette="Reds_d", showfliers=False)
    axes[2].set_title("Temperature Distribution")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Site")

    plt.suptitle("Cross-Site Environmental Regimes: Baseline Meteorology and PM2.5 Levels", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# =====================================================================
# CHECKPOINT 2: CALIBRATION & RECONSTRUCTION PLOTTING ROUTINES
# =====================================================================

def plot_calibrated_vs_bam(
    test_df: pd.DataFrame,
    calibrated_pm25: np.ndarray,
    output_path: Union[str, Path],
) -> Path:
    """
    Plots multi-panel scatter plots of Calibrated PM2.5 vs Reference BAM across all test sites.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = test_df.copy()
    df_plot["calibrated_pm25"] = calibrated_pm25
    sites = df_plot["site"].unique()

    fig, axes = plt.subplots(1, len(sites), figsize=(5.5 * len(sites), 5), sharey=False)
    if len(sites) == 1:
        axes = [axes]

    for idx, site in enumerate(sites):
        ax = axes[idx]
        sub = df_plot[df_plot["site"] == site]
        bam = sub["bam_reference"].values
        cal = sub["calibrated_pm25"].values
        max_val = max(bam.max(), cal.max()) * 1.05

        ax.scatter(bam, cal, color="#2B83BA", alpha=0.35, s=15, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], color="#333333", linestyle="--", linewidth=1.5, label="1:1 Identity")

        # Linear fit line
        slope, intercept = np.polyfit(bam, cal, 1)
        r2 = np.corrcoef(bam, cal)[0, 1] ** 2
        mae = np.mean(np.abs(cal - bam))
        rmse = np.sqrt(np.mean((cal - bam) ** 2))

        ax.plot([0, max_val], [intercept, slope * max_val + intercept], color="#D7191C", linewidth=1.8, label=f"Fit: y={slope:.2f}x+{intercept:.1f}")
        ax.text(
            0.05, 0.92,
            f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f} $\\mu$g/m$^3$\nRMSE = {rmse:.1f} $\\mu$g/m$^3$",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
            fontsize=9,
        )

        ax.set_title(f"{site} — Calibrated vs. BAM (Test Set)")
        ax.set_xlabel(r"Reference BAM $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$)")
        if idx == 0:
            ax.set_ylabel(r"Calibrated $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$)")
        ax.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Calibrated PurpleAir PM2.5 Estimates vs. Reference BAM across Test Partitions", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_calibration_comparison(
    calibration_results_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Bar chart comparing Test RMSE across 3 algorithms and 4 feature subsets (Pooled paradigm).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pooled_df = calibration_results_df[calibration_results_df["paradigm"] == "Pooled"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Test RMSE
    sns.barplot(
        data=pooled_df,
        x="feature_set",
        y="test_rmse",
        hue="algorithm",
        palette="Blues_d",
        ax=ax1,
    )
    ax1.set_title("Calibration Test RMSE by Algorithm and Feature Subset")
    ax1.set_ylabel(r"Test RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax1.set_xlabel("Feature Configuration")
    ax1.tick_params(axis="x", rotation=15)
    ax1.legend(title="Algorithm", loc="upper right")

    # Right: Test R^2
    sns.barplot(
        data=pooled_df,
        x="feature_set",
        y="test_r2",
        hue="algorithm",
        palette="Greens_d",
        ax=ax2,
    )
    ax2.set_title(r"Calibration Test $R^2$ by Algorithm and Feature Subset")
    ax2.set_ylabel(r"Test $R^2$")
    ax2.set_xlabel("Feature Configuration")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(title="Algorithm", loc="lower right")

    plt.suptitle("Comparative Evaluation of Calibration Algorithms (Pooled Training)", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_reconstruction_comparison(
    recon_metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Bar chart comparing MAE and RMSE across the 5 reconstruction baselines on hidden positions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    plot_df = recon_metrics_df.sort_values("hidden_rmse").reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, plot_df["hidden_rmse"], width, label="RMSE", color="#2B83BA")
    rects2 = ax.bar(x + width/2, plot_df["hidden_mae"], width, label="MAE", color="#FDAE61")

    ax.set_ylabel(r"Error ($\mu\mathrm{g}/\mathrm{m}^3$) on Hidden Observations")
    ax.set_title("Missing-Data Reconstruction Baseline Performance (Test Set, 20% Masked)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=15, ha="right")
    ax.legend()

    # Label bars with values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_actual_vs_reconstructed_timeseries(
    df: pd.DataFrame,
    bam_col: str,
    calib_col: str,
    recon_col: str,
    mask: np.ndarray,
    site_name: str,
    output_path: Union[str, Path],
    window_days: int = 10,
) -> Path:
    """
    Plots high-resolution time series of Actual BAM vs Calibrated vs Reconstructed PM2.5,
    highlighting artificially missing outage periods in shaded red.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = df[df["site"] == site_name].sort_values("timestamp").reset_index(drop=True)
    mask_sub = mask[df["site"] == site_name]

    # Select representative 10-day window
    n_points = min(len(sub), window_days * 24)
    sub = sub.iloc[:n_points].copy()
    m_win = mask_sub[:n_points]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(sub["timestamp"], sub[bam_col], label="Reference BAM (True PM2.5)", color="#111111", linewidth=2.2, zorder=5)
    ax.plot(sub["timestamp"], sub[calib_col], label="Stage-1 Calibrated Sensor", color="#7570B3", linestyle="--", alpha=0.7)
    ax.plot(sub["timestamp"], sub[recon_col], label="Stage-2 Reconstructed PM2.5", color="#D95F02", linewidth=1.8, linestyle="-", zorder=6)

    # Highlight masked outage blocks
    is_missing = ~m_win
    if is_missing.any():
        ax.fill_between(
            sub["timestamp"], 0, max(sub[bam_col].max(), sub[recon_col].max()) * 1.1,
            where=is_missing, color="#FEE0D2", alpha=0.5, label="Simulated Sensor Outage (Hidden)", zorder=2
        )

    ax.set_title(f"Actual vs. Reconstructed PM2.5 under Simulated Outages — {site_name} (Test Partition)")
    ax.set_xlabel("Timestamp (IST)")
    ax.set_ylabel(r"$\mathrm{PM}_{2.5}$ Concentration ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_error_vs_gap_length(
    propagation_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Plots RMSE and Error Growth G(g) as a function of consecutive missing gap length (1, 3, 6, 12, 24h).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: RMSE vs Gap Length
    for model_name, grp in propagation_df.groupby("model"):
        ax1.plot(grp["gap_hours"], grp["rmse"], marker="o", linewidth=2.0, label=model_name)
    ax1.set_title("Reconstruction RMSE vs. Outage Gap Duration")
    ax1.set_xlabel("Consecutive Outage Duration (Hours)")
    ax1.set_ylabel(r"Test RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax1.set_xticks(sorted(propagation_df["gap_hours"].unique()))
    ax1.legend(loc="upper left")

    # Right: Error Growth G(g) = RMSE(g) - RMSE(g_0)
    for model_name, grp in propagation_df.groupby("model"):
        ax2.plot(grp["gap_hours"], grp["error_growth_rmse"], marker="s", linewidth=2.0, linestyle="--", label=model_name)
    ax2.set_title(r"Error Growth $\Delta\mathrm{RMSE} = \mathrm{RMSE}(g) - \mathrm{RMSE}(1\mathrm{h})$")
    ax2.set_xlabel("Consecutive Outage Duration (Hours)")
    ax2.set_ylabel(r"Error Growth ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax2.set_xticks(sorted(propagation_df["gap_hours"].unique()))
    ax2.legend(loc="upper left")

    plt.suptitle("Propagation Gap Analysis: Error Scaling Across Outage Durations", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_site_wise_performance(
    site_metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Plots Test RMSE and R^2 across study locations (Delhi, Hamirpur, Bangalore).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=site_metrics_df, x="site", y="test_rmse", hue="model", palette="Set2", ax=ax1)
    ax1.set_title("Reconstruction RMSE by Study Location")
    ax1.set_ylabel(r"Test RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax1.set_xlabel("Site")

    sns.barplot(data=site_metrics_df, x="site", y="test_r2", hue="model", palette="Set2", ax=ax2)
    ax2.set_title(r"Reconstruction $R^2$ by Study Location")
    ax2.set_ylabel(r"Test $R^2$")
    ax2.set_xlabel("Site")

    plt.suptitle("Cross-Site Reconstruction Performance Comparison", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_humidity_conditioned_error(
    humidity_metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Plots RMSE and Mean Bias Error stratified by environmental humidity regime.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=humidity_metrics_df, x="humidity_regime", y="rmse", color="#2B83BA", ax=ax1)
    ax1.set_title("Reconstruction RMSE across Environmental Humidity Regimes")
    ax1.set_ylabel(r"RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax1.set_xlabel("Environmental Regime")
    ax1.tick_params(axis="x", rotation=15)

    sns.barplot(data=humidity_metrics_df, x="humidity_regime", y="mbe", color="#FDAE61", ax=ax2)
    ax2.set_title("Mean Bias Error across Environmental Humidity Regimes")
    ax2.set_ylabel(r"Mean Bias Error ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax2.set_xlabel("Environmental Regime")
    ax2.axhline(0, color="#333333", linestyle="--", linewidth=1.0)
    ax2.tick_params(axis="x", rotation=15)

    plt.suptitle("Impact of Environmental Humidity Regimes on Reconstruction Accuracy", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
