"""
Module: src.visualization.conformal_plots
Generates 10 publication-quality figures for Checkpoint 4 Conformal Uncertainty Estimation:
1. Actual vs predicted PM2.5 with prediction intervals
2. 90% and 95% empirical coverage comparison
3. Interval width by humidity regime
4. Interval width by site
5. Coverage by predicted-pollution regime
6. Coverage by missingness type
7. Coverage versus gap duration
8. Delhi extreme-pollution interval analysis
9. Ordinary versus block conformal comparison
10. Interval width versus prediction error
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

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
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_actual_vs_predicted_with_intervals(
    df: pd.DataFrame,
    site_name: str,
    output_path: Path,
    n_days: int = 10,
):
    """Figure 1: Time series showing actual BAM, predicted PM2.5, and 90%/95% conformal intervals."""
    sub = df[df["site"] == site_name].sort_values("timestamp").reset_index(drop=True)
    n_pts = min(len(sub), n_days * 24)
    sub = sub.iloc[:n_pts].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(sub["timestamp"], sub["lower_95"], sub["upper_95"], color="#CBD5E0", alpha=0.5, label="95% Conformal Interval")
    ax.fill_between(sub["timestamp"], sub["lower_90"], sub["upper_90"], color="#90CDF4", alpha=0.6, label="90% Conformal Interval")
    ax.plot(sub["timestamp"], sub["point_prediction"], color="#2B6CB0", linewidth=1.8, label="Joint TCN Prediction")
    ax.plot(sub["timestamp"], sub["true_bam"], color="#1A202C", linewidth=2.0, linestyle="-", label="Reference BAM")

    ax.set_title(f"Conformal Uncertainty Intervals — {site_name} (Representative {n_days}-Day Window)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(r"$\mathrm{PM}_{2.5}$ Concentration ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_coverage_comparison(cov_df: pd.DataFrame, output_path: Path):
    """Figure 2: 90% and 95% empirical coverage across conformal methods."""
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = cov_df.copy()

    x = np.arange(len(plot_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, plot_df["cov_90"], width, label="Nominal 90% (α=0.10)", color="#3182CE")
    rects2 = ax.bar(x + width/2, plot_df["cov_95"], width, label="Nominal 95% (α=0.05)", color="#319795")

    ax.axhline(90.0, color="#2B6CB0", linestyle="--", linewidth=1.2, label="90% Target Line")
    ax.axhline(95.0, color="#234E52", linestyle=":", linewidth=1.2, label="95% Target Line")

    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_title("Empirical vs. Nominal Coverage across Conformal Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["method"], rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_interval_width_by_humidity(hum_df: pd.DataFrame, output_path: Path):
    """Figure 3: Interval width across humidity regimes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=hum_df, x="humidity_regime", y="mean_interval_width", hue="conformal_method", palette="Blues_r", ax=ax)
    ax.set_title("Mean 90% Conformal Interval Width by Environmental Humidity Regime")
    ax.set_ylabel(r"Mean Interval Width ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.legend(title="Method", loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_interval_width_by_site(site_df: pd.DataFrame, output_path: Path):
    """Figure 4: Interval width across study sites."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=site_df, x="site", y="mean_interval_width", hue="conformal_method", palette="viridis", ax=ax)
    ax.set_title("Mean 90% Conformal Interval Width by Study Location")
    ax.set_ylabel(r"Mean Interval Width ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.legend(title="Method", loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_coverage_by_predicted_pollution(poll_df: pd.DataFrame, output_path: Path):
    """Figure 5: Empirical coverage across operational predicted pollution regimes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=poll_df, x="predicted_pollution_regime", y="empirical_coverage", hue="conformal_method", palette="magma", ax=ax)
    ax.axhline(90.0, color="#E53E3E", linestyle="--", linewidth=1.2, label="90% Nominal Target")
    ax.set_title("Empirical Coverage across Operational Predicted Pollution Regimes")
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="Method", loc="lower left")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_coverage_by_missingness_type(miss_df: pd.DataFrame, output_path: Path):
    """Figure 6: Empirical coverage across missingness mechanisms."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=miss_df, x="missingness_type", y="empirical_coverage", hue="conformal_method", palette="Set2", ax=ax)
    ax.axhline(90.0, color="#E53E3E", linestyle="--", linewidth=1.2, label="90% Nominal Target")
    ax.set_title("Conformal Coverage across Missingness Mechanisms (20% Outage Rate)")
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="Method", loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_coverage_vs_gap_duration(gap_df: pd.DataFrame, output_path: Path):
    """Figure 7: Empirical coverage and interval width vs consecutive gap duration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method, grp in gap_df.groupby("conformal_method"):
        ax1.plot(grp["gap_hours"], grp["empirical_coverage"], marker="o", label=method, linewidth=2.0)
        ax2.plot(grp["gap_hours"], grp["mean_interval_width"], marker="s", label=method, linewidth=2.0)

    ax1.axhline(90.0, color="#E53E3E", linestyle="--", label="90% Nominal")
    ax1.set_title("Empirical Coverage vs. Outage Gap Duration")
    ax1.set_xlabel("Consecutive Outage Duration (Hours)")
    ax1.set_ylabel("Empirical Coverage (%)")
    ax1.set_xticks([1, 3, 6, 12, 24])
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left")

    ax2.set_title("Mean Interval Width vs. Outage Gap Duration")
    ax2.set_xlabel("Consecutive Outage Duration (Hours)")
    ax2.set_ylabel(r"Interval Width ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax2.set_xticks([1, 3, 6, 12, 24])
    ax2.legend(loc="upper left")

    plt.suptitle("Impact of Sensor Outage Duration on Conformal Uncertainty Intervals", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_delhi_extreme_pollution_conformal(ext_df: pd.DataFrame, output_path: Path):
    """Figure 8: Empirical coverage and interval width across true ground-truth BAM bands in Delhi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(ext_df))
    width = 0.35

    rects1 = ax1.bar(x - width/2, ext_df["cov_90"], width, label="90% Nominal Coverage", color="#3182CE")
    rects2 = ax1.bar(x + width/2, ext_df["cov_95"], width, label="95% Nominal Coverage", color="#319795")
    ax1.axhline(90.0, color="#2B6CB0", linestyle="--", linewidth=1.2)
    ax1.axhline(95.0, color="#234E52", linestyle=":", linewidth=1.2)
    ax1.set_title("Empirical Coverage across True BAM Pollution Bands in Delhi")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ext_df["bam_band"], rotation=20, ha="right")
    ax1.set_ylabel("Empirical Coverage (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left")

    rects3 = ax2.bar(x, ext_df["mean_width_90"], width=0.5, color="#E53E3E", label="Mean Width (90%)")
    ax2.set_title("Conformal Interval Width across Pollution Bands")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ext_df["bam_band"], rotation=20, ha="right")
    ax2.set_ylabel(r"Mean Interval Width ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax2.legend(loc="upper left")

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.suptitle("Retrospective Delhi Extreme-Pollution Failure Diagnostics", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ordinary_vs_block_conformal(block_comp_df: pd.DataFrame, output_path: Path):
    """Figure 9: Comparison of Ordinary Split Conformal vs Block Conformal (6h, 12h, 24h)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    plot_df = block_comp_df.copy()
    x = np.arange(len(plot_df))
    width = 0.4

    ax1.bar(x, plot_df["empirical_coverage_90"], width, color="#3182CE")
    ax1.axhline(90.0, color="#E53E3E", linestyle="--", linewidth=1.5, label="90% Nominal")
    ax1.set_title("Empirical Coverage: Ordinary vs. Block Conformal")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["method"], rotation=20, ha="right")
    ax1.set_ylabel("Empirical Coverage (%)")
    ax1.set_ylim(0, 105)
    ax1.legend()

    ax2.bar(x, plot_df["mean_interval_width_90"], width, color="#DD6B20")
    ax2.set_title("Interval Width: Ordinary vs. Block Conformal")
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_df["method"], rotation=20, ha="right")
    ax2.set_ylabel(r"Mean Width ($\mu\mathrm{g}/\mathrm{m}^3$)")

    plt.suptitle("Temporal Dependence: Pointwise Split Conformal vs. Block-Maximum Protection", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_interval_width_vs_prediction_error(pred_df: pd.DataFrame, output_path: Path):
    """Figure 10: Scatter plot of Conformal Interval Width vs Absolute Prediction Error."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sub = pred_df.sample(n=min(len(pred_df), 1500), random_state=42).copy()
    abs_err = np.abs(sub["true_bam"] - sub["point_prediction"])
    width_90 = sub["upper_90"] - sub["lower_90"]

    ax.scatter(abs_err, width_90, alpha=0.35, color="#2B6CB0", s=20)
    ax.axline((0, 0), slope=2.0, color="#E53E3E", linestyle="--", linewidth=1.5, label="Coverage Boundary: Width = 2 * Error")

    ax.set_title("Conformal Interval Width vs. Absolute Prediction Error (Test Set)")
    ax.set_xlabel(r"Absolute Prediction Error $|Y - \widehat{Y}|$ ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.set_ylabel(r"90% Conformal Interval Width ($U_{90} - L_{90}$) ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
