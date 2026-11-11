"""
Module: src.aqi.cpcb_plots (and src.visualization.cpcb_plots)
Generates publication-quality diagnostic plots for CPCB PM2.5 Category Reliability Assessment.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.aqi.cpcb_categories import (
    CPCB_CATEGORIES,
    CPCB_CATEGORY_COLORS,
    CPCB_BOUNDARY_VALUES,
    CPCB_BREAKPOINTS,
)


def set_publication_style():
    """Sets standard matplotlib publication styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_24h_bam_vs_pred_scatter(
    df: pd.DataFrame,
    save_path: Path,
):
    """
    Fig 1: 24-hour BAM reference vs predicted PM2.5 scatter plot with category color zones and 1:1 line.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    sites = df["site"].unique()
    markers = {"Delhi": "o", "Hamirpur": "s", "Bangalore": "^"}
    colors = {"Delhi": "#D95F02", "Hamirpur": "#7570B3", "Bangalore": "#1B9E77"}

    # Plot category threshold lines
    for b in CPCB_BOUNDARY_VALUES:
        ax.axvline(b, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.axhline(b, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    # Category zone annotations along axes
    zone_midpoints = [15.0, 45.0, 75.0, 105.0, 185.0]
    zone_labels = ["Good", "Sat.", "Mod.", "Poor", "V.Poor"]
    for m, l in zip(zone_midpoints, zone_labels):
        ax.text(m, 490, l, ha="center", va="top", fontsize=8, color="darkgray", fontweight="bold")

    for site in sites:
        sub = df[df["site"] == site]
        ax.scatter(
            sub["ref_pm25_24h"],
            sub["pred_pm25_24h"],
            c=colors.get(site, "blue"),
            marker=markers.get(site, "o"),
            alpha=0.75,
            edgecolors="k",
            linewidth=0.5,
            s=45,
            label=f"{site} (N={len(sub)})",
        )

    # 1:1 Line
    max_val = max(df["ref_pm25_24h"].max(), df["pred_pm25_24h"].max()) + 20
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.2, label="1:1 Perfect Agreement")

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Reference 24-Hour BAM $\\mathrm{PM}_{2.5}$ ($\\mu\\mathrm{g}/\\mathrm{m}^3$)")
    ax.set_ylabel("Predicted 24-Hour $\\mathrm{PM}_{2.5}$ ($\\mu\\mathrm{g}/\\mathrm{m}^3$)")
    ax.set_title("24-Hour BAM Reference vs. Model Prediction\nAcross CPCB Air Quality Regimes")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_24h_timeseries_intervals(
    df: pd.DataFrame,
    save_path: Path,
):
    """
    Fig 2: 24-hour prediction time series with uncertainty intervals.
    """
    set_publication_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharey=False)

    site_names = ["Delhi", "Hamirpur", "Bangalore"]
    colors = {"Delhi": "#D95F02", "Hamirpur": "#7570B3", "Bangalore": "#1B9E77"}

    for idx, site in enumerate(site_names):
        ax = axes[idx]
        sub = df[df["site"] == site].sort_values("start_time").reset_index(drop=True)

        if len(sub) == 0:
            ax.text(0.5, 0.5, f"No valid windows for {site}", ha="center", va="center")
            continue

        x = sub["start_time"]
        y_ref = sub["ref_pm25_24h"]
        y_pred = sub["pred_pm25_24h"]
        l90 = sub["lower_90"]
        u90 = sub["upper_90"]

        ax.fill_between(x, l90, u90, color=colors[site], alpha=0.25, label="90% Conformal Interval")
        ax.plot(x, y_pred, color=colors[site], linewidth=1.5, label="24h Model Prediction")
        ax.scatter(x, y_ref, color="black", s=20, zorder=5, label="24h Reference BAM")

        # CPCB boundaries
        for b in [30.0, 60.0, 120.0, 250.0]:
            ax.axhline(b, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

        ax.set_ylabel("$\\mathrm{PM}_{2.5}$ ($\\mu\\mathrm{g}/\\mathrm{m}^3$)")
        ax.set_title(f"{site} — 24-Hour Daily Trajectory with 90% Conformal Uncertainty (N={len(sub)})")
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.85)

    axes[-1].set_xlabel("Date (Start of 24-Hour Window)")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cpcb_category_confusion_matrix(
    cm: np.ndarray,
    categories: List[str],
    save_path: Path,
):
    """
    Fig 3: Normalized and raw 6x6 confusion matrix heatmap.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted CPCB Category")
    axes[0].set_ylabel("Reference CPCB Category")
    axes[0].set_title("CPCB Category Confusion Matrix (Raw Window Counts)")

    # Row-normalized (Recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = (cm / row_sums) * 100.0

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=categories,
        yticklabels=categories,
        cbar=True,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted CPCB Category")
    axes[1].set_ylabel("Reference CPCB Category")
    axes[1].set_title("CPCB Category Confusion Matrix (Row-Normalized Recall %)")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_category_precision_recall_f1(
    cat_metrics: Dict[str, Dict[str, float]],
    save_path: Path,
):
    """
    Fig 4: Category-wise precision, recall, and F1 bar plot.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    cats = [c for c in CPCB_CATEGORIES if c in cat_metrics]
    precisions = [cat_metrics[c]["precision"] for c in cats]
    recalls = [cat_metrics[c]["recall"] for c in cats]
    f1s = [cat_metrics[c]["f1_score"] for c in cats]
    supports = [cat_metrics[c]["support"] for c in cats]

    x = np.arange(len(cats))
    width = 0.25

    ax.bar(x - width, precisions, width, label="Precision (%)", color="#4575b4", alpha=0.9)
    ax.bar(x, recalls, width, label="Recall (%)", color="#74add1", alpha=0.9)
    ax.bar(x + width, f1s, width, label="F1-Score (%)", color="#f46d43", alpha=0.9)

    # Annotate support above bars
    for i, s in enumerate(supports):
        ax.text(x[i], max(precisions[i], recalls[i], f1s[i]) + 2, f"N={s}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.set_title("CPCB PM2.5 Category Classification Performance by Class")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_category_stability_distribution(
    summary: Dict[str, Any],
    save_path: Path,
):
    """
    Fig 5: Category stability vs ambiguity distribution pie and accuracy comparison.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Pie chart of Stable vs Ambiguous
    labels = [f"Stable Category\n({summary['pct_stable']}%)", f"Ambiguous Category\n({summary['pct_ambiguous']}%)"]
    sizes = [summary["n_stable"], summary["n_ambiguous"]]
    colors = ["#2b83ba", "#d7191c"]
    explode = (0.05, 0)

    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    axes[0].set_title(f"Uncertainty-Aware Category Stability\n(Total Test Windows N={summary['n_total']})")

    # Accuracy Comparison Bar Chart
    acc_labels = ["Overall", "Stable Windows", "Ambiguous Windows", "Ref Category\nCompatibility"]
    acc_values = [
        summary["point_accuracy_overall"],
        summary["point_accuracy_stable"],
        summary["point_accuracy_ambiguous"],
        summary["ref_category_compatibility_pct"],
    ]
    bar_colors = ["#abd9e9", "#2b83ba", "#fdae61", "#d7191c"]

    bars = axes[1].bar(acc_labels, acc_values, color=bar_colors, width=0.55, edgecolor="k", linewidth=0.6)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    axes[1].set_ylabel("Accuracy / Compatibility Rate (%)")
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Point Category Accuracy vs. Interval Compatibility")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_category_stability_by_site(
    df: pd.DataFrame,
    save_path: Path,
):
    """
    Fig 6: Stable vs ambiguous category breakdown by site.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    sites = df["site"].unique()
    stable_counts = []
    ambig_counts = []
    stable_pcts = []

    for site in sites:
        sub = df[df["site"] == site]
        n_st = sub["is_stable"].sum()
        n_amb = len(sub) - n_st
        stable_counts.append(n_st)
        ambig_counts.append(n_amb)
        stable_pcts.append((n_st / len(sub)) * 100 if len(sub) > 0 else 0)

    x = np.arange(len(sites))
    width = 0.45

    ax.bar(x, stable_counts, width, label="Stable Category", color="#2b83ba", alpha=0.9)
    ax.bar(x, ambig_counts, width, bottom=stable_counts, label="Ambiguous Category", color="#fdae61", alpha=0.9)

    for i in range(len(sites)):
        tot = stable_counts[i] + ambig_counts[i]
        pct = stable_pcts[i]
        ax.text(x[i], tot + 1, f"Total: {tot}\n({pct:.1f}% Stable)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Number of 24-Hour Windows")
    ax.set_ylim(0, max([s + a for s, a in zip(stable_counts, ambig_counts)]) + 12)
    ax.set_title("Category Stability vs. Ambiguity Across Study Sites")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_reference_pm25_vs_interval_width(
    df: pd.DataFrame,
    save_path: Path,
):
    """
    Fig 7: Reference PM2.5 concentration vs interval width scatter with category boundaries.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sites = df["site"].unique()
    colors = {"Delhi": "#D95F02", "Hamirpur": "#7570B3", "Bangalore": "#1B9E77"}

    for site in sites:
        sub = df[df["site"] == site]
        ax.scatter(
            sub["ref_pm25_24h"],
            sub["interval_width"],
            c=colors.get(site, "gray"),
            alpha=0.8,
            edgecolors="k",
            linewidth=0.5,
            s=40,
            label=site,
        )

    for b in CPCB_BOUNDARY_VALUES:
        ax.axvline(b, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Reference 24-Hour BAM $\\mathrm{PM}_{2.5}$ ($\\mu\\mathrm{g}/\\mathrm{m}^3$)")
    ax.set_ylabel("24-Hour Conformal Interval Width ($\\mu\\mathrm{g}/\\mathrm{m}^3$)")
    ax.set_title("Reference $\\mathrm{PM}_{2.5}$ Concentration vs. Prediction Interval Width")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_boundary_distance_vs_category_error(
    boundary_df: pd.DataFrame,
    save_path: Path,
):
    """
    Fig 8: Category boundary distance vs point error and ambiguity rate.
    """
    set_publication_style()
    fig, ax1 = plt.subplots(figsize=(9, 5))

    labels = boundary_df["distance_tier"].tolist()
    accs = boundary_df["point_accuracy"].tolist()
    ambigs = boundary_df["pct_ambiguous"].tolist()
    supports = boundary_df["n_windows"].tolist()

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax1.bar(x - width/2, accs, width, label="Point Category Accuracy (%)", color="#2b83ba")
    rects2 = ax1.bar(x + width/2, ambigs, width, label="Ambiguity Frequency (%)", color="#fdae61")

    # Annotate values
    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{lbl}\n(N={supp})" for lbl, supp in zip(labels, supports)])
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 110)
    ax1.set_title("Impact of CPCB Boundary Proximity on Category Accuracy and Interval Ambiguity")
    ax1.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_coverage_and_width_by_category(
    cat_df: pd.DataFrame,
    save_path: Path,
):
    """
    Fig 9: Empirical coverage and interval width across CPCB categories.
    """
    set_publication_style()
    fig, ax1 = plt.subplots(figsize=(10, 5))

    cats = cat_df["cpcb_category"].tolist()
    covs = cat_df["empirical_coverage_90"].tolist()
    widths = cat_df["mean_interval_width"].tolist()
    supports = cat_df["n_windows"].tolist()

    x = np.arange(len(cats))
    width = 0.4

    color1 = "#2c7bb6"
    color2 = "#d7191c"

    rects1 = ax1.bar(x, covs, width, color=color1, alpha=0.85, label="Empirical Coverage (%)")
    ax1.axhline(90.0, color="k", linestyle="--", linewidth=1.2, label="Nominal 90% Target")
    ax1.set_ylabel("Empirical 90% Coverage (%)", color=color1)
    ax1.set_ylim(0, 115)
    ax1.tick_params(axis="y", labelcolor=color1)

    for i, r in enumerate(rects1):
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 2, f"{h:.1f}%\n(N={supports[i]})", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats)
    ax1.set_title("24-Hour Empirical Coverage Across CPCB $\\mathrm{PM}_{2.5}$ Categories")
    ax1.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_completeness_sensitivity(
    sens_df: pd.DataFrame,
    save_path: Path,
):
    """
    Fig 10: Sensitivity analysis across 18/24, 20/24, and 24/24 completeness thresholds.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(sens_df))
    labels = sens_df["completeness_threshold"].tolist()
    n_windows = sens_df["n_valid_windows"].tolist()
    covs = sens_df["empirical_coverage_90"].tolist()
    accs = sens_df["category_accuracy"].tolist()

    # Left: Window counts
    bars = axes[0].bar(x, n_windows, color="#4575b4", width=0.5, edgecolor="k")
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 2, f"{int(h)}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Valid Test Windows Count")
    axes[0].set_ylim(0, max(n_windows) * 1.2)
    axes[0].set_title("Valid 24-Hour Windows vs. Completeness Threshold")

    # Right: Coverage and Accuracy
    width = 0.35
    b1 = axes[1].bar(x - width/2, covs, width, label="90% Conformal Coverage (%)", color="#2b83ba")
    b2 = axes[1].bar(x + width/2, accs, width, label="Category Accuracy (%)", color="#fdae61")

    for bar in b1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    for bar in b2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    axes[1].axhline(90.0, color="k", linestyle="--", alpha=0.7, label="Nominal 90% Target")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Coverage & Category Accuracy Sensitivity")
    axes[1].legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
