"""
Module: src.visualization.tcn_plots
Generates publication-quality figures for Checkpoint 3 Joint TCN evaluations:
architecture diagram, loss curves, calibration & reconstruction comparisons,
propagation curves, site/pollution/humidity breakdowns, and ablation analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
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


def plot_joint_tcn_architecture(output_path: Union[str, Path]) -> Path:
    """Draws a clean, publication-ready schematic diagram of the Joint TCN architecture."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_input = "#EBF8FF"
    c_encoder = "#E6FFFA"
    c_calib = "#FEFCBF"
    c_rec = "#FED7D7"
    c_border = "#2B6CB0"

    # Input Box
    ax.add_patch(patches.FancyBboxPatch((4, 25), 18, 50, boxstyle="round,pad=1.5", facecolor=c_input, edgecolor="#3182CE", linewidth=1.5))
    ax.text(13, 70, "INPUT TENSOR (X_t, M_t)", ha="center", va="center", fontweight="bold", fontsize=10.5, color="#2B6CB0")
    ax.text(13, 62, "• PA_CF1, PA_ATM, PA_ALT", ha="center", va="center", fontsize=8.5)
    ax.text(13, 56, "• Relative Humidity (RH)", ha="center", va="center", fontsize=8.5)
    ax.text(13, 50, "• Ambient Temp (T)", ha="center", va="center", fontsize=8.5)
    ax.text(13, 44, "• Observation Mask (M_t)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#C53030")
    ax.text(13, 38, "• Cyclical Diurnal Time", ha="center", va="center", fontsize=8.5)
    ax.text(13, 32, "• Time Step Delta (dt)", ha="center", va="center", fontsize=8.5)

    # Arrow 1
    ax.annotate("", xy=(28, 50), xytext=(23, 50), arrowprops=dict(arrowstyle="->", lw=2.0, color="#4A5568"))

    # Shared Encoder Box
    ax.add_patch(patches.FancyBboxPatch((29, 20), 28, 60, boxstyle="round,pad=1.5", facecolor=c_encoder, edgecolor="#319795", linewidth=2.0))
    ax.text(43, 74, "SHARED TEMPORAL ENCODER", ha="center", va="center", fontweight="bold", fontsize=11, color="#234E52")
    ax.text(43, 67, "Dilated Causal TCN Blocks", ha="center", va="center", fontsize=9.5, style="italic")
    # Residual block layers
    for i, (d, rf) in enumerate([(1, 3), (2, 7), (4, 15), (8, 31)]):
        y_b = 58 - i * 9
        ax.add_patch(patches.Rectangle((32, y_b - 3), 22, 6, facecolor="white", edgecolor="#4FD1C5", linewidth=1.2))
        ax.text(43, y_b, f"ResBlock {i+1}: Dilation d={d} (RF={rf}h)", ha="center", va="center", fontsize=8)

    # Branching arrows
    ax.annotate("", xy=(63, 66), xytext=(58, 55), arrowprops=dict(arrowstyle="->", lw=2.0, color="#4A5568"))
    ax.annotate("", xy=(63, 34), xytext=(58, 45), arrowprops=dict(arrowstyle="->", lw=2.0, color="#4A5568"))

    # Head A: Calibration Head
    ax.add_patch(patches.FancyBboxPatch((64, 55), 26, 25, boxstyle="round,pad=1.5", facecolor=c_calib, edgecolor="#D69E2E", linewidth=1.8))
    ax.text(77, 74, "HEAD A: CALIBRATION", ha="center", va="center", fontweight="bold", fontsize=10.5, color="#744210")
    ax.text(77, 67, "Supervised on Observed Points (M_t = 1)", ha="center", va="center", fontsize=8)
    ax.text(77, 61, "Loss: L_cal = Huber(Y_cal, Y_BAM)", ha="center", va="center", fontsize=8, color="#975A16", fontweight="bold")

    # Connection from Head A to Head B
    ax.annotate("Calibrated Context (Y_cal * M_t)", xy=(77, 44), xytext=(77, 54),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#C53030", linestyle="--"),
                ha="center", va="center", fontsize=7.5, color="#C53030", fontweight="bold")

    # Head B: Reconstruction Head
    ax.add_patch(patches.FancyBboxPatch((64, 18), 26, 25, boxstyle="round,pad=1.5", facecolor=c_rec, edgecolor="#E53E3E", linewidth=1.8))
    ax.text(77, 37, "HEAD B: RECONSTRUCTION", ha="center", va="center", fontweight="bold", fontsize=10.5, color="#742A2A")
    ax.text(77, 30, "Supervised on Hidden Points (M_t = 0)", ha="center", va="center", fontsize=8)
    ax.text(77, 24, "Loss: L_rec = Huber(Y_rec, Y_BAM)", ha="center", va="center", fontsize=8, color="#9B2C2C", fontweight="bold")

    # Joint Multi-Task Loss Box
    ax.add_patch(patches.FancyBboxPatch((64, 3), 26, 11, boxstyle="round,pad=1.0", facecolor="#EDF2F7", edgecolor="#4A5568", linewidth=1.2))
    ax.text(77, 8.5, "L_joint = λ_cal * L_cal + λ_rec * L_rec", ha="center", va="center", fontweight="bold", fontsize=8.5, color="#1A202C")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_training_loss_curves(history_df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Plots training and validation total, calibration, and reconstruction loss curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs = history_df["epoch"]

    # Left: Total Loss
    ax1.plot(epochs, history_df["train_loss"], label="Train Total Loss", color="#3182CE", linewidth=2.0)
    ax1.plot(epochs, history_df["val_loss"], label="Val Total Loss", color="#E53E3E", linewidth=2.0, linestyle="--")
    best_ep = history_df.loc[history_df["val_loss"].idxmin(), "epoch"]
    best_val = history_df["val_loss"].min()
    ax1.scatter([best_ep], [best_val], color="#D69E2E", s=80, zorder=5, label=f"Best Checkpoint (Ep {best_ep})")
    ax1.set_title("Joint Multi-Task Loss Convergence")
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Huber Loss")
    ax1.legend()

    # Right: Task-Specific Losses
    ax2.plot(epochs, history_df["train_cal_loss"], label="Train L_cal", color="#319795", linewidth=1.5)
    ax2.plot(epochs, history_df["val_cal_loss"], label="Val L_cal", color="#319795", linestyle="--", linewidth=1.5)
    ax2.plot(epochs, history_df["train_rec_loss"], label="Train L_rec", color="#DD6B20", linewidth=1.5)
    ax2.plot(epochs, history_df["val_rec_loss"], label="Val L_rec", color="#DD6B20", linestyle="--", linewidth=1.5)
    ax2.set_title("Calibration vs. Reconstruction Loss Curves")
    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("Task Huber Loss")
    ax2.legend()

    plt.suptitle("Joint TCN Convergence Dynamics across Epochs", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_actual_vs_predicted_time_series(
    df_preds: pd.DataFrame,
    site_name: str,
    output_path: Union[str, Path],
    window_days: int = 10,
) -> Path:
    """Plots actual BAM vs Joint TCN composite predictions highlighting outage periods."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = df_preds[df_preds["site"] == site_name].sort_values("timestamp").reset_index(drop=True)
    n_points = min(len(sub), window_days * 24)
    sub = sub.iloc[:n_points].copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(sub["timestamp"], sub["bam_reference"], label="Reference BAM (True Ground Truth)", color="#111111", linewidth=2.2, zorder=5)
    ax.plot(sub["timestamp"], sub["y_final"], label="Joint TCN Composite Output", color="#3182CE", linewidth=1.8, linestyle="-", zorder=6)

    # Shaded simulated outage blocks
    is_missing = sub["is_observed"] == 0
    if is_missing.any():
        ax.fill_between(
            sub["timestamp"], 0, max(sub["bam_reference"].max(), sub["y_final"].max()) * 1.1,
            where=is_missing, color="#FED7D7", alpha=0.55, label="Simulated Sensor Outage (Reconstruction Mode)", zorder=2
        )

    ax.set_title(f"Joint TCN Performance under Simulated Missingness — {site_name} (Representative 10-Day Window)")
    ax.set_xlabel("Date and Time (IST)")
    ax.set_ylabel(r"$\mathrm{PM}_{2.5}$ Concentration ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_propagation_gap_joint_vs_two_stage(
    propagation_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Plots RMSE and Error Growth G(g) comparing Joint TCN against two-stage baselines across gap durations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    palette = {"Joint TCN (Proposed)": "#E53E3E", "Two-Stage Random Forest Imputer": "#3182CE", "Two-Stage Linear Interpolation": "#718096"}
    markers = {"Joint TCN (Proposed)": "o", "Two-Stage Random Forest Imputer": "s", "Two-Stage Linear Interpolation": "^"}

    # Left: RMSE vs Gap Hours
    for model_name, grp in propagation_df.groupby("model"):
        c = palette.get(model_name, "#4A5568")
        m = markers.get(model_name, "o")
        ax1.plot(grp["gap_hours"], grp["rmse"], marker=m, linewidth=2.2, color=c, label=model_name)
    ax1.set_title("Reconstruction RMSE vs. Outage Gap Duration")
    ax1.set_xlabel("Consecutive Outage Duration (Hours)")
    ax1.set_ylabel(r"Test RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax1.set_xticks(sorted(propagation_df["gap_hours"].unique()))
    ax1.legend(loc="upper left")

    # Right: Error Growth G(g) = RMSE(g) - RMSE(1h)
    for model_name, grp in propagation_df.groupby("model"):
        c = palette.get(model_name, "#4A5568")
        m = markers.get(model_name, "o")
        ax2.plot(grp["gap_hours"], grp["error_growth_rmse"], marker=m, linewidth=2.2, linestyle="--", color=c, label=model_name)
    ax2.set_title(r"Error Growth $\Delta\mathrm{RMSE} = \mathrm{RMSE}(g) - \mathrm{RMSE}(1\mathrm{h})$")
    ax2.set_xlabel("Consecutive Outage Duration (Hours)")
    ax2.set_ylabel(r"Error Growth ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax2.set_xticks(sorted(propagation_df["gap_hours"].unique()))
    ax2.legend(loc="upper left")

    plt.suptitle("Propagation Gap Analysis: Joint TCN vs. Conventional Two-Stage Pipeline", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_ablation_comparison(ablation_df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Plots comparative RMSE across the 6 ablation configurations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_df = ablation_df.copy()
    x = np.arange(len(plot_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, plot_df["cal_observed_rmse"], width, label="Observed Calibration RMSE", color="#319795")
    rects2 = ax.bar(x + width/2, plot_df["rec_hidden_rmse"], width, label="Hidden Reconstruction RMSE", color="#DD6B20")

    ax.set_ylabel(r"RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.set_title("Systematic Architecture Ablation Study (Test Partition)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["ablation_configuration"], rotation=20, ha="right")
    ax.legend()

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


def plot_stratified_breakdowns(
    site_df: pd.DataFrame,
    poll_df: pd.DataFrame,
    hum_df: pd.DataFrame,
    figures_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Generates site-wise, pollution-band, and humidity-regime performance bar charts."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Site-wise
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=site_df, x="site", y="hidden_rmse", hue="model", palette="Set2", ax=ax)
    ax.set_title("Reconstruction RMSE by Study Location (Hidden Observations)")
    ax.set_ylabel(r"RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    p_site = figures_dir / "fig_site_wise_joint_tcn.png"
    fig.savefig(p_site)
    plt.close(fig)
    paths["site"] = p_site

    # 2. Pollution Band
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=poll_df, x="pollution_band", y="hidden_rmse", hue="model", palette="Set2", ax=ax)
    ax.set_title("Reconstruction RMSE across Pollution Concentration Bands")
    ax.set_ylabel(r"RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    p_poll = figures_dir / "fig_pollution_band_joint_tcn.png"
    fig.savefig(p_poll)
    plt.close(fig)
    paths["pollution"] = p_poll

    # 3. Humidity Regime
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hum_df, x="humidity_regime", y="hidden_rmse", hue="model", palette="Set2", ax=ax)
    ax.set_title("Reconstruction RMSE across Environmental Humidity Regimes")
    ax.set_ylabel(r"RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    p_hum = figures_dir / "fig_humidity_regime_joint_tcn.png"
    fig.savefig(p_hum)
    plt.close(fig)
    paths["humidity"] = p_hum

    return paths
