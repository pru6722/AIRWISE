"""
Module: src.evaluation.detailed_investigation
Generates rigorous same-split comparisons between raw sensor channels (PA_ATM, PA_CF1, PA_ALT)
and calibration models across sites, pollution ranges, and multi-channel configurations.
Addresses specific research critique points from Checkpoint 2.
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import create_chronological_splits
from src.evaluation.metrics import compute_regression_metrics


def run_detailed_calibration_investigation(
    cleaned_parquet_path: str = "data/processed/cleaned_data.parquet",
    models_dir: str = "outputs/models/calibration",
    reports_dir: str = "outputs/reports",
    figures_dir: str = "outputs/figures",
):
    reports_path = Path(reports_dir)
    figures_path = Path(figures_dir)
    models_path = Path(models_dir)

    reports_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data and create splits
    df = pd.read_parquet(cleaned_parquet_path)
    train_df, val_df, test_df, _ = create_chronological_splits(df)

    FEATURE_SETS = {
        "cf1_only": ["pa_cf1", "relative_humidity", "temperature"],
        "atm_only": ["pa_atm", "relative_humidity", "temperature"],
        "alt_only": ["pa_alt", "relative_humidity", "temperature"],
        "all_channels": ["pa_cf1", "pa_atm", "pa_alt", "relative_humidity", "temperature"],
    }

    # Load pooled models
    rf_all = joblib.load(models_path / "pooled_random_forest_all_channels.joblib")
    lr_all = joblib.load(models_path / "pooled_linear_regression_all_channels.joblib")
    xgb_all = joblib.load(models_path / "pooled_xgboost_all_channels.joblib")

    rf_atm = joblib.load(models_path / "pooled_random_forest_atm_only.joblib")
    lr_atm = joblib.load(models_path / "pooled_linear_regression_atm_only.joblib")

    test_df["pred_rf_all"] = rf_all.predict(test_df[FEATURE_SETS["all_channels"]].values)
    test_df["pred_lr_all"] = lr_all.predict(test_df[FEATURE_SETS["all_channels"]].values)
    test_df["pred_xgb_all"] = xgb_all.predict(test_df[FEATURE_SETS["all_channels"]].values)

    test_df["pred_rf_atm"] = rf_atm.predict(test_df[FEATURE_SETS["atm_only"]].values)
    test_df["pred_lr_atm"] = lr_atm.predict(test_df[FEATURE_SETS["atm_only"]].values)

    # =========================================================================
    # PART 1: EXACT SAME-SPLIT COMPARISON BY POLLUTION RANGE
    # =========================================================================
    bands = [
        ("Low / Satisfactory (<=60 ug/m3)", test_df["bam_reference"] <= 60.0),
        ("Moderate / Poor (61-120 ug/m3)", (test_df["bam_reference"] > 60.0) & (test_df["bam_reference"] <= 120.0)),
        ("Very Poor / Severe (>120 ug/m3)", test_df["bam_reference"] > 120.0),
        ("All Concentrations (Pooled)", np.ones(len(test_df), dtype=bool)),
    ]

    candidate_models = [
        ("Raw PA_ATM", "pa_atm"),
        ("Raw PA_CF1", "pa_cf1"),
        ("Raw PA_ALT", "pa_alt"),
        ("Linear Regression (ATM only)", "pred_lr_atm"),
        ("Linear Regression (All Channels)", "pred_lr_all"),
        ("Random Forest (ATM only)", "pred_rf_atm"),
        ("Random Forest (All Channels)", "pred_rf_all"),
        ("XGBoost (All Channels)", "pred_xgb_all"),
    ]

    poll_rows = []
    for band_label, msk in bands:
        sub = test_df[msk]
        n_samples = len(sub)
        bam = sub["bam_reference"].values
        mean_bam = float(np.mean(bam))

        for model_label, col_name in candidate_models:
            preds = sub[col_name].values
            met = compute_regression_metrics(bam, preds)
            poll_rows.append({
                "pollution_range": band_label,
                "n_samples": n_samples,
                "mean_bam_ug_m3": round(mean_bam, 2),
                "model": model_label,
                "rmse": met["rmse"],
                "mae": met["mae"],
                "r2": met["r2"],
                "mbe": met["mbe"],
                "rel_bias_pct": met["relative_bias_pct"],
            })

    poll_df = pd.DataFrame(poll_rows)
    poll_df.to_csv(reports_path / "calibration_by_pollution_range.csv", index=False)
    print("Saved calibration_by_pollution_range.csv")

    # =========================================================================
    # PART 2: MULTI-CHANNEL VS SINGLE-CHANNEL COMPARISON BY SITE
    # =========================================================================
    mc_rows = []
    sites = ["Bangalore", "Hamirpur", "Delhi", "Pooled (All Sites)"]

    for site in sites:
        if site == "Pooled (All Sites)":
            sub = test_df.copy()
        else:
            sub = test_df[test_df["site"] == site].copy()

        bam = sub["bam_reference"].values

        # Raw ATM
        m_raw = compute_regression_metrics(bam, sub["pa_atm"])
        mc_rows.append({
            "site": site,
            "feature_set": "Raw PA_ATM (Baseline)",
            "algorithm": "Uncalibrated",
            "rmse": m_raw["rmse"],
            "mae": m_raw["mae"],
            "r2": m_raw["r2"],
            "mbe": m_raw["mbe"],
        })

        for algo in ["linear_regression", "random_forest", "xgboost"]:
            for fset in ["atm_only", "cf1_only", "alt_only", "all_channels"]:
                model_obj = joblib.load(models_path / f"pooled_{algo}_{fset}.joblib")
                p = model_obj.predict(sub[FEATURE_SETS[fset]].values)
                m = compute_regression_metrics(bam, p)
                mc_rows.append({
                    "site": site,
                    "feature_set": fset,
                    "algorithm": algo.replace("_", " ").title(),
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "mbe": m["mbe"],
                })

    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(reports_path / "multichannel_comparison_by_site.csv", index=False)
    print("Saved multichannel_comparison_by_site.csv")

    # =========================================================================
    # PART 3: GENERATE HIGH-RESOLUTION PUBLICATION DIAGNOSTIC FIGURE
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: RMSE by Pollution Range
    poll_sub = poll_df[
        poll_df["model"].isin(["Raw PA_ATM", "Linear Regression (All Channels)", "Random Forest (All Channels)"])
        & (poll_df["pollution_range"] != "All Concentrations (Pooled)")
    ]
    sns.barplot(data=poll_sub, x="pollution_range", y="rmse", hue="model", palette=["#7570B3", "#1B9E77", "#D95F02"], ax=axes[0, 0])
    axes[0, 0].set_title("(a) Test RMSE by Pollution Range (Same-Split Comparison)")
    axes[0, 0].set_ylabel(r"RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    axes[0, 0].set_xlabel("Pollution Concentration Band")
    axes[0, 0].tick_params(axis="x", rotation=12)
    axes[0, 0].legend(title="Model", loc="upper left")

    # Panel B: Relative Bias (%) by Pollution Range
    sns.barplot(data=poll_sub, x="pollution_range", y="rel_bias_pct", hue="model", palette=["#7570B3", "#1B9E77", "#D95F02"], ax=axes[0, 1])
    axes[0, 1].axhline(0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("(b) Relative Bias (%) by Pollution Range")
    axes[0, 1].set_ylabel("Relative Bias (%)")
    axes[0, 1].set_xlabel("Pollution Concentration Band")
    axes[0, 1].tick_params(axis="x", rotation=12)
    axes[0, 1].legend(title="Model", loc="lower left")

    # Panel C: Multi-Channel vs Single-Channel across Sites (Random Forest)
    rf_mc = mc_df[
        (mc_df["algorithm"] == "Random Forest")
        & (mc_df["site"].isin(["Bangalore", "Hamirpur", "Delhi"]))
    ]
    sns.barplot(data=rf_mc, x="site", y="rmse", hue="feature_set", palette="Blues_d", ax=axes[1, 0])
    axes[1, 0].set_title("(c) Random Forest RMSE: Multi-Channel vs. Single-Channel by Site")
    axes[1, 0].set_ylabel(r"Test RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    axes[1, 0].set_xlabel("Study Location")
    axes[1, 0].legend(title="Feature Configuration", loc="upper left")

    # Panel D: Site-wise RMSE: Raw PA_ATM vs Calibrated Models
    site_comp = poll_df[poll_df["pollution_range"] == "All Concentrations (Pooled)"].copy()
    # Pull site-wise from mc_df
    site_models = mc_df[
        mc_df["site"].isin(["Bangalore", "Hamirpur", "Delhi"])
        & (mc_df["feature_set"].isin(["Raw PA_ATM (Baseline)", "all_channels"]))
    ].copy()
    site_models["label"] = site_models["algorithm"] + " (" + site_models["feature_set"] + ")"
    sns.barplot(data=site_models, x="site", y="rmse", hue="label", palette=["#7570B3", "#1B9E77", "#D95F02", "#E7298A"], ax=axes[1, 1])
    axes[1, 1].set_title("(d) Overall Test RMSE by Location: Raw PA_ATM vs. Calibrated Models")
    axes[1, 1].set_ylabel(r"Test RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    axes[1, 1].set_xlabel("Study Location")
    axes[1, 1].legend(title="Model Configuration", loc="upper left")

    plt.suptitle("Empirical Diagnostic Breakdown: Calibration Fidelity, Bias Cancellation, and Multi-Channel Benefit", y=1.01, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(figures_path / "fig_calibration_vs_raw_atm_breakdown.png")
    plt.close(fig)
    print("Saved fig_calibration_vs_raw_atm_breakdown.png")


if __name__ == "__main__":
    run_detailed_calibration_investigation()
