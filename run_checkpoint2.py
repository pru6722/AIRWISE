#!/usr/bin/env python3
"""
Research Pipeline: Checkpoint 2 Execution Script
Comprehensive Calibration and Missing-Data Reconstruction Baseline Experiments.
Executes Module A (Calibration), Module B (Reconstruction), Module C (Missingness),
Module D (Propagation-Gap Analysis), generates publication figures, and compiles research reports.
"""

from pathlib import Path
import json
import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib

# Preload libomp for XGBoost if available
import ctypes
try:
    if os.path.exists("/opt/homebrew/opt/libomp/lib/libomp.dylib"):
        ctypes.cdll.LoadLibrary("/opt/homebrew/opt/libomp/lib/libomp.dylib")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_cleaned_dataset
from src.data.feature_engineering import create_chronological_splits
from src.models.calibration import run_all_calibration_experiments
from src.models.reconstruction import evaluate_reconstruction_baselines
from src.missingness.random_mask import generate_mcar_mask
from src.missingness.block_mask import generate_block_outage_mask
from src.evaluation.propagation import (
    evaluate_propagation_across_gaps,
    evaluate_propagation_by_regimes_and_mechanisms,
)
from src.evaluation.metrics import compute_regression_metrics
from src.visualization.plots import (
    plot_raw_vs_bam_timeseries,
    plot_sensor_vs_bam_scatter,
    plot_calibrated_vs_bam,
    plot_calibration_comparison,
    plot_reconstruction_comparison,
    plot_actual_vs_reconstructed_timeseries,
    plot_error_vs_gap_length,
    plot_site_wise_performance,
    plot_humidity_conditioned_error,
)
from src.visualization.report import (
    generate_calibration_report_html,
    generate_reconstruction_report_html,
)


def main():
    print("=" * 75)
    print("STARTING CHECKPOINT 2: BASELINE MODEL DEVELOPMENT & VALIDATION")
    print("=" * 75)

    # 1. Setup paths and directories
    with open(PROJECT_ROOT / "configs" / "model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    with open(PROJECT_ROOT / "configs" / "experiment_config.yaml", "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    random_seed = model_cfg.get("random_seed", 42)
    np.random.seed(random_seed)

    reports_dir = PROJECT_ROOT / exp_cfg.get("reports_dir", "outputs/reports")
    figures_dir = PROJECT_ROOT / exp_cfg.get("figures_dir", "outputs/figures")
    models_dir = PROJECT_ROOT / exp_cfg.get("models_dir", "outputs/models")
    predictions_dir = PROJECT_ROOT / exp_cfg.get("predictions_dir", "outputs/predictions")
    logs_dir = PROJECT_ROOT / exp_cfg.get("logs_dir", "outputs/logs")
    masks_dir = PROJECT_ROOT / exp_cfg.get("synthetic_masks_dir", "data/synthetic_masks")

    for d in [reports_dir, figures_dir, models_dir / "calibration", models_dir / "reconstruction", predictions_dir, logs_dir, masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load validated dataset
    data_path = PROJECT_ROOT / exp_cfg.get("data_path", "data/processed/cleaned_data.parquet")
    print(f"[1/7] Loading validated dataset from: {data_path.resolve()}")
    df = load_cleaned_dataset(data_path)
    print(f"      Total records: {len(df):,}, columns: {list(df.columns)}")

    # 3. Create chronological train/validation/test partitions
    print("[2/7] Constructing chronological 60/20/20 train/val/test splits within each site...")
    train_df, val_df, test_df, split_info = create_chronological_splits(
        df,
        train_ratio=model_cfg["split_ratios"]["train"],
        val_ratio=model_cfg["split_ratios"]["val"],
        test_ratio=model_cfg["split_ratios"]["test"],
    )
    for site, info in split_info.items():
        if site != "pooled":
            print(f"      - {site:10s} -> Train: {info['train_count']:,} | Val: {info['val_count']:,} | Test: {info['test_count']:,}")
    print(f"      Pooled -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 4. MODULE A: Calibration Experiments
    print("[3/7] Running Module A: Calibration Experiments (LR, RF, XGBoost across 4 feature subsets)...")
    calib_results_df, calib_models_dict, best_calib_bundle = run_all_calibration_experiments(
        train_df, val_df, test_df,
        models_save_dir=models_dir / "calibration",
        random_state=random_seed,
    )
    print(f"      Best Selected Calibrator (based on Val RMSE): {best_calib_bundle['algorithm']} ({best_calib_bundle['feature_set']})")
    print(f"      Validation RMSE: {best_calib_bundle['val_rmse']:.2f} | Test RMSE: {best_calib_bundle['test_rmse']:.2f} | Test R2: {best_calib_bundle['test_r2']:.4f}")

    # Generate test calibrated PM2.5 with best calibrator
    X_test_best = test_df[best_calib_bundle["feature_cols"]].to_numpy(dtype=np.float32)
    test_calibrated_pm25 = best_calib_bundle["model"].predict(X_test_best)

    # Save calibration predictions
    calib_pred_df = test_df[["timestamp", "site", "bam_reference"] + best_calib_bundle["feature_cols"]].copy()
    calib_pred_df["calibrated_pm25"] = test_calibrated_pm25
    calib_pred_path = predictions_dir / "calibration_predictions.csv"
    calib_pred_df.to_csv(calib_pred_path, index=False)
    print(f"      Saved calibration predictions to: {calib_pred_path.name}")

    # 5. MODULE B: Reconstruction Baselines (Two-Stage Pipeline)
    print("[4/7] Running Module B: Reconstruction Experiments (Linear, KNN, RF, GRU, LSTM)...")
    # Generate canonical MCAR 20% benchmark mask for reconstruction comparison
    recon_benchmark_mask = generate_mcar_mask(len(test_df), missing_rate=0.20, random_seed=random_seed)
    # Save benchmark mask
    np.save(masks_dir / "benchmark_test_mask_mcar_20.npy", recon_benchmark_mask)

    covariate_cols = ["relative_humidity", "temperature", "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
    recon_metrics_df, recon_predictions = evaluate_reconstruction_baselines(
        test_df,
        calibration_model=best_calib_bundle["model"],
        calib_feature_cols=best_calib_bundle["feature_cols"],
        mask=recon_benchmark_mask,
        covariate_cols=covariate_cols,
        random_state=random_seed,
    )
    print("      Reconstruction Baselines Performance on Hidden Points (20% Masked):")
    for _, row in recon_metrics_df.iterrows():
        print(f"      - {row['model']:22s} -> Hidden RMSE: {row['hidden_rmse']:5.2f} | MAE: {row['hidden_mae']:5.2f} | R2: {row['hidden_r2']:0.4f}")

    # Save reconstruction predictions
    recon_pred_df = test_df[["timestamp", "site", "bam_reference"]].copy()
    recon_pred_df["calibrated_pm25"] = test_calibrated_pm25
    recon_pred_df["is_observed"] = recon_benchmark_mask.astype(int)
    for model_name, preds in recon_predictions.items():
        col_clean = "recon_" + model_name.lower().replace(" ", "_")
        recon_pred_df[col_clean] = preds
    recon_pred_path = predictions_dir / "reconstruction_predictions.csv"
    recon_pred_df.to_csv(recon_pred_path, index=False)
    print(f"      Saved reconstruction predictions to: {recon_pred_path.name}")

    # 6. MODULE C & D: Missingness & Propagation-Gap Experiments
    print("[5/7] Running Module D: Error Propagation across Gap Durations (1h, 3h, 6h, 12h, 24h)...")
    propagation_df, prop_metadata = evaluate_propagation_across_gaps(
        test_df,
        calibration_model=best_calib_bundle["model"],
        calib_feature_cols=best_calib_bundle["feature_cols"],
        covariate_cols=covariate_cols,
        gap_hours_list=[1, 3, 6, 12, 24],
        missing_rate=0.20,
        random_state=random_seed,
    )
    prop_csv_path = reports_dir / "propagation_gap_metrics.csv"
    propagation_df.to_csv(prop_csv_path, index=False)
    print(f"      Saved propagation gap metrics to: {prop_csv_path.name}")

    # Stratified regimes and mechanisms
    df_mech, df_hum = evaluate_propagation_by_regimes_and_mechanisms(
        test_df,
        calibration_model=best_calib_bundle["model"],
        calib_feature_cols=best_calib_bundle["feature_cols"],
        covariate_cols=covariate_cols,
        missing_rate=0.20,
        random_state=random_seed,
    )
    hum_csv_path = reports_dir / "humidity_metrics.csv"
    df_hum.to_csv(hum_csv_path, index=False)
    print(f"      Saved humidity regime metrics to: {hum_csv_path.name}")

    # Site-wise reconstruction breakdown
    site_rows = []
    for site_name, s_df in test_df.groupby("site", sort=False):
        site_mask = recon_benchmark_mask[test_df["site"] == site_name]
        hidden_s = ~site_mask
        bam_s = s_df["bam_reference"].to_numpy(dtype=float)
        for m_name, p_vals in recon_predictions.items():
            p_s = p_vals[test_df["site"] == site_name]
            m_s = compute_regression_metrics(bam_s[hidden_s], p_s[hidden_s])
            site_rows.append({
                "site": site_name,
                "model": m_name,
                "test_mae": m_s["mae"],
                "test_rmse": m_s["rmse"],
                "test_r2": m_s["r2"],
                "test_mbe": m_s["mbe"],
            })
    site_metrics_df = pd.DataFrame(site_rows)
    site_csv_path = reports_dir / "site_wise_metrics.csv"
    site_metrics_df.to_csv(site_csv_path, index=False)
    print(f"      Saved site-wise metrics to: {site_csv_path.name}")

    # Baseline overall summary table
    summary_rows = []
    # Add raw sensor metrics on test set
    for ch in ["pa_cf1", "pa_atm", "pa_alt"]:
        raw_m = compute_regression_metrics(test_df["bam_reference"], test_df[ch])
        summary_rows.append({
            "stage": "Raw Sensor",
            "component": ch.upper(),
            "rmse": raw_m["rmse"],
            "mae": raw_m["mae"],
            "r2": raw_m["r2"],
            "mbe": raw_m["mbe"],
        })
    # Add best calibration model
    summary_rows.append({
        "stage": "Stage-1 Calibration",
        "component": f"{best_calib_bundle['algorithm']} ({best_calib_bundle['feature_set']})",
        "rmse": best_calib_bundle["test_rmse"],
        "mae": best_calib_bundle["test_mae"],
        "r2": best_calib_bundle["test_r2"],
        "mbe": 0.0,
    })
    # Add best reconstruction baseline on hidden
    best_recon_row = recon_metrics_df.iloc[0]
    summary_rows.append({
        "stage": "Stage-2 Reconstruction",
        "component": f"{best_recon_row['model']} (Hidden Points)",
        "rmse": best_recon_row["hidden_rmse"],
        "mae": best_recon_row["hidden_mae"],
        "r2": best_recon_row["hidden_r2"],
        "mbe": best_recon_row["hidden_mbe"],
    })
    baseline_summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = reports_dir / "baseline_summary.csv"
    baseline_summary_df.to_csv(summary_csv_path, index=False)
    print(f"      Saved overall baseline summary to: {summary_csv_path.name}")

    # 7. Generate Publication Figures
    print("[6/7] Generating publication figures in outputs/figures/...")
    # Raw vs BAM
    site_test_dfs = {s: test_df[test_df["site"] == s] for s in test_df["site"].unique()}
    plot_sensor_vs_bam_scatter(site_test_dfs, figures_dir / "fig_raw_vs_bam.png")
    # Calibrated vs BAM
    plot_calibrated_vs_bam(test_df, test_calibrated_pm25, figures_dir / "fig_calibrated_vs_bam.png")
    # Calibration Comparison
    plot_calibration_comparison(calib_results_df, figures_dir / "fig_calibration_comparison.png")
    # Reconstruction Comparison
    plot_reconstruction_comparison(recon_metrics_df, figures_dir / "fig_reconstruction_comparison.png")
    # Actual vs Reconstructed time series
    best_recon_col = "recon_" + best_recon_row["model"].lower().replace(" ", "_")
    plot_actual_vs_reconstructed_timeseries(
        recon_pred_df,
        bam_col="bam_reference",
        calib_col="calibrated_pm25",
        recon_col=best_recon_col,
        mask=recon_benchmark_mask,
        site_name="Delhi",
        output_path=figures_dir / "fig_actual_vs_reconstructed.png",
    )
    # Error vs gap length
    plot_error_vs_gap_length(propagation_df, figures_dir / "fig_error_vs_gap_length.png")
    # Site-wise performance
    plot_site_wise_performance(site_metrics_df, figures_dir / "fig_site_wise_performance.png")
    # Humidity-conditioned error
    plot_humidity_conditioned_error(df_hum, figures_dir / "fig_humidity_conditioned_error.png")
    print("      All 8 publication figures generated successfully.")

    # 8. Render HTML Research Reports
    print("[7/7] Compiling HTML research reports in outputs/reports/...")
    calib_html_path = generate_calibration_report_html(
        calib_results_df, best_calib_bundle, reports_dir / "calibration_baseline_report.html"
    )
    recon_html_path = generate_reconstruction_report_html(
        recon_metrics_df, propagation_df, df_hum, site_metrics_df, reports_dir / "reconstruction_baseline_report.html"
    )
    print(f"      - {calib_html_path.name}")
    print(f"      - {recon_html_path.name}")

    # Save experiment logs and config snapshot
    config_snapshot = {
        "experiment_name": "checkpoint2_baselines",
        "random_seed": random_seed,
        "split_counts": split_info,
        "best_calibration_model": {
            "algorithm": best_calib_bundle["algorithm"],
            "feature_set": best_calib_bundle["feature_set"],
            "feature_cols": best_calib_bundle["feature_cols"],
            "val_rmse": float(best_calib_bundle["val_rmse"]),
            "test_rmse": float(best_calib_bundle["test_rmse"]),
            "test_r2": float(best_calib_bundle["test_r2"]),
        },
        "best_reconstruction_baseline": {
            "model": best_recon_row["model"],
            "hidden_rmse": float(best_recon_row["hidden_rmse"]),
            "hidden_mae": float(best_recon_row["hidden_mae"]),
            "hidden_r2": float(best_recon_row["hidden_r2"]),
        },
    }
    with open(logs_dir / "experiment_configurations.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    calib_results_df.to_csv(logs_dir / "training_logs.csv", index=False)
    print("=" * 75)
    print("CHECKPOINT 2 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
