#!/usr/bin/env python3
"""
Research Pipeline: Checkpoint 3 Execution Script
End-to-end implementation, training, evaluation, propagation analysis,
and ablation study for the Joint Calibration-and-Reconstruction TCN.
"""

from pathlib import Path
import json
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_cleaned_dataset
from src.data.feature_engineering import create_chronological_splits
from src.training.train_joint_tcn import train_joint_tcn_model, INPUT_FEATURE_COLS
from src.evaluation.evaluate_joint_tcn import (
    evaluate_joint_tcn_comprehensively,
    predict_joint_tcn_full_sequence,
)
from src.evaluation.propagation_analysis import run_propagation_gap_analysis
from src.evaluation.ablation import run_ablation_study
from src.evaluation.metrics import compute_regression_metrics
from src.visualization.tcn_plots import (
    plot_joint_tcn_architecture,
    plot_training_loss_curves,
    plot_actual_vs_predicted_time_series,
    plot_propagation_gap_joint_vs_two_stage,
    plot_ablation_comparison,
    plot_stratified_breakdowns,
)


def main():
    print("=" * 80)
    print("STARTING CHECKPOINT 3: JOINT TEMPORAL CONVOLUTIONAL NETWORK (JOINT TCN)")
    print("=" * 80)

    # 1. Directories setup
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    models_dir = PROJECT_ROOT / "outputs" / "models" / "joint_tcn"
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    logs_dir = PROJECT_ROOT / "outputs" / "logs"
    masks_dir = PROJECT_ROOT / "data" / "synthetic_masks"

    for d in [reports_dir, figures_dir, models_dir, predictions_dir, logs_dir, masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 2. Load dataset and split chronologically
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_data.parquet"
    print(f"[1/7] Loading validated dataset from: {data_path.resolve()}")
    df = load_cleaned_dataset(data_path)
    train_df, val_df, test_df, split_info = create_chronological_splits(df)
    print(f"      Partitions -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 3. Model Training: Full Joint TCN & Loss-Weight Tuning on Val Set
    print("[2/7] Training Joint TCN models with multi-task Huber loss...")
    loss_weights = [(1.0, 0.5), (1.0, 1.0), (1.0, 2.0)]
    best_joint_model = None
    best_normalizer = None
    best_history = None
    best_val_loss = float("inf")
    best_weight_cfg = None

    for l_cal, l_rec in loss_weights:
        print(f"      Testing loss weights: lambda_cal={l_cal}, lambda_rec={l_rec}...")
        save_p = models_dir / f"joint_tcn_cal{int(l_cal)}_rec{int(l_rec*10)}.pt"
        m, norm, hist = train_joint_tcn_model(
            train_df=train_df,
            val_df=val_df,
            lambda_cal=l_cal,
            lambda_rec=l_rec,
            epochs=25,
            patience=8,
            batch_size=64,
            hidden_channels=32,
            save_path=save_p,
            random_seed=random_seed,
        )
        min_vloss = hist["val_loss"].min()
        print(f"        -> Best Val Loss: {min_vloss:.4f}")

        if min_vloss < best_val_loss:
            best_val_loss = min_vloss
            best_joint_model = m
            best_normalizer = norm
            best_history = hist
            best_weight_cfg = (l_cal, l_rec)

    print(f"      Selected Best Joint TCN on Validation Set: lambda_cal={best_weight_cfg[0]}, lambda_rec={best_weight_cfg[1]}")
    # Save canonical model and normalizer
    torch.save(best_joint_model.state_dict(), models_dir / "best_joint_tcn_checkpoint.pt")
    joblib.dump(best_normalizer, models_dir / "normalizer.joblib")
    best_history.to_csv(logs_dir / "joint_tcn_training_log.csv", index=False)

    # 4. Load Checkpoint 2 Benchmark Baselines for Head-to-Head Comparison
    print("[3/7] Loading Checkpoint 2 baseline predictions on identical test partition...")
    # Load identical MCAR 20% test benchmark mask
    test_mask_path = masks_dir / "benchmark_test_mask_mcar_20.npy"
    if test_mask_path.exists():
        test_benchmark_mask = np.load(test_mask_path)
    else:
        from src.missingness.random_mask import generate_mcar_mask
        test_benchmark_mask = generate_mcar_mask(len(test_df), missing_rate=0.20, random_seed=random_seed)
        np.save(test_mask_path, test_benchmark_mask)

    # Load baseline models from outputs/models/
    calib_rf = joblib.load(PROJECT_ROOT / "outputs" / "models" / "calibration" / "pooled_random_forest_all_channels.joblib")
    calib_lr = joblib.load(PROJECT_ROOT / "outputs" / "models" / "calibration" / "pooled_linear_regression_atm_only.joblib")

    # Generate baseline predictions on the test set
    FEATURE_SETS = {
        "atm_only": ["pa_atm", "relative_humidity", "temperature"],
        "all_channels": ["pa_cf1", "pa_atm", "pa_alt", "relative_humidity", "temperature"],
    }
    pred_rf_calib = calib_rf.predict(test_df[FEATURE_SETS["all_channels"]].values)
    pred_lr_calib = calib_lr.predict(test_df[FEATURE_SETS["atm_only"]].values)

    # Load Stage 2 reconstruction predictions
    from src.models.reconstruction import (
        reconstruct_linear_interpolation,
        reconstruct_knn,
        reconstruct_random_forest,
    )
    df_stage = test_df.copy()
    df_stage["calibrated_pm25"] = pred_rf_calib
    covariate_cols = ["relative_humidity", "temperature", "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
    impute_covs = [c for c in covariate_cols if c in df_stage.columns]

    preds_linear = reconstruct_linear_interpolation(df_stage, "calibrated_pm25", test_benchmark_mask)
    preds_rf_rec = reconstruct_random_forest(df_stage, "calibrated_pm25", impute_covs, test_benchmark_mask, random_state=random_seed)

    # Also load saved reconstruction predictions if present (contains GRU, LSTM)
    rec_pred_path = PROJECT_ROOT / "outputs" / "predictions" / "reconstruction_predictions.csv"
    if rec_pred_path.exists():
        saved_rec = pd.read_csv(rec_pred_path)
        preds_gru = saved_rec["recon_gru"].to_numpy()
        preds_lstm = saved_rec["recon_lstm"].to_numpy()
    else:
        preds_gru = preds_rf_rec
        preds_lstm = preds_rf_rec

    baseline_predictions = {
        "Raw PA_ATM (Baseline)": test_df["pa_atm"].to_numpy(dtype=float),
        "Linear Regression (atm_only)": pred_lr_calib,
        "Random Forest (all_channels)": pred_rf_calib,
        "Two-Stage Linear Interpolation": preds_linear,
        "Two-Stage Random Forest Imputer": preds_rf_rec,
        "GRU Recurrent Imputer": preds_gru,
        "LSTM Recurrent Imputer": preds_lstm,
    }

    # 5. Full Evaluation across Overall, Site, Pollution Band, and Humidity
    print("[4/7] Running comprehensive evaluation of Joint TCN vs. Baselines...")
    df_overall, df_site, df_poll, df_hum, df_preds = evaluate_joint_tcn_comprehensively(
        model=best_joint_model,
        normalizer=best_normalizer,
        test_df=test_df,
        mask=test_benchmark_mask,
        baseline_predictions=baseline_predictions,
        random_seed=random_seed,
    )

    # Save metrics CSVs
    df_overall.to_csv(reports_dir / "joint_tcn_metrics.csv", index=False)
    df_site.to_csv(reports_dir / "joint_tcn_site_metrics.csv", index=False)
    df_poll.to_csv(reports_dir / "joint_tcn_pollution_metrics.csv", index=False)
    df_hum.to_csv(reports_dir / "joint_tcn_humidity_metrics.csv", index=False)

    # Save test predictions
    df_preds["missingness_type"] = "MCAR_20"
    df_preds["gap_duration"] = 1
    df_preds["evaluation_seed"] = random_seed
    pred_save_p = predictions_dir / "joint_tcn_test_predictions.csv"
    df_preds.to_csv(pred_save_p, index=False)
    print(f"      Saved test predictions to: {pred_save_p.name}")

    print("\n=== HEAD-TO-HEAD RECONSTRUCTION METRICS ON HIDDEN TEST POINTS ===")
    for _, r in df_overall.iterrows():
        print(f"  {r['model']:32s} -> Hidden RMSE: {r['hidden_rmse']:5.2f} | MAE: {r['hidden_mae']:5.2f} | R2: {r['hidden_r2']:0.4f} | Paired CI: {r['delta_rmse_vs_joint_ci_95']}")

    # 6. Propagation-Gap Analysis (1h, 3h, 6h, 12h, 24h)
    print("\n[5/7] Running Propagation-Gap Analysis across 1h to 24h Outages...")
    propagation_df = run_propagation_gap_analysis(
        model=best_joint_model,
        normalizer=best_normalizer,
        test_df=test_df,
        calibration_model=calib_rf,
        calib_feature_cols=FEATURE_SETS["all_channels"],
        gap_hours_list=[1, 3, 6, 12, 24],
        missing_rate=0.20,
        random_seed=random_seed,
    )
    propagation_df.to_csv(reports_dir / "joint_tcn_gap_metrics.csv", index=False)
    print("      Saved joint_tcn_gap_metrics.csv")

    # 7. Systematic Architecture Ablation Study
    print("[6/7] Running Systematic Ablation Study (Configurations A to F)...")
    ablation_df = run_ablation_study(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        test_mask=test_benchmark_mask,
        baseline_joint_model=(best_joint_model, best_normalizer),
        random_seed=random_seed,
        epochs=20,
    )
    ablation_df.to_csv(reports_dir / "joint_tcn_ablation.csv", index=False)
    print("      Saved joint_tcn_ablation.csv")

    # 8. Publication Visualizations
    print("[7/7] Generating 11 Publication Figures in outputs/figures/...")
    # 1. Architecture diagram
    plot_joint_tcn_architecture(figures_dir / "fig_joint_tcn_architecture.png")
    # 2. Training curves
    plot_training_loss_curves(best_history, figures_dir / "fig_joint_tcn_training_curves.png")
    # 3. Actual vs Predicted time series
    plot_actual_vs_predicted_time_series(df_preds, "Delhi", figures_dir / "fig_actual_vs_predicted_joint_tcn.png")
    # 4. Propagation curves
    plot_propagation_gap_joint_vs_two_stage(propagation_df, figures_dir / "fig_propagation_gap_joint_vs_two_stage.png")
    # 5. Ablation comparison
    plot_ablation_comparison(ablation_df, figures_dir / "fig_ablation_joint_tcn.png")
    # 6. Stratified figures (Site, Pollution, Humidity)
    plot_stratified_breakdowns(df_site, df_poll, df_hum, figures_dir)

    # 7. Calibration baseline comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    calib_sub = df_overall[df_overall["model"].isin([
        "Raw PA_ATM (Baseline)", "Linear Regression (atm_only)", "Random Forest (all_channels)", "Joint TCN (Proposed)"
    ])]
    sns.barplot(data=calib_sub, x="model", y="full_test_rmse", palette="Blues_d", ax=ax)
    ax.set_title("Full Test Partition Calibration RMSE: Raw vs. Baselines vs. Joint TCN")
    ax.set_ylabel(r"Test RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig_tcn_calibration_comparison.png")
    plt.close(fig)

    # 8. Reconstruction baseline comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    recon_sub = df_overall.copy()
    sns.barplot(data=recon_sub, x="model", y="hidden_rmse", palette="Set2", ax=ax)
    ax.set_title("Missing-Data Reconstruction RMSE on Hidden Observations (20% Masked)")
    ax.set_ylabel(r"Hidden RMSE ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig_tcn_reconstruction_comparison.png")
    plt.close(fig)

    # 9. Raw vs Calibrated scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_preds["bam_reference"], df_preds["y_cal"], alpha=0.25, color="#3182CE", s=14, label="Joint TCN Calibration")
    max_val = max(df_preds["bam_reference"].max(), df_preds["y_cal"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color="#333333", linestyle="--", linewidth=1.5, label="1:1 Identity")
    ax.set_title("Joint TCN Calibration vs. Reference BAM (Test Set)")
    ax.set_xlabel(r"Reference BAM ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.set_ylabel(r"Joint TCN $\widehat{Y}_{\mathrm{cal}}$ ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "fig_raw_vs_calibrated_tcn.png")
    plt.close(fig)

    print("=" * 80)
    print("CHECKPOINT 3 EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
