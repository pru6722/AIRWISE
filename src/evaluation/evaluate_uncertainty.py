"""
Module: src.evaluation.evaluate_uncertainty
Master orchestration engine for Checkpoint 4:
- Four-way chronological data splitting (60% Train, 10% Val-Model, 10% Calib-Conformal, 20% Test)
- Retrains frozen Joint TCN on Train and Val-Model; saves model and normalizer
- Fits Ordinary Split Conformal, Regime-Conditioned Conformal, and Temporal Block Conformal
- Evaluates 90% and 95% intervals on the untouched test partition (N=3,720)
- Computes comprehensive metrics across site, humidity, predicted-pollution, missingness types, gap lengths, and extreme pollution
- Saves predictions, metrics CSVs, reports, and publication figures
"""

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import torch
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from src.data.preprocessing import load_cleaned_dataset
from src.uncertainty.split_manager import create_conformal_splits
from src.training.train_joint_tcn import train_joint_tcn_model, DataNormalizer, INPUT_FEATURE_COLS
from src.evaluation.evaluate_joint_tcn import predict_joint_tcn_full_sequence
from src.uncertainty.conformal import ConformalCalibrator, compute_uncertainty_metrics
from src.uncertainty.regime_conformal import (
    RegimeConformalCalibrator,
    assign_humidity_regime,
    assign_predicted_pollution_regime,
)
from src.uncertainty.block_conformal import (
    TemporalBlockConformalCalibrator,
    construct_temporal_blocks,
)
from src.missingness.random_mask import generate_mcar_mask
from src.missingness.block_mask import generate_block_outage_mask
from src.missingness.regime_mask import generate_humidity_dependent_mask, generate_pollution_dependent_mask
from src.visualization.conformal_plots import (
    plot_actual_vs_predicted_with_intervals,
    plot_coverage_comparison,
    plot_interval_width_by_humidity,
    plot_interval_width_by_site,
    plot_coverage_by_predicted_pollution,
    plot_coverage_by_missingness_type,
    plot_coverage_vs_gap_duration,
    plot_delhi_extreme_pollution_conformal,
    plot_ordinary_vs_block_conformal,
    plot_interval_width_vs_prediction_error,
)


def run_checkpoint4_pipeline():
    print("=" * 80)
    print("STARTING CHECKPOINT 4: REGIME-CONDITIONED CONFORMAL UNCERTAINTY ESTIMATION")
    print("=" * 80)

    # 1. Setup Directories
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    models_dir = PROJECT_ROOT / "outputs" / "models" / "conformal"
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    logs_dir = PROJECT_ROOT / "outputs" / "logs"
    masks_dir = PROJECT_ROOT / "data" / "synthetic_masks"

    for d in [reports_dir, figures_dir, models_dir, predictions_dir, logs_dir, masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 2. Strict Conformal Data Splitting (60/10/10/20)
    print("[1/6] Slicing strict 4-way chronological partitions (Train, Val-Model, Calib-Conformal, Test)...")
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_data.parquet"
    df = load_cleaned_dataset(data_path)
    train_df, val_model_df, calib_conformal_df, test_df, split_summary = create_conformal_splits(df)

    print(f"      Partitions -> Train: {len(train_df):,} | Val-Model: {len(val_model_df):,} | Calib-Conformal: {len(calib_conformal_df):,} | Test: {len(test_df):,}")
    with open(models_dir / "split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    # 3. Retrain Joint TCN Strictly on Train and Val-Model
    print("[2/6] Retraining Joint TCN from scratch on Train (60%) and Val-Model (10%)...")
    frozen_model_path = models_dir / "frozen_joint_tcn_checkpoint.pt"
    frozen_normalizer_path = models_dir / "normalizer.joblib"

    model, normalizer, train_hist = train_joint_tcn_model(
        train_df=train_df,
        val_df=val_model_df,
        lambda_cal=1.0,
        lambda_rec=0.5,
        hidden_channels=32,
        epochs=25,
        patience=8,
        batch_size=64,
        save_path=frozen_model_path,
        random_seed=random_seed,
    )
    joblib.dump(normalizer, frozen_normalizer_path)
    train_hist.to_csv(logs_dir / "retrained_joint_tcn_log.csv", index=False)
    print(f"      Retrained Joint TCN frozen and saved to {frozen_model_path.name}")

    # 4. Calibration Phase: Conformal Calibration on Holdout Set (N=1,858)
    print("[3/6] Generating calibration predictions and fitting conformal calibrators (N=1,858)...")
    calib_conformal_df = calib_conformal_df.reset_index(drop=True)

    # Synthetic calibration mask: 20% MCAR to evaluate reconstruction residuals
    rng_cal = np.random.default_rng(random_seed + 100)
    calib_mask = rng_cal.uniform(0.0, 1.0, size=len(calib_conformal_df)) >= 0.20

    y_cal_calib, y_rec_calib, y_fin_calib = predict_joint_tcn_full_sequence(
        model=model,
        normalizer=normalizer,
        df=calib_conformal_df,
        mask=calib_mask,
    )
    calib_bam = calib_conformal_df["bam_reference"].to_numpy(dtype=float)

    # Observed points for calibration head; Hidden points for reconstruction head
    calib_obs_idx = calib_mask
    calib_hid_idx = ~calib_mask

    # A. Ordinary Split Conformal Calibrators
    calibrator_cal = ConformalCalibrator(name="OrdinarySplit_CalHead")
    calibrator_cal.fit(calib_bam[calib_obs_idx], y_cal_calib[calib_obs_idx], alphas=[0.10, 0.05])

    calibrator_rec = ConformalCalibrator(name="OrdinarySplit_RecHead")
    calibrator_rec.fit(calib_bam[calib_hid_idx], y_rec_calib[calib_hid_idx], alphas=[0.10, 0.05])

    calibrator_pooled = ConformalCalibrator(name="OrdinarySplit_Composite")
    calibrator_pooled.fit(calib_bam, y_fin_calib, alphas=[0.10, 0.05])

    # Save quantiles
    quantiles_summary = {
        "cal_head_q90": calibrator_cal.quantiles[0.10],
        "cal_head_q95": calibrator_cal.quantiles[0.05],
        "rec_head_q90": calibrator_rec.quantiles[0.10],
        "rec_head_q95": calibrator_rec.quantiles[0.05],
        "composite_q90": calibrator_pooled.quantiles[0.10],
        "composite_q95": calibrator_pooled.quantiles[0.05],
    }

    # B. Regime-Conditioned Calibrators
    # 1. Humidity-conditioned
    calib_rh = calib_conformal_df["relative_humidity"].to_numpy(dtype=float)
    calib_hum_regimes = assign_humidity_regime(calib_rh)
    calibrator_hum = RegimeConformalCalibrator(conditioning_variable="humidity", min_group_size=30, alphas=[0.10, 0.05])
    calibrator_hum.fit(calib_bam, y_fin_calib, calib_hum_regimes)

    # 2. Predicted-pollution-conditioned
    calib_poll_regimes = assign_predicted_pollution_regime(y_fin_calib)
    calibrator_poll = RegimeConformalCalibrator(conditioning_variable="predicted_pollution", min_group_size=30, alphas=[0.10, 0.05])
    calibrator_poll.fit(calib_bam, y_fin_calib, calib_poll_regimes)

    # 3. Site-conditioned
    calib_sites = calib_conformal_df["site"].to_numpy()
    calibrator_site = RegimeConformalCalibrator(conditioning_variable="site", min_group_size=30, alphas=[0.10, 0.05])
    calibrator_site.fit(calib_bam, y_fin_calib, calib_sites)

    # C. Temporal Block Conformal Calibrators
    calib_conformal_df["y_pred"] = y_fin_calib
    block_calibrators = {}
    for b_size in [6, 12, 24]:
        b_cal = TemporalBlockConformalCalibrator(block_size=b_size, alphas=[0.10, 0.05])
        b_cal.fit(calib_conformal_df, y_true_col="bam_reference", y_pred_col="y_pred")
        block_calibrators[b_size] = b_cal
        quantiles_summary[f"block_{b_size}h_q90"] = b_cal.block_quantiles[0.10]
        quantiles_summary[f"block_{b_size}h_q95"] = b_cal.block_quantiles[0.05]

    with open(models_dir / "conformal_quantiles.json", "w") as f:
        json.dump(quantiles_summary, f, indent=2)
    print(f"      Fitted Quantiles: Ordinary 90%={calibrator_pooled.quantiles[0.10]:.2f}, 95%={calibrator_pooled.quantiles[0.05]:.2f}")

    # 5. Test Evaluation Phase: Predictions on Untouched Test Partition (N=3,720)
    print("[4/6] Evaluating conformal prediction intervals on untouched test partition (N=3,720)...")
    test_df = test_df.reset_index(drop=True)
    test_mask_path = masks_dir / "benchmark_test_mask_mcar_20.npy"
    test_mask = np.load(test_mask_path)
    test_bam = test_df["bam_reference"].to_numpy(dtype=float)

    y_cal_test, y_rec_test, y_fin_test = predict_joint_tcn_full_sequence(
        model=model,
        normalizer=normalizer,
        df=test_df,
        mask=test_mask,
    )

    test_rh = test_df["relative_humidity"].to_numpy(dtype=float)
    test_hum_regimes = assign_humidity_regime(test_rh)
    test_poll_regimes = assign_predicted_pollution_regime(y_fin_test)
    test_sites = test_df["site"].to_numpy()

    # Generate Intervals for all methods
    # Method 1: Ordinary Split Conformal (Head-Specific: Cal on Obs, Rec on Hid)
    low_ord_90 = np.empty(len(test_df), dtype=float)
    upp_ord_90 = np.empty(len(test_df), dtype=float)
    low_ord_95 = np.empty(len(test_df), dtype=float)
    upp_ord_95 = np.empty(len(test_df), dtype=float)

    # Observed points
    l_cal_90, u_cal_90 = calibrator_cal.predict_interval(y_cal_test[test_mask], alpha=0.10)
    l_cal_95, u_cal_95 = calibrator_cal.predict_interval(y_cal_test[test_mask], alpha=0.05)
    low_ord_90[test_mask] = l_cal_90
    upp_ord_90[test_mask] = u_cal_90
    low_ord_95[test_mask] = l_cal_95
    upp_ord_95[test_mask] = u_cal_95

    # Hidden points
    l_rec_90, u_rec_90 = calibrator_rec.predict_interval(y_rec_test[~test_mask], alpha=0.10)
    l_rec_95, u_rec_95 = calibrator_rec.predict_interval(y_rec_test[~test_mask], alpha=0.05)
    low_ord_90[~test_mask] = l_rec_90
    upp_ord_90[~test_mask] = u_rec_90
    low_ord_95[~test_mask] = l_rec_95
    upp_ord_95[~test_mask] = u_rec_95

    # Method 2: Humidity-Conditioned Conformal
    low_hum_90, upp_hum_90, _ = calibrator_hum.predict_intervals(y_fin_test, test_hum_regimes, alpha=0.10)
    low_hum_95, upp_hum_95, _ = calibrator_hum.predict_intervals(y_fin_test, test_hum_regimes, alpha=0.05)

    # Method 3: Predicted-Pollution-Conditioned Conformal
    low_poll_90, upp_poll_90, _ = calibrator_poll.predict_intervals(y_fin_test, test_poll_regimes, alpha=0.10)
    low_poll_95, upp_poll_95, _ = calibrator_poll.predict_intervals(y_fin_test, test_poll_regimes, alpha=0.05)

    # Method 4: Site-Conditioned Conformal
    low_site_90, upp_site_90, _ = calibrator_site.predict_intervals(y_fin_test, test_sites, alpha=0.10)
    low_site_95, upp_site_95, _ = calibrator_site.predict_intervals(y_fin_test, test_sites, alpha=0.05)

    # Method 5: Block Conformal (12h default)
    low_blk12_90, upp_blk12_90 = block_calibrators[12].predict_intervals(y_fin_test, alpha=0.10)
    low_blk12_95, upp_blk12_95 = block_calibrators[12].predict_intervals(y_fin_test, alpha=0.05)

    # Assemble Test Predictions DataFrame
    pred_export_df = pd.DataFrame({
        "timestamp": test_df["timestamp"],
        "site": test_df["site"],
        "true_bam": test_bam,
        "point_prediction": y_fin_test,
        "lower_90": low_ord_90,
        "upper_90": upp_ord_90,
        "lower_95": low_ord_95,
        "upper_95": upp_ord_95,
        "prediction_head": np.where(test_mask, "Calibration", "Reconstruction"),
        "humidity_regime": test_hum_regimes,
        "predicted_pollution_regime": test_poll_regimes,
        "missingness_type": "MCAR_20",
        "gap_duration": 1,
        "conformal_method": "OrdinarySplit_DualHead",
        "fallback_group_identifier": "None",
    })
    pred_save_path = predictions_dir / "conformal_test_predictions.csv"
    pred_export_df.to_csv(pred_save_path, index=False)
    print(f"      Saved conformal predictions to {pred_save_path.name}")

    # 6. Metrics Computation & Tables Generation
    print("[5/6] Computing comprehensive uncertainty evaluation tables...")
    methods_dict = {
        "Ordinary Split (Dual Head)": (low_ord_90, upp_ord_90, low_ord_95, upp_ord_95),
        "Humidity-Conditioned": (low_hum_90, upp_hum_90, low_hum_95, upp_hum_95),
        "Predicted-Pollution-Conditioned": (low_poll_90, upp_poll_90, low_poll_95, upp_poll_95),
        "Site-Conditioned": (low_site_90, upp_site_90, low_site_95, upp_site_95),
        "Temporal Block Conformal (12h)": (low_blk12_90, upp_blk12_90, low_blk12_95, upp_blk12_95),
    }

    # Table 1: Overall Coverage and Width Summary
    cov_rows = []
    for mname, (l90, u90, l95, u95) in methods_dict.items():
        m90 = compute_uncertainty_metrics(test_bam, y_fin_test, l90, u90, alpha=0.10)
        m95 = compute_uncertainty_metrics(test_bam, y_fin_test, l95, u95, alpha=0.05)
        cov_rows.append({
            "method": mname,
            "n_samples": m90["n_samples"],
            "cov_90": m90["empirical_coverage"],
            "cov_error_90": m90["coverage_error"],
            "mean_width_90": m90["mean_interval_width"],
            "median_width_90": m90["median_interval_width"],
            "winkler_90": m90["winkler_score"],
            "cov_95": m95["empirical_coverage"],
            "cov_error_95": m95["coverage_error"],
            "mean_width_95": m95["mean_interval_width"],
            "winkler_95": m95["winkler_score"],
            "point_rmse": m90["point_rmse"],
            "point_mae": m90["point_mae"],
            "point_mbe": m90["point_mbe"],
        })
    df_coverage = pd.DataFrame(cov_rows)
    df_coverage.to_csv(reports_dir / "conformal_coverage.csv", index=False)
    df_coverage.to_csv(reports_dir / "conformal_interval_width.csv", index=False)

    # Table 2: Site-wise Metrics
    site_rows = []
    for sname in ["Delhi", "Hamirpur", "Bangalore"]:
        s_idx = test_sites == sname
        for mname, (l90, u90, l95, u95) in methods_dict.items():
            m90 = compute_uncertainty_metrics(test_bam[s_idx], y_fin_test[s_idx], l90[s_idx], u90[s_idx], alpha=0.10)
            site_rows.append({
                "site": sname,
                "conformal_method": mname,
                "n_samples": m90["n_samples"],
                "empirical_coverage": m90["empirical_coverage"],
                "coverage_error": m90["coverage_error"],
                "mean_interval_width": m90["mean_interval_width"],
                "median_interval_width": m90["median_interval_width"],
                "winkler_score": m90["winkler_score"],
                "point_rmse": m90["point_rmse"],
                "point_mbe": m90["point_mbe"],
            })
    df_site_metrics = pd.DataFrame(site_rows)
    df_site_metrics.to_csv(reports_dir / "conformal_site_metrics.csv", index=False)

    # Table 3: Humidity-wise Metrics
    hum_rows = []
    for hname in ["Low (<40% RH)", "Medium (40-65% RH)", "High (>65% RH)"]:
        h_idx = test_hum_regimes == hname
        for mname, (l90, u90, l95, u95) in methods_dict.items():
            m90 = compute_uncertainty_metrics(test_bam[h_idx], y_fin_test[h_idx], l90[h_idx], u90[h_idx], alpha=0.10)
            hum_rows.append({
                "humidity_regime": hname,
                "conformal_method": mname,
                "n_samples": m90["n_samples"],
                "empirical_coverage": m90["empirical_coverage"],
                "coverage_error": m90["coverage_error"],
                "mean_interval_width": m90["mean_interval_width"],
                "median_interval_width": m90["median_interval_width"],
                "winkler_score": m90["winkler_score"],
                "point_rmse": m90["point_rmse"],
                "point_mbe": m90["point_mbe"],
            })
    df_hum_metrics = pd.DataFrame(hum_rows)
    df_hum_metrics.to_csv(reports_dir / "conformal_humidity_metrics.csv", index=False)

    # Table 4: Predicted-Pollution-wise Metrics
    poll_rows = []
    for pname in ["Low (<=60 ug/m3)", "Moderate (60-120 ug/m3)", "Severe (>120 ug/m3)"]:
        p_idx = test_poll_regimes == pname
        for mname, (l90, u90, l95, u95) in methods_dict.items():
            m90 = compute_uncertainty_metrics(test_bam[p_idx], y_fin_test[p_idx], l90[p_idx], u90[p_idx], alpha=0.10)
            poll_rows.append({
                "predicted_pollution_regime": pname,
                "conformal_method": mname,
                "n_samples": m90["n_samples"],
                "empirical_coverage": m90["empirical_coverage"],
                "coverage_error": m90["coverage_error"],
                "mean_interval_width": m90["mean_interval_width"],
                "median_interval_width": m90["median_interval_width"],
                "winkler_score": m90["winkler_score"],
                "point_rmse": m90["point_rmse"],
                "point_mbe": m90["point_mbe"],
            })
    df_poll_metrics = pd.DataFrame(poll_rows)
    df_poll_metrics.to_csv(reports_dir / "conformal_pollution_metrics.csv", index=False)

    # Table 5: Block Conformal Comparison (6h, 12h, 24h vs Ordinary Split)
    block_rows = []
    # Ordinary split baseline
    m_ord_blk = compute_uncertainty_metrics(test_bam, y_fin_test, low_ord_90, upp_ord_90, alpha=0.10)
    block_rows.append({
        "method": "Ordinary Split Conformal",
        "block_duration_hours": 1,
        "n_calib_blocks": len(calib_conformal_df),
        "empirical_coverage_90": m_ord_blk["empirical_coverage"],
        "mean_interval_width_90": m_ord_blk["mean_interval_width"],
        "winkler_score_90": m_ord_blk["winkler_score"],
    })
    for b_sz in [6, 12, 24]:
        b_c = block_calibrators[b_sz]
        l_b90, u_b90 = b_c.predict_intervals(y_fin_test, alpha=0.10)
        m_b = compute_uncertainty_metrics(test_bam, y_fin_test, l_b90, u_b90, alpha=0.10)
        block_rows.append({
            "method": f"Block Conformal ({b_sz}h)",
            "block_duration_hours": b_sz,
            "n_calib_blocks": b_c.n_calib_blocks,
            "empirical_coverage_90": m_b["empirical_coverage"],
            "mean_interval_width_90": m_b["mean_interval_width"],
            "winkler_score_90": m_b["winkler_score"],
        })
    df_block_comp = pd.DataFrame(block_rows)
    df_block_comp.to_csv(reports_dir / "conformal_block_comparison.csv", index=False)

    # Table 6: Gap-Duration Evaluation
    gap_rows = []
    for g_hours in [1, 3, 6, 12, 24]:
        g_mask = generate_block_outage_mask(test_df, gap_hours=g_hours, missing_rate=0.20, segment_col="segment_id", random_seed=random_seed)
        _, y_rec_g, y_fin_g = predict_joint_tcn_full_sequence(model, normalizer, test_df, g_mask)
        hid_g = ~g_mask

        # Ordinary Head-Specific
        l_g90, u_g90 = calibrator_rec.predict_interval(y_rec_g[hid_g], alpha=0.10)
        mg_ord = compute_uncertainty_metrics(test_bam[hid_g], y_rec_g[hid_g], l_g90, u_g90, alpha=0.10)

        gap_rows.append({
            "gap_hours": g_hours,
            "conformal_method": "Ordinary Split (Rec Head)",
            "hidden_samples": int(np.sum(hid_g)),
            "empirical_coverage": mg_ord["empirical_coverage"],
            "mean_interval_width": mg_ord["mean_interval_width"],
            "point_rmse": mg_ord["point_rmse"],
            "point_mbe": mg_ord["point_mbe"],
        })

        # Block Conformal (12h)
        l_gb90, u_gb90 = block_calibrators[12].predict_intervals(y_rec_g[hid_g], alpha=0.10)
        mg_blk = compute_uncertainty_metrics(test_bam[hid_g], y_rec_g[hid_g], l_gb90, u_gb90, alpha=0.10)
        gap_rows.append({
            "gap_hours": g_hours,
            "conformal_method": "Block Conformal (12h)",
            "hidden_samples": int(np.sum(hid_g)),
            "empirical_coverage": mg_blk["empirical_coverage"],
            "mean_interval_width": mg_blk["mean_interval_width"],
            "point_rmse": mg_blk["point_rmse"],
            "point_mbe": mg_blk["point_mbe"],
        })
    df_gap_metrics = pd.DataFrame(gap_rows)
    df_gap_metrics.to_csv(reports_dir / "conformal_gap_metrics.csv", index=False)

    # Table 7: Extreme-Pollution Failure Analysis (Delhi Ground-Truth BAM Bands)
    d_mask = test_sites == "Delhi"
    d_bam = test_bam[d_mask]
    d_pred = y_fin_test[d_mask]
    d_l90 = low_ord_90[d_mask]
    d_u90 = upp_ord_90[d_mask]
    d_l95 = low_ord_95[d_mask]
    d_u95 = upp_ord_95[d_mask]

    ext_bands = [
        ("<= 60 ug/m3", d_bam <= 60.0),
        ("60-120 ug/m3", (d_bam > 60.0) & (d_bam <= 120.0)),
        ("120-250 ug/m3", (d_bam > 120.0) & (d_bam <= 250.0)),
        ("> 250 ug/m3", d_bam > 250.0),
    ]

    ext_rows = []
    for blabel, bcond in ext_bands:
        n_b = int(np.sum(bcond))
        m90_b = compute_uncertainty_metrics(d_bam[bcond], d_pred[bcond], d_l90[bcond], d_u90[bcond], alpha=0.10)
        m95_b = compute_uncertainty_metrics(d_bam[bcond], d_pred[bcond], d_l95[bcond], d_u95[bcond], alpha=0.05)
        ext_rows.append({
            "bam_band": blabel,
            "sample_count": n_b,
            "point_rmse": m90_b["point_rmse"],
            "point_mbe": m90_b["point_mbe"],
            "cov_90": m90_b["empirical_coverage"],
            "cov_95": m95_b["empirical_coverage"],
            "mean_width_90": m90_b["mean_interval_width"],
            "median_width_90": m90_b["median_interval_width"],
            "winkler_score_90": m90_b["winkler_score"],
        })
    df_extreme = pd.DataFrame(ext_rows)
    df_extreme.to_csv(reports_dir / "conformal_extreme_pollution.csv", index=False)

    # 7. Generate All 10 Publication Figures
    print("[6/6] Generating 10 publication figures in outputs/figures/...")
    # Fig 1: Time Series
    plot_actual_vs_predicted_with_intervals(pred_export_df, "Delhi", figures_dir / "fig_actual_vs_predicted_conformal_intervals.png")
    # Fig 2: Coverage Comparison
    plot_coverage_comparison(df_coverage, figures_dir / "fig_coverage_comparison_90_95.png")
    # Fig 3: Interval Width by Humidity
    plot_interval_width_by_humidity(df_hum_metrics, figures_dir / "fig_interval_width_by_humidity.png")
    # Fig 4: Interval Width by Site
    plot_interval_width_by_site(df_site_metrics, figures_dir / "fig_interval_width_by_site.png")
    # Fig 5: Coverage by Predicted Pollution
    plot_coverage_by_predicted_pollution(df_poll_metrics, figures_dir / "fig_coverage_by_predicted_pollution.png")

    # Missingness Mechanisms DataFrame for Fig 6
    miss_rows = []
    # MCAR 20%
    m_mcar = compute_uncertainty_metrics(test_bam[~test_mask], y_fin_test[~test_mask], low_ord_90[~test_mask], upp_ord_90[~test_mask], alpha=0.10)
    miss_rows.append({"missingness_type": "MCAR (20%)", "conformal_method": "Ordinary Split", "empirical_coverage": m_mcar["empirical_coverage"]})
    # Block Outage 6h
    m_blk6 = generate_block_outage_mask(test_df, gap_hours=6, missing_rate=0.20, segment_col="segment_id", random_seed=random_seed)
    m_b6 = compute_uncertainty_metrics(test_bam[~m_blk6], y_fin_test[~m_blk6], low_ord_90[~m_blk6], upp_ord_90[~m_blk6], alpha=0.10)
    miss_rows.append({"missingness_type": "Block 6h (20%)", "conformal_method": "Ordinary Split", "empirical_coverage": m_b6["empirical_coverage"]})
    # Humidity-Dependent
    m_hum = generate_humidity_dependent_mask(test_df, missing_rate=0.20, random_seed=random_seed)
    m_h = compute_uncertainty_metrics(test_bam[~m_hum], y_fin_test[~m_hum], low_ord_90[~m_hum], upp_ord_90[~m_hum], alpha=0.10)
    miss_rows.append({"missingness_type": "Humidity-Dep", "conformal_method": "Ordinary Split", "empirical_coverage": m_h["empirical_coverage"]})
    # Pollution-Dependent
    m_poll = generate_pollution_dependent_mask(test_df, missing_rate=0.20, random_seed=random_seed)
    m_p = compute_uncertainty_metrics(test_bam[~m_poll], y_fin_test[~m_poll], low_ord_90[~m_poll], upp_ord_90[~m_poll], alpha=0.10)
    miss_rows.append({"missingness_type": "Pollution-Dep", "conformal_method": "Ordinary Split", "empirical_coverage": m_p["empirical_coverage"]})

    # Add Block Conformal to Fig 6
    miss_rows.append({"missingness_type": "MCAR (20%)", "conformal_method": "Block Conformal (12h)", "empirical_coverage": compute_uncertainty_metrics(test_bam[~test_mask], y_fin_test[~test_mask], low_blk12_90[~test_mask], upp_blk12_90[~test_mask], alpha=0.10)["empirical_coverage"]})
    miss_rows.append({"missingness_type": "Block 6h (20%)", "conformal_method": "Block Conformal (12h)", "empirical_coverage": compute_uncertainty_metrics(test_bam[~m_blk6], y_fin_test[~m_blk6], low_blk12_90[~m_blk6], upp_blk12_90[~m_blk6], alpha=0.10)["empirical_coverage"]})
    miss_rows.append({"missingness_type": "Humidity-Dep", "conformal_method": "Block Conformal (12h)", "empirical_coverage": compute_uncertainty_metrics(test_bam[~m_hum], y_fin_test[~m_hum], low_blk12_90[~m_hum], upp_blk12_90[~m_hum], alpha=0.10)["empirical_coverage"]})
    miss_rows.append({"missingness_type": "Pollution-Dep", "conformal_method": "Block Conformal (12h)", "empirical_coverage": compute_uncertainty_metrics(test_bam[~m_poll], y_fin_test[~m_poll], low_blk12_90[~m_poll], upp_blk12_90[~m_poll], alpha=0.10)["empirical_coverage"]})
    df_missing_types = pd.DataFrame(miss_rows)

    # Fig 6: Coverage by Missingness Type
    plot_coverage_by_missingness_type(df_missing_types, figures_dir / "fig_coverage_by_missingness_type.png")
    # Fig 7: Coverage vs Gap Duration
    plot_coverage_vs_gap_duration(df_gap_metrics, figures_dir / "fig_coverage_vs_gap_duration.png")
    # Fig 8: Delhi Extreme Pollution
    plot_delhi_extreme_pollution_conformal(df_extreme, figures_dir / "fig_delhi_extreme_pollution_conformal.png")
    # Fig 9: Ordinary vs Block Conformal
    plot_ordinary_vs_block_conformal(df_block_comp, figures_dir / "fig_ordinary_vs_block_conformal.png")
    # Fig 10: Interval Width vs Prediction Error
    plot_interval_width_vs_prediction_error(pred_export_df, figures_dir / "fig_interval_width_vs_prediction_error.png")

    # Copy all figures to Artifact directory
    artifact_fig_dir = Path("/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f/figures")
    for fig_p in figures_dir.glob("fig_*.png"):
        shutil.copy(fig_p, artifact_fig_dir / fig_p.name)
    print("      Copied all Checkpoint 4 figures to artifact directory.")

    print("=" * 80)
    print("CHECKPOINT 4 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_checkpoint4_pipeline()
