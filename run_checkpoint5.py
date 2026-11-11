"""
Master Orchestrator Script for Checkpoint 5:
PM2.5-Based CPCB Category Reliability Assessment Using 24-Hour Aggregation
and Conformal Uncertainty Intervals.

Executes Phases 0 to 12 deterministically:
1. Loads cleaned data and strictly four-way chronological partitions.
2. Generates aligned predictions for calibration (10%) and test (20%) using the frozen Joint TCN.
3. Constructs valid 24-hour windows across completeness thresholds (18/24, 20/24, 24/24).
4. Direct 24-hour conformal calibration (alpha=0.10 and alpha=0.05).
5. Comprehensive point evaluation and coverage evaluation.
6. CPCB category classification, confusion matrix, and multi-class metrics.
7. Uncertainty-aware category stability analysis (stable vs ambiguous).
8. Category boundary neighborhood analysis (distance to boundaries).
9. Site-wise and pollution-regime reliability diagnostics.
10. Generates all 10 publication figures.
11. Exports all prediction CSVs, metric CSVs, and final research report.
"""

from pathlib import Path
import os
import shutil
import json
import time
import numpy as np
import pandas as pd
import torch
import joblib

from src.models.joint_tcn import JointTCN
from src.uncertainty.split_manager import create_conformal_splits
from src.uncertainty.conformal import compute_conformal_quantile, compute_uncertainty_metrics
from src.evaluation.evaluate_uncertainty import predict_joint_tcn_full_sequence
from src.aqi.cpcb_categories import (
    CPCB_CATEGORIES,
    CPCB_BOUNDARY_VALUES,
    classify_pm25,
    get_distance_to_nearest_boundary,
    assess_interval_category_stability,
)
from src.aqi.windowing import construct_24h_windows
from src.aqi.category_evaluator import (
    compute_24h_point_metrics,
    compute_cpcb_classification_metrics,
    evaluate_uncertainty_aware_stability,
)
from src.visualization.cpcb_plots import (
    plot_24h_bam_vs_pred_scatter,
    plot_24h_timeseries_intervals,
    plot_cpcb_category_confusion_matrix,
    plot_category_precision_recall_f1,
    plot_category_stability_distribution,
    plot_category_stability_by_site,
    plot_reference_pm25_vs_interval_width,
    plot_boundary_distance_vs_category_error,
    plot_coverage_and_width_by_category,
    plot_completeness_sensitivity,
)


def main():
    start_time = time.time()
    print("=" * 80)
    print("STARTING CHECKPOINT 5: CPCB PM2.5 CATEGORY RELIABILITY ASSESSMENT")
    print("=" * 80)

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    models_dir = project_root / "outputs" / "models" / "conformal"
    predictions_dir = project_root / "outputs" / "predictions"
    metrics_dir = project_root / "outputs" / "metrics"
    reports_dir = project_root / "outputs" / "reports"
    figures_dir = project_root / "outputs" / "figures" / "checkpoint5"
    artifact_figs_dir = Path("/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f/figures")

    for d in [predictions_dir, metrics_dir, reports_dir, figures_dir, artifact_figs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Data and Partitions
    print("\n[Step 1/8] Loading dataset and constructing four-way chronological partitions...")
    df = pd.read_parquet(data_dir / "processed" / "cleaned_data.parquet")
    train_df, val_df, calib_df, test_df, split_info = create_conformal_splits(df)

    calib_df = calib_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"      Calibration Partition: {len(calib_df)} rows")
    print(f"      Test Partition:        {len(test_df)} rows (Delhi: {len(test_df[test_df['site']=='Delhi'])}, Bangalore: {len(test_df[test_df['site']=='Bangalore'])}, Hamirpur: {len(test_df[test_df['site']=='Hamirpur'])})")

    # 2. Load Frozen Joint TCN and Generate Aligned Predictions
    print("\n[Step 2/8] Generating aligned hourly predictions using frozen Joint TCN...")
    normalizer = joblib.load(models_dir / "normalizer.joblib")
    sd = torch.load(models_dir / "frozen_joint_tcn_checkpoint.pt", weights_only=False, map_location="cpu")
    model = JointTCN(input_dim=11, hidden_channels=32)
    model.load_state_dict(sd)
    model.eval()

    # Calibration predictions with synthetic MCAR 20% mask
    rng_cal = np.random.default_rng(42 + 100)
    calib_mask = rng_cal.uniform(0.0, 1.0, size=len(calib_df)) >= 0.20
    _, _, y_fin_calib = predict_joint_tcn_full_sequence(model, normalizer, calib_df, calib_mask)
    calib_df["point_prediction"] = y_fin_calib

    # Test predictions with identical benchmark mask
    test_mask = np.load(data_dir / "synthetic_masks" / "benchmark_test_mask_mcar_20.npy")
    _, _, y_fin_test = predict_joint_tcn_full_sequence(model, normalizer, test_df, test_mask)
    test_df["point_prediction"] = y_fin_test

    # 3. Construct Aligned 24-Hour Windows
    print("\n[Step 3/8] Constructing aligned 24-hour windows across completeness thresholds...")
    # Primary: non-overlapping >=18/24 (75% completeness)
    calib_win_df, calib_audit = construct_24h_windows(calib_df, min_valid_hours=18, non_overlapping=True, partition_name="calib_conformal")
    test_win_df, test_audit = construct_24h_windows(test_df, min_valid_hours=18, non_overlapping=True, partition_name="test")

    print(f"      Calibration Windows (>=18h): {len(calib_win_df)} valid out of {calib_audit['total_evaluated_windows']} evaluated")
    print(f"      Test Windows (>=18h):        {len(test_win_df)} valid out of {test_audit['total_evaluated_windows']} evaluated")

    # Sensitivity windowing: 20/24 and 24/24
    calib_win_20, _ = construct_24h_windows(calib_df, min_valid_hours=20, non_overlapping=True, partition_name="calib_conformal")
    test_win_20, _ = construct_24h_windows(test_df, min_valid_hours=20, non_overlapping=True, partition_name="test")
    calib_win_24, _ = construct_24h_windows(calib_df, min_valid_hours=24, non_overlapping=True, partition_name="calib_conformal")
    test_win_24, _ = construct_24h_windows(test_df, min_valid_hours=24, non_overlapping=True, partition_name="test")

    # Rolling windowing sensitivity
    calib_roll_df, _ = construct_24h_windows(calib_df, min_valid_hours=18, non_overlapping=False, partition_name="calib_conformal")
    test_roll_df, _ = construct_24h_windows(test_df, min_valid_hours=18, non_overlapping=False, partition_name="test")

    # 4. Direct 24-Hour Conformal Calibration
    print("\n[Step 4/8] Computing direct 24-hour finite-sample conformal quantiles...")
    calib_scores_24h = calib_win_df["abs_residual_24h"].values
    n_calib_win = len(calib_scores_24h)

    q90_24h = compute_conformal_quantile(calib_scores_24h, alpha=0.10)
    q95_24h = compute_conformal_quantile(calib_scores_24h, alpha=0.05)
    print(f"      24h Finite-Sample Quantiles (N_calib={n_calib_win}):")
    print(f"        q90_24h = {q90_24h:.2f} ug/m3 (Full interval width = {2*q90_24h:.2f} ug/m3)")
    print(f"        q95_24h = {q95_24h:.2f} ug/m3 (Full interval width = {2*q95_24h:.2f} ug/m3)")

    # Apply 24h quantiles to test windows
    test_win_df["lower_90"] = test_win_df["pred_pm25_24h"] - q90_24h
    test_win_df["upper_90"] = test_win_df["pred_pm25_24h"] + q90_24h
    test_win_df["lower_95"] = test_win_df["pred_pm25_24h"] - q95_24h
    test_win_df["upper_95"] = test_win_df["pred_pm25_24h"] + q95_24h
    test_win_df["interval_width_90"] = 2 * q90_24h
    test_win_df["interval_width_95"] = 2 * q95_24h

    # 5. Evaluate Point and Coverage Metrics
    print("\n[Step 5/8] Evaluating 24-hour point prediction and conformal coverage...")
    y_true_test = test_win_df["ref_pm25_24h"].values
    y_pred_test = test_win_df["pred_pm25_24h"].values

    m90_overall = compute_uncertainty_metrics(y_true_test, y_pred_test, test_win_df["lower_90"].values, test_win_df["upper_90"].values, alpha=0.10)
    m95_overall = compute_uncertainty_metrics(y_true_test, y_pred_test, test_win_df["lower_95"].values, test_win_df["upper_95"].values, alpha=0.05)

    print(f"      Pooled 24h Point Performance: RMSE={m90_overall['point_rmse']:.2f}, MAE={m90_overall['point_mae']:.2f}, MBE={m90_overall['point_mbe']:.2f}")
    print(f"      Pooled 24h 90% Coverage:      {m90_overall['empirical_coverage']:.2f}% (Error: {m90_overall['coverage_error']:.2f}%, Winkler: {m90_overall['winkler_score']:.2f})")
    print(f"      Pooled 24h 95% Coverage:      {m95_overall['empirical_coverage']:.2f}% (Error: {m95_overall['coverage_error']:.2f}%, Winkler: {m95_overall['winkler_score']:.2f})")

    # Point metrics table (Overall and by Site)
    point_rows = []
    # Overall
    pt_all = compute_24h_point_metrics(y_true_test, y_pred_test)
    pt_all["subset"] = "Pooled Test"
    point_rows.append(pt_all)

    # By Site
    for site, grp in test_win_df.groupby("site"):
        pt_s = compute_24h_point_metrics(grp["ref_pm25_24h"].values, grp["pred_pm25_24h"].values)
        pt_s["subset"] = f"Site: {site}"
        point_rows.append(pt_s)

    # Classify 24h means into CPCB categories
    test_win_df["ref_category"] = test_win_df["ref_pm25_24h"].apply(classify_pm25)
    test_win_df["pred_category"] = test_win_df["pred_pm25_24h"].apply(classify_pm25)

    # Point metrics by Reference Category
    for cat in CPCB_CATEGORIES:
        grp = test_win_df[test_win_df["ref_category"] == cat]
        if len(grp) > 0:
            pt_c = compute_24h_point_metrics(grp["ref_pm25_24h"].values, grp["pred_pm25_24h"].values)
            pt_c["subset"] = f"Ref Category: {cat}"
            point_rows.append(pt_c)

    df_point_metrics = pd.DataFrame(point_rows)
    df_point_metrics.to_csv(metrics_dir / "checkpoint5_24h_point_metrics.csv", index=False)

    # Coverage metrics table (Overall and by Site and by Category)
    cov_rows = []
    # Overall
    cov_rows.append({
        "subset": "Pooled Test",
        "n_samples": m90_overall["n_samples"],
        "cov_90": m90_overall["empirical_coverage"],
        "cov_error_90": m90_overall["coverage_error"],
        "mean_width_90": m90_overall["mean_interval_width"],
        "winkler_90": m90_overall["winkler_score"],
        "cov_95": m95_overall["empirical_coverage"],
        "cov_error_95": m95_overall["coverage_error"],
        "mean_width_95": m95_overall["mean_interval_width"],
        "winkler_95": m95_overall["winkler_score"],
    })

    # By Site
    site_reliability_rows = []
    for site, grp in test_win_df.groupby("site"):
        m90_s = compute_uncertainty_metrics(grp["ref_pm25_24h"].values, grp["pred_pm25_24h"].values, grp["lower_90"].values, grp["upper_90"].values, alpha=0.10)
        m95_s = compute_uncertainty_metrics(grp["ref_pm25_24h"].values, grp["pred_pm25_24h"].values, grp["lower_95"].values, grp["upper_95"].values, alpha=0.05)
        cov_rows.append({
            "subset": f"Site: {site}",
            "n_samples": m90_s["n_samples"],
            "cov_90": m90_s["empirical_coverage"],
            "cov_error_90": m90_s["coverage_error"],
            "mean_width_90": m90_s["mean_interval_width"],
            "winkler_90": m90_s["winkler_score"],
            "cov_95": m95_s["empirical_coverage"],
            "cov_error_95": m95_s["coverage_error"],
            "mean_width_95": m95_s["mean_interval_width"],
            "winkler_95": m95_s["winkler_score"],
        })
        # For site reliability table
        pt_s = compute_24h_point_metrics(grp["ref_pm25_24h"].values, grp["pred_pm25_24h"].values)
        class_s = compute_cpcb_classification_metrics(grp["ref_category"].tolist(), grp["pred_category"].tolist())
        res_s = grp["pred_pm25_24h"] - grp["ref_pm25_24h"]
        r1_s = float(res_s.autocorr(lag=1)) if len(res_s) > 2 else float("nan")
        site_reliability_rows.append({
            "site": site,
            "n_windows": len(grp),
            "point_rmse": pt_s["rmse"],
            "point_mae": pt_s["mae"],
            "point_mbe": pt_s["mbe"],
            "residual_lag1_autocorr": round(r1_s, 4),
            "category_accuracy": class_s["accuracy"],
            "category_f1_macro": class_s["macro_f1"],
            "cohen_kappa_unweighted": class_s["cohen_kappa_unweighted"],
            "cohen_kappa_linear": class_s["cohen_kappa_linear"],
            "cohen_kappa_quadratic": class_s["cohen_kappa_quadratic"],
            "cov_90": m90_s["empirical_coverage"],
            "cov_error_90": m90_s["coverage_error"],
            "mean_width_90": m90_s["mean_interval_width"],
            "winkler_90": m90_s["winkler_score"],
        })

    # By Reference Category
    for cat in CPCB_CATEGORIES:
        grp = test_win_df[test_win_df["ref_category"] == cat]
        if len(grp) > 0:
            m90_c = compute_uncertainty_metrics(grp["ref_pm25_24h"].values, grp["pred_pm25_24h"].values, grp["lower_90"].values, grp["upper_90"].values, alpha=0.10)
            m95_c = compute_uncertainty_metrics(grp["ref_pm25_24h"].values, grp["pred_pm25_24h"].values, grp["lower_95"].values, grp["upper_95"].values, alpha=0.05)
            cov_rows.append({
                "subset": f"Category: {cat}",
                "n_samples": m90_c["n_samples"],
                "cov_90": m90_c["empirical_coverage"],
                "cov_error_90": m90_c["coverage_error"],
                "mean_width_90": m90_c["mean_interval_width"],
                "winkler_90": m90_c["winkler_score"],
                "cov_95": m95_c["empirical_coverage"],
                "cov_error_95": m95_c["coverage_error"],
                "mean_width_95": m95_c["mean_interval_width"],
                "winkler_95": m95_c["winkler_score"],
            })

    df_cov_metrics = pd.DataFrame(cov_rows)
    df_cov_metrics.to_csv(metrics_dir / "checkpoint5_24h_coverage_metrics.csv", index=False)

    df_site_rel = pd.DataFrame(site_reliability_rows)
    df_site_rel.to_csv(metrics_dir / "checkpoint5_site_reliability.csv", index=False)

    # 6. CPCB Category Classification Metrics
    print("\n[Step 6/8] Computing CPCB category classification and confusion matrix...")
    class_metrics = compute_cpcb_classification_metrics(
        y_true_cat=test_win_df["ref_category"].tolist(),
        y_pred_cat=test_win_df["pred_category"].tolist(),
    )
    print(f"      Overall Category Classification Accuracy: {class_metrics['accuracy']:.2f}%")
    print(f"      Balanced Accuracy:                        {class_metrics['balanced_accuracy']:.2f}%")
    print(f"      Macro F1-Score:                           {class_metrics['macro_f1']:.2f}%")
    print(f"      Weighted F1-Score:                        {class_metrics['weighted_f1']:.2f}%")
    print(f"      Unweighted Cohen's Kappa:                 {class_metrics['cohen_kappa_unweighted']:.4f}")
    print(f"      Linear Weighted Cohen's Kappa:            {class_metrics['cohen_kappa_linear']:.4f}")
    print(f"      Quadratic Weighted Cohen's Kappa:         {class_metrics['cohen_kappa_quadratic']:.4f}")
    print(f"      Exact Matches:                            {class_metrics['pct_exact']:.2f}% ({class_metrics['n_exact']}/{class_metrics['n_samples']})")
    print(f"      Adjacent Category Errors (|delta|=1):     {class_metrics['pct_adjacent_error']:.2f}% ({class_metrics['n_adjacent_error']}/{class_metrics['n_samples']})")
    print(f"      Large Category Errors (|delta|>=2):       {class_metrics['pct_large_error']:.2f}% ({class_metrics['n_large_error']}/{class_metrics['n_samples']})")

    # Export Category Metrics table
    cat_summary_rows = []
    for cname, vals in class_metrics["category_metrics"].items():
        cat_summary_rows.append({
            "cpcb_category": cname,
            "support": vals["support"],
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1_score": vals["f1_score"],
        })
    df_cat_summary = pd.DataFrame(cat_summary_rows)
    df_cat_summary.to_csv(metrics_dir / "checkpoint5_category_metrics.csv", index=False)

    # 7. Uncertainty-Aware Category Stability and Boundary Analysis
    print("\n[Step 7/8] Assessing uncertainty-aware category stability and boundary proximity...")
    test_win_assessed, stab_summary = evaluate_uncertainty_aware_stability(test_win_df)

    print(f"      Stable Windows:                           {stab_summary['pct_stable']:.2f}% ({stab_summary['n_stable']}/{stab_summary['n_total']})")
    print(f"      Ambiguous Windows:                        {stab_summary['pct_ambiguous']:.2f}% ({stab_summary['n_ambiguous']}/{stab_summary['n_total']})")
    print(f"      Point Accuracy in Stable Windows:         {stab_summary['point_accuracy_stable']:.2f}%")
    print(f"      Point Accuracy in Ambiguous Windows:      {stab_summary['point_accuracy_ambiguous']:.2f}%")
    print(f"      True Reference Category Compatibility:    {stab_summary['ref_category_compatibility_pct']:.2f}%")

    # Boundary Neighborhood Analysis
    dist_results = [get_distance_to_nearest_boundary(val) for val in test_win_assessed["ref_pm25_24h"]]
    test_win_assessed["dist_to_boundary"] = [d[0] for d in dist_results]
    test_win_assessed["nearest_boundary"] = [d[1] for d in dist_results]

    boundary_rows = []
    tier_defs = [
        ("Near (<= 5 ug/m3)", 0.0, 5.0),
        ("Intermediate (5 - 10 ug/m3)", 5.0, 10.0),
        ("Far (> 10 ug/m3)", 10.0, float("inf")),
    ]
    for tier_label, low, high in tier_defs:
        sub = test_win_assessed[(test_win_assessed["dist_to_boundary"] >= low) & (test_win_assessed["dist_to_boundary"] < high)]
        n_sub = len(sub)
        if n_sub > 0:
            acc = float(np.mean(sub["is_point_correct"]) * 100.0)
            ambig = float(np.mean(~sub["is_stable"]) * 100.0)
            compat = float(np.mean(sub["is_ref_compatible"]) * 100.0)
            boundary_rows.append({
                "distance_tier": tier_label,
                "n_windows": n_sub,
                "point_accuracy": round(acc, 2),
                "pct_ambiguous": round(ambig, 2),
                "pct_ref_compatible": round(compat, 2),
            })
    df_boundary = pd.DataFrame(boundary_rows)
    df_boundary.to_csv(metrics_dir / "checkpoint5_boundary_analysis.csv", index=False)

    # Save Stability Summary table
    stability_metrics_table = pd.DataFrame([{
        "n_total_windows": stab_summary["n_total"],
        "n_stable_windows": stab_summary["n_stable"],
        "pct_stable": stab_summary["pct_stable"],
        "n_ambiguous_windows": stab_summary["n_ambiguous"],
        "pct_ambiguous": stab_summary["pct_ambiguous"],
        "point_accuracy_overall": stab_summary["point_accuracy_overall"],
        "point_accuracy_stable": stab_summary["point_accuracy_stable"],
        "point_accuracy_ambiguous": stab_summary["point_accuracy_ambiguous"],
        "ref_category_compatibility_pct": stab_summary["ref_category_compatibility_pct"],
        "stability_accuracy_gain": round(stab_summary["point_accuracy_stable"] - stab_summary["point_accuracy_ambiguous"], 2),
    }])
    stability_metrics_table.to_csv(metrics_dir / "checkpoint5_category_stability.csv", index=False)

    # Completeness Sensitivity Comparison Table
    sens_rows = []
    for thresh, cal_df_sens, tst_df_sens in [
        (">=18 Hours (75%)", calib_win_df, test_win_df),
        (">=20 Hours (83.3%)", calib_win_20, test_win_20),
        ("24 Hours (100%)", calib_win_24, test_win_24),
    ]:
        q90_s = compute_conformal_quantile(cal_df_sens["abs_residual_24h"].values, alpha=0.10)
        tst_df_sens["lower_90"] = tst_df_sens["pred_pm25_24h"] - q90_s
        tst_df_sens["upper_90"] = tst_df_sens["pred_pm25_24h"] + q90_s
        cov_s = float(np.mean((tst_df_sens["ref_pm25_24h"] >= tst_df_sens["lower_90"]) & (tst_df_sens["ref_pm25_24h"] <= tst_df_sens["upper_90"])) * 100.0)
        t_ref_cat = tst_df_sens["ref_pm25_24h"].apply(classify_pm25)
        t_pred_cat = tst_df_sens["pred_pm25_24h"].apply(classify_pm25)
        acc_s = float(np.mean(t_ref_cat == t_pred_cat) * 100.0)
        sens_rows.append({
            "completeness_threshold": thresh,
            "n_calib_windows": len(cal_df_sens),
            "n_valid_windows": len(tst_df_sens),
            "calib_q90": round(q90_s, 2),
            "empirical_coverage_90": round(cov_s, 2),
            "category_accuracy": round(acc_s, 2),
        })
    df_sens = pd.DataFrame(sens_rows)

    # Save Prediction and Assessment CSVs
    print("\n[Step 8/8] Saving CSV outputs and generating all 10 figures...")
    # 1. Predictions
    pred_export_df = test_win_assessed[[
        "window_id", "site", "start_time", "end_time", "n_valid_hours", "completeness_pct",
        "ref_pm25_24h", "pred_pm25_24h", "ref_category", "pred_category",
    ]].copy()
    pred_export_df.to_csv(predictions_dir / "checkpoint5_24h_predictions.csv", index=False)

    # 2. Intervals
    intervals_export_df = test_win_assessed[[
        "window_id", "site", "start_time", "end_time", "ref_pm25_24h", "pred_pm25_24h",
        "lower_90", "upper_90", "interval_width_90", "lower_95", "upper_95", "interval_width_95",
    ]].copy()
    intervals_export_df.to_csv(predictions_dir / "checkpoint5_24h_intervals.csv", index=False)

    # 3. Assessment
    test_win_assessed.to_csv(predictions_dir / "checkpoint5_category_assessment.csv", index=False)

    # Generate Figures
    plot_24h_bam_vs_pred_scatter(test_win_assessed, figures_dir / "fig_24h_bam_vs_pred_scatter.png")
    plot_24h_timeseries_intervals(test_win_assessed, figures_dir / "fig_24h_prediction_timeseries_intervals.png")
    plot_cpcb_category_confusion_matrix(class_metrics["confusion_matrix"], CPCB_CATEGORIES, figures_dir / "fig_cpcb_category_confusion_matrix.png")
    plot_category_precision_recall_f1(class_metrics["category_metrics"], figures_dir / "fig_cpcb_category_precision_recall_f1.png")
    plot_category_stability_distribution(stab_summary, figures_dir / "fig_category_stability_distribution.png")
    plot_category_stability_by_site(test_win_assessed, figures_dir / "fig_category_stability_by_site.png")
    plot_reference_pm25_vs_interval_width(test_win_assessed, figures_dir / "fig_reference_pm25_vs_interval_width.png")
    plot_boundary_distance_vs_category_error(df_boundary, figures_dir / "fig_boundary_distance_vs_category_error.png")

    # Dataframe for fig 9: coverage and width by category
    cat_cov_rows = []
    for cat in CPCB_CATEGORIES:
        grp = test_win_assessed[test_win_assessed["ref_category"] == cat]
        if len(grp) > 0:
            cov90 = float(np.mean((grp["ref_pm25_24h"] >= grp["lower_90"]) & (grp["ref_pm25_24h"] <= grp["upper_90"])) * 100.0)
            cat_cov_rows.append({
                "cpcb_category": cat,
                "n_windows": len(grp),
                "empirical_coverage_90": round(cov90, 2),
                "mean_interval_width": round(float(grp["interval_width_90"].mean()), 2),
            })
    df_cat_cov = pd.DataFrame(cat_cov_rows)
    plot_coverage_and_width_by_category(df_cat_cov, figures_dir / "fig_coverage_and_width_by_category.png")
    plot_completeness_sensitivity(df_sens, figures_dir / "fig_completeness_sensitivity_comparison.png")

    # Copy all figures to artifact directory for embedding in conversation artifact
    for fig_file in figures_dir.glob("*.png"):
        shutil.copy2(fig_file, artifact_figs_dir / fig_file.name)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Checkpoint 5 execution finished successfully in {elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
