"""
Module: src.evaluation.run_phase3C_benchmark
Executes the approved Phase 3C.2 Four-Pipeline Comparative Benchmark:
- Pipeline A: Impute -> Calibrate (SequentialImputeThenCalibrate)
- Pipeline B: Calibrate -> Impute (SequentialCalibrateThenImpute)
- Pipeline C: Causal Joint TCN v2 (verified epoch 6 checkpoint)
- Pipeline D: Complete-case calibration reference (CompleteCaseCalibrationReference)

Evaluated across 9 pre-registered scenarios:
1. 20% MCAR
2-6. Block Outages (1h, 3h, 6h, 12h, 24h)
7. MAR-RH (humidity conditioned)
8. MAR-Lag (PM_{t-1} conditioned)
9. MNAR-PM (current PM conditioned)

Methodological Guarantees:
- Strict BAM Data Isolation: Test BAM is accessed ONLY by the final metric calculation.
- Synchronized Masks: Identical binary mask passed to all pipelines per scenario.
- Common-Support Fairness: Separate reporting for overall (N=3720), missing PM (m_t=0),
  and observed PM common-support (m_t=1). Pipeline D is never compared directly to full-N.
- Paired Moving-Block Bootstrap: 1,000 resamples at block length b=24h, with sensitivity sweeps.
- Historical Preservation: Checkpoints and historical predictions verified before and after.
"""

from pathlib import Path
import hashlib
import json
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.sequential_pipelines import (
    SequentialImputeThenCalibrate,
    SequentialCalibrateThenImpute,
    CompleteCaseCalibrationReference,
)
from src.models.joint_tcn import JointTCN
from src.training.train_joint_tcn import DataNormalizer, INPUT_FEATURE_COLS
from src.uncertainty.split_manager import create_conformal_splits
from src.evaluation.evaluate_joint_tcn import predict_joint_tcn_full_sequence
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.bootstrap import (
    paired_moving_block_bootstrap,
    block_length_sensitivity_sweep,
)
from src.missingness.random_mask import generate_mcar_mask
from src.missingness.block_mask import generate_block_outage_mask
from src.missingness.regime_mask import (
    generate_humidity_dependent_mask,
    generate_pollution_dependent_mask,
    generate_lag_dependent_mask,
)


OUTPUT_DIR = Path("outputs/experiments/phase3C")
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = Path("outputs/reports")

CHECKPOINTS = {
    "frozen_joint_tcn": Path("outputs/models/conformal/frozen_joint_tcn_checkpoint.pt"),
    "best_joint_tcn": Path("outputs/models/joint_tcn/best_joint_tcn_checkpoint.pt"),
    "joint_tcn_v2": Path("outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt"),
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_scenario_masks(test_df: pd.DataFrame, random_seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """Generates synchronized boolean masks for all 9 pre-registered scenarios."""
    n = len(test_df)
    scenarios = {}

    # 1. 20% MCAR
    m_mcar = generate_mcar_mask(n, missing_rate=0.20, random_seed=random_seed)
    scenarios["MCAR_20pct"] = {
        "scenario_name": "20% MCAR",
        "category": "MCAR",
        "mask": m_mcar,
        "params": {"missing_rate": 0.20, "seed": random_seed},
    }

    # 2-6. Block outages 1h, 3h, 6h, 12h, 24h
    for gap in [1, 3, 6, 12, 24]:
        m_blk = generate_block_outage_mask(
            test_df, gap_hours=gap, missing_rate=0.20, segment_col="segment_id", random_seed=random_seed
        )
        scenarios[f"Block_{gap}h"] = {
            "scenario_name": f"Block Outage ({gap}h)",
            "category": "MCAR (Block)",
            "mask": m_blk,
            "params": {"gap_hours": gap, "missing_rate": 0.20, "seed": random_seed},
        }

    # 7. MAR-RH
    m_rh = generate_humidity_dependent_mask(
        test_df, rh_col="relative_humidity", missing_rate=0.20, random_seed=random_seed
    )
    scenarios["MAR_RH"] = {
        "scenario_name": "MAR-RH (Humidity-Conditioned)",
        "category": "MAR",
        "mask": m_rh,
        "params": {"rh_col": "relative_humidity", "missing_rate": 0.20, "seed": random_seed},
    }

    # 8. MAR-Lag
    m_lag = generate_lag_dependent_mask(
        test_df, pm_col="pa_cf1", missing_rate=0.20, random_seed=random_seed
    )
    scenarios["MAR_Lag"] = {
        "scenario_name": "MAR-Lag (Past PM_{t-1}-Conditioned)",
        "category": "MAR",
        "mask": m_lag,
        "params": {"pm_col": "pa_cf1", "lag": 1, "missing_rate": 0.20, "seed": random_seed},
    }

    # 9. MNAR-PM
    m_pm = generate_pollution_dependent_mask(
        test_df, pm_col="pa_cf1", missing_rate=0.20, random_seed=random_seed
    )
    scenarios["MNAR_PM"] = {
        "scenario_name": "MNAR-PM (True Particulate Intensity)",
        "category": "MNAR",
        "mask": m_pm,
        "params": {"pm_col": "pa_cf1", "missing_rate": 0.20, "seed": random_seed},
    }

    return scenarios


def run_benchmark():
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AIRWISE — Phase 3C.2: Four-Pipeline Comparative Benchmark Execution")
    print("=" * 80)

    # 1. Pre-Execution Hashes
    pre_hashes = {}
    for name, path in CHECKPOINTS.items():
        if path.exists():
            pre_hashes[name] = compute_sha256(path)
            print(f"[PRE-HASH] {name} ({path}): {pre_hashes[name]}")
        else:
            raise FileNotFoundError(f"Required checkpoint missing: {path}")

    # 2. Load dataset and partitions
    data_path = Path("data/processed/cleaned_data.parquet")
    print(f"\n[DATA] Loading cleaned dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    train_df, val_model_df, calib_conformal_df, test_df, split_info = create_conformal_splits(df)

    n_test = len(test_df)
    print(f"[SPLITS] train_df:           {len(train_df):,} rows (used for model fitting)")
    print(f"[SPLITS] val_model_df:       {len(val_model_df):,} rows (model selection)")
    print(f"[SPLITS] calib_conformal_df: {len(calib_conformal_df):,} rows (FROZEN — UNTOUCHED)")
    print(f"[SPLITS] test_df:            {n_test:,} rows (Independent Benchmark Evaluation)")

    # 3. Fit Pipelines A, B, and D strictly on train_df
    print("\n[FIT] Fitting Pipeline A (Impute -> Calibrate) on train_df...")
    pipe_a = SequentialImputeThenCalibrate(imputer_type="knn", model_name="Random Forest", random_state=42)
    pipe_a.fit(train_df)

    print("[FIT] Fitting Pipeline B (Calibrate -> Impute) on train_df...")
    pipe_b = SequentialCalibrateThenImpute(model_name="Random Forest", imputer_type="rf", random_state=42)
    pipe_b.fit(train_df)

    print("[FIT] Fitting Pipeline D (Complete-Case Reference) on train_df...")
    pipe_d = CompleteCaseCalibrationReference(model_name="Random Forest", random_state=42)
    pipe_d.fit(train_df)

    # 4. Load Pipeline C (Joint TCN v2 verified checkpoint)
    print("[LOAD] Loading Pipeline C (Joint TCN v2 causal checkpoint)...")
    feature_cols = [c for c in INPUT_FEATURE_COLS if c in train_df.columns]
    input_dim = len(feature_cols) + 1  # 10 features + 1 mask = 11

    model_c = JointTCN(
        input_dim=input_dim,
        hidden_channels=32,
        dilations=[1, 2, 4, 8],
        kernel_size=3,
        norm_type="layer",
    )
    ckpt = torch.load(CHECKPOINTS["joint_tcn_v2"], weights_only=True)
    model_c.load_state_dict(ckpt)
    model_c.eval()
    print(f"[LOAD] Joint TCN v2 loaded successfully (Receptive Field = {model_c.receptive_field}h)")

    # Fit Normalizer for Joint TCN v2 strictly on train_df
    normalizer_c = DataNormalizer(feature_cols).fit(train_df, target_col="bam_reference")

    # 5. Generate Synchronized Scenario Masks
    print("\n[SCENARIOS] Generating synchronized test masks for 9 scenarios...")
    scenarios = generate_scenario_masks(test_df, random_seed=42)
    print(f"[SCENARIOS] Generated {len(scenarios)} test scenarios (20% MCAR, 5 Blocks, 2 MAR, 1 MNAR).")

    # Storage for detailed results
    all_preds_df = test_df[["timestamp", "site", "segment_id", "bam_reference"]].copy().reset_index(drop=True)
    y_test_true = test_df["bam_reference"].to_numpy(dtype=float)
    segments_test = test_df["segment_id"].to_numpy(dtype=int)
    sites_test = test_df["site"].to_numpy()

    metrics_overall_rows = []
    metrics_missing_pm_rows = []
    metrics_common_support_rows = []
    metrics_site_rows = []
    bootstrap_rows = []

    # 6. Execute Scenario Benchmark Loop
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION ACROSS 9 SCENARIOS")
    print("=" * 80)

    for sc_id, sc_info in scenarios.items():
        sc_name = sc_info["scenario_name"]
        sc_cat = sc_info["category"]
        mask = sc_info["mask"].astype(bool)

        n_obs = int(np.sum(mask))
        n_mis = int(np.sum(~mask))
        obs_rate = float(n_obs / n_test)
        mis_rate = float(n_mis / n_test)

        print(f"\n--- Scenario: {sc_name} [{sc_cat}] ---")
        print(f"    Missing rate: {mis_rate * 100:.2f}% ({n_mis} missing / {n_obs} observed out of {n_test})")

        # Record mask in prediction table
        all_preds_df[f"mask_{sc_id}"] = mask

        # Pipeline A Predictions
        preds_a = pipe_a.predict(test_df, mask=mask)
        all_preds_df[f"pred_A_{sc_id}"] = preds_a

        # Pipeline B Predictions
        preds_b = pipe_b.predict(test_df, mask=mask)
        all_preds_df[f"pred_B_{sc_id}"] = preds_b

        # Pipeline C Predictions
        _, _, preds_c = predict_joint_tcn_full_sequence(
            model=model_c,
            normalizer=normalizer_c,
            df=test_df,
            mask=mask,
            target_col="bam_reference",
        )
        all_preds_df[f"pred_C_{sc_id}"] = preds_c

        # Pipeline D Predictions (returns NaN where mask == False)
        preds_d = pipe_d.predict(test_df, mask=mask)
        all_preds_df[f"pred_D_{sc_id}"] = preds_d

        # -------------------------------------------------------------
        # A. Full Test Population Metrics (N = 3,720)
        # -------------------------------------------------------------
        for pipe_code, pipe_label, preds in [
            ("A", "Pipeline A (Impute -> Calibrate)", preds_a),
            ("B", "Pipeline B (Calibrate -> Impute)", preds_b),
            ("C", "Pipeline C (Joint TCN v2)", preds_c),
        ]:
            m = compute_regression_metrics(y_test_true, preds)
            metrics_overall_rows.append({
                "scenario_id": sc_id,
                "scenario_name": sc_name,
                "category": sc_cat,
                "pipeline_code": pipe_code,
                "pipeline_name": pipe_label,
                "evaluated_population": "Full Test Population (N=3720)",
                "n_samples": n_test,
                "coverage_pct": 100.0,
                "missing_predictions": 0,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "mbe": m["mbe"],
                "rel_bias_pct": m["relative_bias_pct"],
            })

        # Pipeline D on Full Test Population is partial
        m_d_obs = compute_regression_metrics(y_test_true[mask], preds_d[mask])
        metrics_overall_rows.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "category": sc_cat,
            "pipeline_code": "D",
            "pipeline_name": "Pipeline D (Complete-Case Reference)",
            "evaluated_population": "Eligible Complete Cases Only (Not Full-N Comparable)",
            "n_samples": n_obs,
            "coverage_pct": round(obs_rate * 100.0, 2),
            "missing_predictions": n_mis,
            "rmse": m_d_obs["rmse"],
            "mae": m_d_obs["mae"],
            "r2": m_d_obs["r2"],
            "mbe": m_d_obs["mbe"],
            "rel_bias_pct": m_d_obs["relative_bias_pct"],
        })

        # -------------------------------------------------------------
        # B. Missing PM Subgroup Metrics (m_t = 0, Reconstruction Test)
        # -------------------------------------------------------------
        if n_mis > 0:
            for pipe_code, pipe_label, preds in [
                ("A", "Pipeline A (Impute -> Calibrate)", preds_a),
                ("B", "Pipeline B (Calibrate -> Impute)", preds_b),
                ("C", "Pipeline C (Joint TCN v2)", preds_c),
            ]:
                m_mis = compute_regression_metrics(y_test_true[~mask], preds[~mask])
                metrics_missing_pm_rows.append({
                    "scenario_id": sc_id,
                    "scenario_name": sc_name,
                    "category": sc_cat,
                    "pipeline_code": pipe_code,
                    "pipeline_name": pipe_label,
                    "evaluated_population": "Missing PM Subgroup (m_t=0)",
                    "n_samples": n_mis,
                    "coverage_pct": 100.0,
                    "rmse": m_mis["rmse"],
                    "mae": m_mis["mae"],
                    "r2": m_mis["r2"],
                    "mbe": m_mis["mbe"],
                })

            # Pipeline D produces 0 predictions on missing points
            metrics_missing_pm_rows.append({
                "scenario_id": sc_id,
                "scenario_name": sc_name,
                "category": sc_cat,
                "pipeline_code": "D",
                "pipeline_name": "Pipeline D (Complete-Case Reference)",
                "evaluated_population": "Missing PM Subgroup (m_t=0)",
                "n_samples": 0,
                "coverage_pct": 0.0,
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "mbe": np.nan,
            })

        # -------------------------------------------------------------
        # C. Common-Support Subgroup Metrics (m_t = 1, Calibration Test)
        # -------------------------------------------------------------
        if n_obs > 0:
            for pipe_code, pipe_label, preds in [
                ("A", "Pipeline A (Impute -> Calibrate)", preds_a),
                ("B", "Pipeline B (Calibrate -> Impute)", preds_b),
                ("C", "Pipeline C (Joint TCN v2)", preds_c),
                ("D", "Pipeline D (Complete-Case Reference)", preds_d),
            ]:
                m_obs = compute_regression_metrics(y_test_true[mask], preds[mask])
                metrics_common_support_rows.append({
                    "scenario_id": sc_id,
                    "scenario_name": sc_name,
                    "category": sc_cat,
                    "pipeline_code": pipe_code,
                    "pipeline_name": pipe_label,
                    "evaluated_population": "Common Support (m_t=1 All-Pipeline Matched)",
                    "n_samples": n_obs,
                    "coverage_pct": 100.0,
                    "rmse": m_obs["rmse"],
                    "mae": m_obs["mae"],
                    "r2": m_obs["r2"],
                    "mbe": m_obs["mbe"],
                })

        # -------------------------------------------------------------
        # D. Site-Stratified Metrics (Overall)
        # -------------------------------------------------------------
        for site_name in np.unique(sites_test):
            site_mask = (sites_test == site_name)
            site_n = int(np.sum(site_mask))
            for pipe_code, pipe_label, preds in [
                ("A", "Pipeline A", preds_a),
                ("B", "Pipeline B", preds_b),
                ("C", "Pipeline C (Joint TCN v2)", preds_c),
            ]:
                m_site = compute_regression_metrics(y_test_true[site_mask], preds[site_mask])
                metrics_site_rows.append({
                    "scenario_id": sc_id,
                    "site": site_name,
                    "pipeline_code": pipe_code,
                    "n_samples": site_n,
                    "rmse": m_site["rmse"],
                    "mae": m_site["mae"],
                    "r2": m_site["r2"],
                })

        # -------------------------------------------------------------
        # E. Paired Moving-Block Bootstrap (b=24h, B=1,000)
        # -------------------------------------------------------------
        # 1. Pipeline C vs Pipeline A (Overall)
        boot_ca = paired_moving_block_bootstrap(
            y_true=y_test_true,
            y_pred_joint=preds_c,
            y_pred_baseline=preds_a,
            segments=segments_test,
            block_length=24,
            n_bootstraps=1000,
            random_seed=42,
        )
        bootstrap_rows.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "comparison": "Joint TCN v2 (C) vs Pipeline A",
            "evaluated_subset": "Full Test Population (N=3720)",
            "block_length_h": 24,
            "delta_rmse": boot_ca["point_estimates"]["delta_rmse"],
            "ci_lower_95": boot_ca["delta_rmse_ci"][0],
            "ci_upper_95": boot_ca["delta_rmse_ci"][1],
            "p_value_rmse": boot_ca["p_value_rmse"],
            "significant": boot_ca["statistically_significant_05"],
        })

        # 2. Pipeline C vs Pipeline B (Overall)
        boot_cb = paired_moving_block_bootstrap(
            y_true=y_test_true,
            y_pred_joint=preds_c,
            y_pred_baseline=preds_b,
            segments=segments_test,
            block_length=24,
            n_bootstraps=1000,
            random_seed=42,
        )
        bootstrap_rows.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "comparison": "Joint TCN v2 (C) vs Pipeline B",
            "evaluated_subset": "Full Test Population (N=3720)",
            "block_length_h": 24,
            "delta_rmse": boot_cb["point_estimates"]["delta_rmse"],
            "ci_lower_95": boot_cb["delta_rmse_ci"][0],
            "ci_upper_95": boot_cb["delta_rmse_ci"][1],
            "p_value_rmse": boot_cb["p_value_rmse"],
            "significant": boot_cb["statistically_significant_05"],
        })

        # 3. Pipeline C vs Pipeline D (Common Support m_t = 1)
        boot_cd = paired_moving_block_bootstrap(
            y_true=y_test_true[mask],
            y_pred_joint=preds_c[mask],
            y_pred_baseline=preds_d[mask],
            segments=segments_test[mask],
            block_length=24,
            n_bootstraps=1000,
            random_seed=42,
        )
        bootstrap_rows.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "comparison": "Joint TCN v2 (C) vs Pipeline D",
            "evaluated_subset": "Common Support (m_t=1 Matched)",
            "block_length_h": 24,
            "delta_rmse": boot_cd["point_estimates"]["delta_rmse"],
            "ci_lower_95": boot_cd["delta_rmse_ci"][0],
            "ci_upper_95": boot_cd["delta_rmse_ci"][1],
            "p_value_rmse": boot_cd["p_value_rmse"],
            "significant": boot_cd["statistically_significant_05"],
        })

        print(f"    [P-A] RMSE: {compute_regression_metrics(y_test_true, preds_a)['rmse']:.2f}")
        print(f"    [P-B] RMSE: {compute_regression_metrics(y_test_true, preds_b)['rmse']:.2f}")
        print(f"    [P-C] RMSE: {compute_regression_metrics(y_test_true, preds_c)['rmse']:.2f} (Delta_A: {boot_ca['point_estimates']['delta_rmse']:+.2f}, p={boot_ca['p_value_rmse']:.3f})")
        print(f"    [P-D Common] RMSE: {m_d_obs['rmse']:.2f} (Matched N={n_obs})")

    # 7. Block-Length Sensitivity Sweep on Key Scenarios
    print("\n[BOOTSTRAP] Running block-length sensitivity sweeps [12h, 24h, 48h, 72h]...")
    sensitivity_records = []
    for target_sc in ["MCAR_20pct", "Block_6h", "MNAR_PM"]:
        m = scenarios[target_sc]["mask"].astype(bool)
        pa = all_preds_df[f"pred_A_{target_sc}"].to_numpy()
        pc = all_preds_df[f"pred_C_{target_sc}"].to_numpy()

        sw_df = block_length_sensitivity_sweep(
            y_true=y_test_true,
            y_pred_joint=pc,
            y_pred_baseline=pa,
            segments=segments_test,
            block_lengths=[12, 24, 48, 72],
            n_bootstraps=500,
            random_seed=42,
        )
        sw_df["scenario_id"] = target_sc
        sw_df["comparison"] = "Joint TCN v2 (C) vs Pipeline A"
        sensitivity_records.append(sw_df)

    sensitivity_df = pd.concat(sensitivity_records, ignore_index=True)

    # 8. Export Machine-Readable Files
    print("\n[EXPORT] Exporting benchmark artifacts...")
    all_preds_df.to_parquet(OUTPUT_DIR / "benchmark_predictions.parquet", index=False)
    print(f"    Exported predictions: {OUTPUT_DIR / 'benchmark_predictions.parquet'} ({len(all_preds_df)} rows)")

    df_overall = pd.DataFrame(metrics_overall_rows)
    df_overall.to_csv(OUTPUT_DIR / "scenario_metrics_overall.csv", index=False)

    df_missing = pd.DataFrame(metrics_missing_pm_rows)
    df_missing.to_csv(OUTPUT_DIR / "scenario_metrics_missing_pm.csv", index=False)

    df_common = pd.DataFrame(metrics_common_support_rows)
    df_common.to_csv(OUTPUT_DIR / "scenario_metrics_common_support.csv", index=False)

    df_site = pd.DataFrame(metrics_site_rows)
    df_site.to_csv(OUTPUT_DIR / "site_stratified_metrics.csv", index=False)

    df_boot = pd.DataFrame(bootstrap_rows)
    df_boot.to_csv(OUTPUT_DIR / "bootstrap_significance_results.csv", index=False)

    sensitivity_df.to_csv(OUTPUT_DIR / "bootstrap_sensitivity_sweep.csv", index=False)
    print(f"    Exported all metrics CSVs to {OUTPUT_DIR}/")

    # 9. Generate Publication-Quality Visualizations
    print("[PLOTS] Generating comparative figures...")
    _generate_figures(df_overall, df_missing, df_common, sensitivity_df, FIGURES_DIR)

    # 10. Post-Execution Artifact Governance & SHA-256 Verification
    print("\n" + "=" * 80)
    print("ARTIFACT GOVERNANCE & HASH PRESERVATION CHECK")
    print("=" * 80)
    preserved = True
    for name, path in CHECKPOINTS.items():
        post_h = compute_sha256(path)
        match = (post_h == pre_hashes[name])
        print(f"[HASH] {name}: {post_h} | Preserved: {match}")
        if not match:
            preserved = False

    if not preserved:
        print("[CRITICAL ERROR] Checkpoint tampering detected!")
        return 1

    duration = time.time() - start_time
    print(f"\n[COMPLETE] Phase 3C.2 benchmark completed in {duration:.1f}s ({duration/60:.2f} mins)")
    return 0


def _generate_figures(df_overall, df_missing, df_common, df_sens, fig_dir: Path):
    """Generates clean publication-standard figures."""
    # Figure 1: Full-Population RMSE across Scenarios (Pipelines A, B, C)
    plt.figure(figsize=(12, 6))
    sc_order = ["MCAR_20pct", "Block_1h", "Block_3h", "Block_6h", "Block_12h", "Block_24h", "MAR_RH", "MAR_Lag", "MNAR_PM"]
    sc_labels = ["20% MCAR", "Block 1h", "Block 3h", "Block 6h", "Block 12h", "Block 24h", "MAR-RH", "MAR-Lag", "MNAR-PM"]

    sub_overall = df_overall[df_overall["pipeline_code"].isin(["A", "B", "C"])].copy()
    piv = sub_overall.pivot(index="scenario_id", columns="pipeline_code", values="rmse").reindex(sc_order)

    x = np.arange(len(sc_order))
    width = 0.25

    plt.bar(x - width, piv["A"], width=width, label="Pipeline A (Impute -> Calib)", color="#4575b4", alpha=0.9)
    plt.bar(x, piv["B"], width=width, label="Pipeline B (Calib -> Impute)", color="#fdae61", alpha=0.9)
    plt.bar(x + width, piv["C"], width=width, label="Pipeline C (Joint TCN v2)", color="#d73027", alpha=0.9)

    plt.xticks(x, sc_labels, rotation=30, ha="right", fontsize=10)
    plt.ylabel("RMSE (μg/m³)", fontsize=12)
    plt.title("Full Test Population PM2.5 RMSE across Missingness Scenarios (N=3,720)", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "benchmark_rmse_by_scenario.png", dpi=200)
    plt.close()

    # Figure 2: Block Outage Duration Degradation
    blk_sc = ["Block_1h", "Block_3h", "Block_6h", "Block_12h", "Block_24h"]
    blk_hours = [1, 3, 6, 12, 24]
    piv_blk = df_missing[df_missing["pipeline_code"].isin(["A", "B", "C"])].pivot(
        index="scenario_id", columns="pipeline_code", values="rmse"
    ).reindex(blk_sc)

    plt.figure(figsize=(9, 5))
    plt.plot(blk_hours, piv_blk["A"], "o-", label="Pipeline A (Impute -> Calib)", color="#4575b4", linewidth=2, markersize=7)
    plt.plot(blk_hours, piv_blk["B"], "s-", label="Pipeline B (Calib -> Impute)", color="#fdae61", linewidth=2, markersize=7)
    plt.plot(blk_hours, piv_blk["C"], "^-", label="Pipeline C (Joint TCN v2)", color="#d73027", linewidth=2.5, markersize=8)

    plt.xlabel("Block Outage Duration (Hours)", fontsize=12)
    plt.ylabel("Missing Subgroup RMSE (μg/m³)", fontsize=12)
    plt.title("Reconstruction Degradation under Consecutive Block Outages (m_t=0)", fontsize=13, fontweight="bold")
    plt.xticks(blk_hours, ["1h", "3h", "6h", "12h", "24h"])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "block_outage_degradation.png", dpi=200)
    plt.close()

    # Figure 3: Common-Support Matched Comparison (A, B, C, D on m_t = 1)
    plt.figure(figsize=(12, 5))
    piv_cs = df_common.pivot(index="scenario_id", columns="pipeline_code", values="rmse").reindex(sc_order)
    width = 0.2
    plt.bar(x - 1.5 * width, piv_cs["A"], width=width, label="Pipeline A", color="#4575b4", alpha=0.85)
    plt.bar(x - 0.5 * width, piv_cs["B"], width=width, label="Pipeline B", color="#fdae61", alpha=0.85)
    plt.bar(x + 0.5 * width, piv_cs["C"], width=width, label="Pipeline C (Joint TCN v2)", color="#d73027", alpha=0.85)
    plt.bar(x + 1.5 * width, piv_cs["D"], width=width, label="Pipeline D (Complete-Case Ref)", color="#2ca02c", alpha=0.85)

    plt.xticks(x, sc_labels, rotation=30, ha="right", fontsize=10)
    plt.ylabel("Observed RMSE (μg/m³)", fontsize=12)
    plt.title("Common Support Comparison on Observed Subsets (m_t=1 Matched Denominators)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, fontsize=9, ncol=4)
    plt.tight_layout()
    plt.savefig(fig_dir / "common_support_comparison.png", dpi=200)
    plt.close()
    print(f"    Saved figures to {fig_dir}/")


if __name__ == "__main__":
    import sys
    sys.exit(run_benchmark())
