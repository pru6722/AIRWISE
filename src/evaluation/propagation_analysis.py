"""
Module: src.evaluation.propagation_analysis
Compares error propagation G(g) = RMSE(g) - RMSE(1h) between the conventional
two-stage calibration+imputation pipeline and the proposed Joint TCN across gap durations.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.models.joint_tcn import JointTCN
from src.training.train_joint_tcn import DataNormalizer
from src.evaluation.evaluate_joint_tcn import predict_joint_tcn_full_sequence
from src.evaluation.metrics import compute_regression_metrics
from src.missingness.block_mask import generate_block_outage_mask
from src.models.reconstruction import (
    reconstruct_linear_interpolation,
    reconstruct_random_forest,
)


def run_propagation_gap_analysis(
    model: JointTCN,
    normalizer: DataNormalizer,
    test_df: pd.DataFrame,
    calibration_model: Any,
    calib_feature_cols: List[str],
    gap_hours_list: List[int] = [1, 3, 6, 12, 24],
    missing_rate: float = 0.20,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluates Joint TCN vs Two-Stage Baselines across block outage lengths.
    Computes RMSE, MAE, R^2, and Error Growth G(g) = RMSE(g) - RMSE(1h).
    """
    bam = test_df["bam_reference"].to_numpy(dtype=float)

    # Stage 1: Generate calibrated predictions on test set
    X_calib = test_df[calib_feature_cols].to_numpy(dtype=np.float32)
    calibrated_pm25 = calibration_model.predict(X_calib)
    df_eval = test_df.copy()
    df_eval["calibrated_pm25"] = calibrated_pm25

    impute_covariates = ["relative_humidity", "temperature", "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
    impute_features = [c for c in impute_covariates if c in df_eval.columns]

    rows = []
    baseline_1h_rmse = {}

    for g_hours in gap_hours_list:
        mask = generate_block_outage_mask(
            df_eval,
            gap_hours=g_hours,
            missing_rate=missing_rate,
            segment_col="segment_id",
            random_seed=random_seed,
        )
        hidden_idx = ~mask
        n_hidden = int(np.sum(hidden_idx))

        # 1. Two-Stage: Linear Interpolation
        preds_lin = reconstruct_linear_interpolation(df_eval, "calibrated_pm25", mask)
        # 2. Two-Stage: Random Forest Imputer
        preds_rf = reconstruct_random_forest(df_eval, "calibrated_pm25", impute_features, mask, random_state=random_seed)
        # 3. Proposed: Joint TCN Reconstruction
        _, y_rec_tcn, _ = predict_joint_tcn_full_sequence(model, normalizer, test_df, mask)

        models_eval = [
            ("Two-Stage Linear Interpolation", preds_lin),
            ("Two-Stage Random Forest Imputer", preds_rf),
            ("Joint TCN (Proposed)", y_rec_tcn),
        ]

        for m_name, p_vals in models_eval:
            m = compute_regression_metrics(bam[hidden_idx], p_vals[hidden_idx])

            if g_hours == gap_hours_list[0]:
                baseline_1h_rmse[m_name] = m["rmse"]

            growth = round(m["rmse"] - baseline_1h_rmse.get(m_name, m["rmse"]), 2)

            rows.append({
                "model": m_name,
                "gap_hours": g_hours,
                "missing_rate": missing_rate,
                "hidden_samples": n_hidden,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "mbe": m["mbe"],
                "error_growth_rmse": growth,
            })

    results_df = pd.DataFrame(rows)
    return results_df
