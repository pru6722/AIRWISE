"""
Module: src.evaluation.propagation
Conducts error propagation analysis across consecutive missing-gap lengths (1, 3, 6, 12, 24 hours),
missingness rates (5%, 10%, 20%, 30%), environmental humidity regimes, and study sites.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_regression_metrics
from src.missingness.block_mask import generate_block_outage_mask
from src.missingness.random_mask import generate_mcar_mask
from src.missingness.regime_mask import generate_humidity_dependent_mask, generate_pollution_dependent_mask
from src.models.reconstruction import (
    reconstruct_linear_interpolation,
    reconstruct_knn,
    reconstruct_random_forest,
)


def evaluate_propagation_across_gaps(
    test_df: pd.DataFrame,
    calibration_model: Any,
    calib_feature_cols: List[str],
    covariate_cols: List[str],
    gap_hours_list: List[int] = [1, 3, 6, 12, 24],
    missing_rate: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates the two-stage pipeline under varying gap lengths.
    Computes RMSE, MAE, and Error Growth G(g) = RMSE(g) - RMSE(g_0) where g_0 = 1 hour.
    """
    # 1. Generate calibrated estimates on test set
    X_calib = test_df[calib_feature_cols].to_numpy(dtype=np.float32)
    calibrated_pm25 = calibration_model.predict(X_calib)

    df_eval = test_df.copy()
    df_eval["calibrated_pm25"] = calibrated_pm25
    bam_target = df_eval["bam_reference"].to_numpy(dtype=float)

    impute_features = [c for c in covariate_cols if c in df_eval.columns]

    results_rows = []
    baseline_rmse_1h: Dict[str, float] = {}

    for g_hours in gap_hours_list:
        mask = generate_block_outage_mask(
            df_eval,
            gap_hours=g_hours,
            missing_rate=missing_rate,
            segment_col="segment_id",
            random_seed=random_state,
        )
        hidden_idx = ~mask

        # Run fast representative baselines: Linear Interpolation and Random Forest
        preds_lin = reconstruct_linear_interpolation(df_eval, "calibrated_pm25", mask)
        preds_rf = reconstruct_random_forest(df_eval, "calibrated_pm25", impute_features, mask, random_state=random_state)

        for model_name, preds in [("Linear Interpolation", preds_lin), ("Random Forest Imputer", preds_rf)]:
            m = compute_regression_metrics(bam_target[hidden_idx], preds[hidden_idx])

            if g_hours == gap_hours_list[0]:
                baseline_rmse_1h[model_name] = m["rmse"]

            growth = round(m["rmse"] - baseline_rmse_1h.get(model_name, m["rmse"]), 2)

            results_rows.append({
                "model": model_name,
                "gap_hours": g_hours,
                "missing_rate": missing_rate,
                "hidden_count": int(np.sum(hidden_idx)),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "mbe": m["mbe"],
                "error_growth_rmse": growth,
            })

    results_df = pd.DataFrame(results_rows)
    return results_df, {"calibrated_pm25": calibrated_pm25}


def evaluate_propagation_by_regimes_and_mechanisms(
    test_df: pd.DataFrame,
    calibration_model: Any,
    calib_feature_cols: List[str],
    covariate_cols: List[str],
    missing_rate: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates reconstruction performance stratified by:
    1. Environmental humidity regimes (Low <40%, Med 40-65%, High >65%)
    2. Missingness mechanisms (MCAR, Consecutive 6h Block, MAR Humidity, MAR Pollution)
    """
    X_calib = test_df[calib_feature_cols].to_numpy(dtype=np.float32)
    df_eval = test_df.copy()
    df_eval["calibrated_pm25"] = calibration_model.predict(X_calib)
    bam = df_eval["bam_reference"].to_numpy(dtype=float)
    rh = df_eval["relative_humidity"].to_numpy(dtype=float)

    impute_features = [c for c in covariate_cols if c in df_eval.columns]

    # Generate masks for 4 mechanisms
    masks = {
        "MCAR Random": generate_mcar_mask(len(df_eval), missing_rate=missing_rate, random_seed=random_state),
        "Consecutive 6h Blocks": generate_block_outage_mask(df_eval, gap_hours=6, missing_rate=missing_rate, random_seed=random_state),
        "MAR Humidity-Conditioned": generate_humidity_dependent_mask(df_eval, missing_rate=missing_rate, random_seed=random_state),
        "MAR Pollution-Conditioned": generate_pollution_dependent_mask(df_eval, missing_rate=missing_rate, random_seed=random_state),
    }

    # 1. Mechanism comparison
    mech_rows = []
    for mech_name, msk in masks.items():
        hidden = ~msk
        recon = reconstruct_linear_interpolation(df_eval, "calibrated_pm25", msk)
        met = compute_regression_metrics(bam[hidden], recon[hidden])
        mech_rows.append({
            "mechanism": mech_name,
            "missing_rate": missing_rate,
            "hidden_samples": int(np.sum(hidden)),
            "mae": met["mae"],
            "rmse": met["rmse"],
            "r2": met["r2"],
            "mbe": met["mbe"],
        })
    df_mech = pd.DataFrame(mech_rows)

    # 2. Humidity regime comparison under 6h blocks
    block_mask = masks["Consecutive 6h Blocks"]
    recon_block = reconstruct_linear_interpolation(df_eval, "calibrated_pm25", block_mask)
    hidden_block = ~block_mask

    humidity_strata = {
        "Low Humidity (RH < 40%)": hidden_block & (rh < 40.0),
        "Medium Humidity (40% <= RH <= 65%)": hidden_block & (rh >= 40.0) & (rh <= 65.0),
        "High Humidity (RH > 65%)": hidden_block & (rh > 65.0),
    }

    hum_rows = []
    for regime_name, r_idx in humidity_strata.items():
        if np.sum(r_idx) > 0:
            met = compute_regression_metrics(bam[r_idx], recon_block[r_idx])
            hum_rows.append({
                "humidity_regime": regime_name,
                "hidden_samples": int(np.sum(r_idx)),
                "mae": met["mae"],
                "rmse": met["rmse"],
                "r2": met["r2"],
                "mbe": met["mbe"],
                "rel_bias_pct": met["relative_bias_pct"],
            })
    df_hum = pd.DataFrame(hum_rows)

    return df_mech, df_hum
