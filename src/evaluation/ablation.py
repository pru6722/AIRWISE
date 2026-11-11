"""
Module: src.evaluation.ablation
Conducts systematic ablation study of the Joint TCN architecture:
A. Full Joint TCN (Multi-channel, Meteorology, Observation Masks, Joint Loss)
B. Without meteorological inputs (No RH, No Temperature)
C. Without multi-channel inputs (Single PA_ATM channel only)
D. Without observation masks (Missing values zeroed without mask indicator)
E. Calibration-only TCN (lambda_cal=1, lambda_rec=0)
F. Reconstruction-only TCN (lambda_cal=0, lambda_rec=1)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch

from src.training.train_joint_tcn import train_joint_tcn_model, INPUT_FEATURE_COLS
from src.evaluation.evaluate_joint_tcn import predict_joint_tcn_full_sequence
from src.evaluation.metrics import compute_regression_metrics


def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_mask: np.ndarray,
    baseline_joint_model: Optional[Tuple[Any, Any]] = None,
    random_seed: int = 42,
    epochs: int = 20,
) -> pd.DataFrame:
    """
    Executes all 6 required ablation configurations under identical data splits and benchmark masks.
    Returns:
        ablation_df: Comparison metrics across configurations.
    """
    bam = test_df["bam_reference"].to_numpy(dtype=float)
    hidden_mask = ~test_mask
    observed_mask = test_mask.astype(bool)

    ablation_configs = [
        {
            "name": "A. Full Joint TCN",
            "features": INPUT_FEATURE_COLS,
            "lambda_cal": 1.0,
            "lambda_rec": 1.0,
            "desc": "All 3 PM2.5 channels + meteorology + temporal cyclical + observation mask + joint multi-task loss",
        },
        {
            "name": "B. Without Meteorological Inputs",
            "features": [c for c in INPUT_FEATURE_COLS if c not in ["relative_humidity", "temperature"]],
            "lambda_cal": 1.0,
            "lambda_rec": 1.0,
            "desc": "Ablates ambient RH and temperature covariates",
        },
        {
            "name": "C. Without Multi-Channel Inputs",
            "features": [c for c in INPUT_FEATURE_COLS if c not in ["pa_cf1", "pa_alt"]],
            "lambda_cal": 1.0,
            "lambda_rec": 1.0,
            "desc": "Single PA_ATM channel only (removes PA_CF1 and PA_ALT)",
        },
        {
            "name": "D. Without Observation Masks",
            "features": INPUT_FEATURE_COLS,
            "lambda_cal": 1.0,
            "lambda_rec": 1.0,
            "desc": "Removes observation mask indicator (zeros imputed blindly)",
            "no_mask": True,
        },
        {
            "name": "E. Calibration-Only TCN",
            "features": INPUT_FEATURE_COLS,
            "lambda_cal": 1.0,
            "lambda_rec": 0.0,
            "desc": "TCN trained strictly on calibration loss (no reconstruction objective)",
        },
        {
            "name": "F. Reconstruction-Only TCN",
            "features": INPUT_FEATURE_COLS,
            "lambda_cal": 0.0,
            "lambda_rec": 1.0,
            "desc": "TCN trained strictly on reconstruction loss (no calibration objective)",
        },
    ]

    rows = []

    for cfg in ablation_configs:
        print(f"    Running ablation: {cfg['name']}...")
        fcols = cfg["features"]
        l_cal = cfg["lambda_cal"]
        l_rec = cfg["lambda_rec"]
        use_mask = not cfg.get("no_mask", False)

        # Train model or reuse baseline for Configuration A
        if cfg["name"].startswith("A.") and baseline_joint_model is not None:
            model, normalizer = baseline_joint_model
        else:
            model, normalizer, _ = train_joint_tcn_model(
                train_df=train_df,
                val_df=val_df,
                feature_cols=fcols,
                lambda_cal=l_cal,
                lambda_rec=l_rec,
                use_mask_feature=use_mask,
                epochs=epochs,
                random_seed=random_seed,
            )

        # Inference on test set
        y_cal, y_rec, y_fin = predict_joint_tcn_full_sequence(
            model, normalizer, test_df, test_mask, use_mask_feature=use_mask
        )

        # Calibration metrics on observed points
        m_cal = compute_regression_metrics(bam[observed_mask], y_cal[observed_mask])
        # Reconstruction metrics on hidden points
        m_rec = compute_regression_metrics(bam[hidden_mask], y_rec[hidden_mask])
        # Composite metrics on full test set
        m_all = compute_regression_metrics(bam, y_fin)

        rows.append({
            "ablation_configuration": cfg["name"],
            "description": cfg["desc"],
            "lambda_cal": l_cal,
            "lambda_rec": l_rec,
            "cal_observed_rmse": m_cal["rmse"],
            "cal_observed_mae": m_cal["mae"],
            "cal_observed_r2": m_cal["r2"],
            "rec_hidden_rmse": m_rec["rmse"],
            "rec_hidden_mae": m_rec["mae"],
            "rec_hidden_r2": m_rec["r2"],
            "rec_hidden_mbe": m_rec["mbe"],
            "full_test_rmse": m_all["rmse"],
            "full_test_r2": m_all["r2"],
        })

    ablation_df = pd.DataFrame(rows)
    return ablation_df
