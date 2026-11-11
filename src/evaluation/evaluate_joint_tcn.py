"""
Module: src.evaluation.evaluate_joint_tcn
Performs full causal sequence inference with the trained Joint TCN,
computes stratified metrics (site, pollution band, humidity regime, gap length),
calculates block-bootstrap confidence intervals, and formats predictions.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch

from src.models.joint_tcn import JointTCN
from src.training.train_joint_tcn import DataNormalizer, INPUT_FEATURE_COLS
from src.evaluation.metrics import compute_regression_metrics


def predict_joint_tcn_full_sequence(
    model: JointTCN,
    normalizer: DataNormalizer,
    df: pd.DataFrame,
    mask: np.ndarray,
    target_col: str = "bam_reference",
    use_mask_feature: bool = True,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes causal inference across segments for an entire DataFrame.
    Guarantees that predictions are generated for every single observation
    without crossing segment boundaries or leaking future information.

    Args:
        model: Trained JointTCN model instance
        normalizer: Fitted DataNormalizer
        df: Input DataFrame with features, segment_id, and site
        mask: 1D boolean array (True = observed, False = hidden)
        target_col: Target column name
        device: 'cpu' or 'cuda'

    Returns:
        y_cal_real: Calibrated predictions on original BAM scale (N,)
        y_rec_real: Reconstruction predictions on original BAM scale (N,)
        y_final_real: Composite predictions (N,)
    """
    model.eval()
    df = df.reset_index(drop=True)

    # Transform features using fitted normalizer
    feature_cols = [c for c in INPUT_FEATURE_COLS if c in df.columns]
    X_norm, _ = normalizer.transform(df, target_col=target_col)
    m = mask.astype(np.float32)

    # Low-cost PM2.5 channels (first 3 features) zeroed where mask == 0
    X_masked = X_norm.copy()
    X_masked[m == 0, :3] = 0.0

    # Append mask column -> (N, D+1)
    if use_mask_feature:
        input_features = np.column_stack([X_masked, m])
    else:
        input_features = np.column_stack([X_masked, np.ones_like(m)])

    n_samples = len(df)
    y_cal_accum = np.zeros(n_samples, dtype=np.float32)
    y_rec_accum = np.zeros(n_samples, dtype=np.float32)
    y_fin_accum = np.zeros(n_samples, dtype=np.float32)

    # Process segment by segment to respect segment boundaries
    with torch.no_grad():
        for _, seg_df in df.groupby("segment_id", sort=False):
            idx = seg_df.index.to_numpy()
            seg_feat = input_features[idx]  # (L_seg, Channels)
            seg_mask = m[idx]              # (L_seg,)

            # Reshape to (1, Channels, Length) for CausalConv1d
            x_tensor = torch.tensor(seg_feat.T, dtype=torch.float32).unsqueeze(0).to(device)
            m_tensor = torch.tensor(seg_mask, dtype=torch.float32).unsqueeze(0).to(device)

            cal_norm, rec_norm, fin_norm = model(x_tensor, m_tensor)

            y_cal_accum[idx] = cal_norm.squeeze(0).cpu().numpy()
            y_rec_accum[idx] = rec_norm.squeeze(0).cpu().numpy()
            y_fin_accum[idx] = fin_norm.squeeze(0).cpu().numpy()

    # Inverse transform target predictions from normalized scale to original ug/m3
    y_cal_real = normalizer.inverse_transform_target(y_cal_accum)
    y_rec_real = normalizer.inverse_transform_target(y_rec_accum)
    y_final_real = normalizer.inverse_transform_target(y_fin_accum)

    return y_cal_real, y_rec_real, y_final_real


def compute_block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    block_size: int = 24,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes block-bootstrap confidence intervals for the paired difference in RMSE:
    Delta RMSE = RMSE(Model 1) - RMSE(Model 2).
    Preserves temporal autocorrelation by resampling blocks of contiguous hours.
    """
    n = len(y_true)
    if n < block_size * 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_seed)
    n_blocks = n // block_size
    diffs = []

    for _ in range(n_bootstraps):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in block_starts])

        yt_b = y_true[indices]
        yp1_b = y_pred1[indices]
        yp2_b = y_pred2[indices]

        rmse1 = np.sqrt(np.mean((yp1_b - yt_b) ** 2))
        rmse2 = np.sqrt(np.mean((yp2_b - yt_b) ** 2))
        diffs.append(rmse1 - rmse2)

    diffs = np.array(diffs)
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(diffs, alpha * 100))
    median = float(np.median(diffs))
    upper = float(np.percentile(diffs, (1.0 - alpha) * 100))

    return round(lower, 2), round(median, 2), round(upper, 2)


def evaluate_joint_tcn_comprehensively(
    model: JointTCN,
    normalizer: DataNormalizer,
    test_df: pd.DataFrame,
    mask: np.ndarray,
    baseline_predictions: Dict[str, np.ndarray],
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes full multi-dimensional evaluation of Joint TCN against baselines:
    - Overall test metrics
    - Site-wise metrics
    - Pollution-concentration bands
    - Environmental humidity regimes
    - Paired comparisons with block-bootstrap CI
    """
    y_cal, y_rec, y_final = predict_joint_tcn_full_sequence(
        model, normalizer, test_df, mask
    )

    bam = test_df["bam_reference"].to_numpy(dtype=float)
    hidden_mask = ~mask
    observed_mask = mask.astype(bool)

    # 1. Overall Head-to-Head Comparison
    overall_rows = []

    # Baselines on hidden points
    for b_name, b_preds in baseline_predictions.items():
        m_hid = compute_regression_metrics(bam[hidden_mask], b_preds[hidden_mask])
        m_all = compute_regression_metrics(bam, b_preds)
        ci_low, ci_med, ci_high = compute_block_bootstrap_ci(
            bam[hidden_mask], y_rec[hidden_mask], b_preds[hidden_mask], random_seed=random_seed
        )
        overall_rows.append({
            "model": b_name,
            "hidden_rmse": m_hid["rmse"],
            "hidden_mae": m_hid["mae"],
            "hidden_r2": m_hid["r2"],
            "hidden_mbe": m_hid["mbe"],
            "delta_rmse_vs_joint_ci_95": f"[{ci_low:+.2f}, {ci_high:+.2f}]" if not np.isnan(ci_low) else "N/A",
            "full_test_rmse": m_all["rmse"],
            "full_test_r2": m_all["r2"],
        })

    # Joint TCN Reconstruction on hidden
    m_tcn_hid = compute_regression_metrics(bam[hidden_mask], y_rec[hidden_mask])
    m_tcn_all = compute_regression_metrics(bam, y_final)
    overall_rows.append({
        "model": "Joint TCN (Proposed)",
        "hidden_rmse": m_tcn_hid["rmse"],
        "hidden_mae": m_tcn_hid["mae"],
        "hidden_r2": m_tcn_hid["r2"],
        "hidden_mbe": m_tcn_hid["mbe"],
        "delta_rmse_vs_joint_ci_95": "Reference",
        "full_test_rmse": m_tcn_all["rmse"],
        "full_test_r2": m_tcn_all["r2"],
    })
    df_overall = pd.DataFrame(overall_rows)

    # 2. Site-wise Breakdown
    site_rows = []
    site_arr = test_df["site"].to_numpy()
    for site_name in ["Delhi", "Hamirpur", "Bangalore"]:
        s_mask = site_arr == site_name
        s_hid = hidden_mask & s_mask
        if np.sum(s_hid) > 0:
            for b_name, b_preds in list(baseline_predictions.items()) + [("Joint TCN (Proposed)", y_rec)]:
                m_s = compute_regression_metrics(bam[s_hid], b_preds[s_hid])
                site_rows.append({
                    "site": site_name,
                    "model": b_name,
                    "hidden_rmse": m_s["rmse"],
                    "hidden_mae": m_s["mae"],
                    "hidden_r2": m_s["r2"],
                    "hidden_mbe": m_s["mbe"],
                })
    df_site = pd.DataFrame(site_rows)

    # 3. Pollution Band Breakdown (BAM <=60, 61-120, >120)
    bands = [
        ("Low (<=60 ug/m3)", bam <= 60.0),
        ("Moderate (61-120 ug/m3)", (bam > 60.0) & (bam <= 120.0)),
        ("Severe (>120 ug/m3)", bam > 120.0),
    ]
    poll_rows = []
    for band_label, p_idx in bands:
        p_hid = hidden_mask & p_idx
        if np.sum(p_hid) > 0:
            for b_name, b_preds in list(baseline_predictions.items()) + [("Joint TCN (Proposed)", y_rec)]:
                m_p = compute_regression_metrics(bam[p_hid], b_preds[p_hid])
                poll_rows.append({
                    "pollution_band": band_label,
                    "hidden_samples": int(np.sum(p_hid)),
                    "model": b_name,
                    "hidden_rmse": m_p["rmse"],
                    "hidden_mae": m_p["mae"],
                    "hidden_r2": m_p["r2"],
                    "hidden_mbe": m_p["mbe"],
                })
    df_poll = pd.DataFrame(poll_rows)

    # 4. Humidity Regime Breakdown (RH <40%, 40-65%, >65%)
    rh = test_df["relative_humidity"].to_numpy(dtype=float)
    hum_strata = [
        ("Low (<40% RH)", rh < 40.0),
        ("Medium (40-65% RH)", (rh >= 40.0) & (rh <= 65.0)),
        ("High (>65% RH)", rh > 65.0),
    ]
    hum_rows = []
    for hum_label, h_idx in hum_strata:
        h_hid = hidden_mask & h_idx
        if np.sum(h_hid) > 0:
            for b_name, b_preds in list(baseline_predictions.items()) + [("Joint TCN (Proposed)", y_rec)]:
                m_h = compute_regression_metrics(bam[h_hid], b_preds[h_hid])
                hum_rows.append({
                    "humidity_regime": hum_label,
                    "hidden_samples": int(np.sum(h_hid)),
                    "model": b_name,
                    "hidden_rmse": m_h["rmse"],
                    "hidden_mae": m_h["mae"],
                    "hidden_r2": m_h["r2"],
                    "hidden_mbe": m_h["mbe"],
                })
    df_hum = pd.DataFrame(hum_rows)

    # Predictions DataFrame
    df_preds = test_df[["timestamp", "site", "bam_reference"]].copy()
    df_preds["is_observed"] = mask.astype(int)
    df_preds["y_cal"] = y_cal
    df_preds["y_rec"] = y_rec
    df_preds["y_final"] = y_final

    return df_overall, df_site, df_poll, df_hum, df_preds
