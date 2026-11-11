"""
Module: src.evaluation.bootstrap
Implements segment-constrained moving-block bootstrap (MBB) and paired hypothesis testing:
- Resamples contiguous temporal blocks strictly within segments/sites to preserve autocorrelation.
- Evaluates paired metric differences: Delta_RMSE = RMSE_Joint - RMSE_Baseline.
- Computes 95% bootstrap percentile confidence intervals and empirical p-values.
- Supports sensitivity sweeps across block lengths (12h, 24h, 48h, 72h).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def get_segment_block_indices(
    segments: np.ndarray,
    block_length: int = 24,
) -> List[Tuple[int, int]]:
    """
    Identifies all valid start/end index pairs of length `block_length`
    that fall entirely within the same contiguous segment.
    """
    n = len(segments)
    valid_blocks = []
    for start in range(n - block_length + 1):
        end = start + block_length
        # Ensure all points in this block belong to the exact same segment
        if segments[start] == segments[end - 1]:
            valid_blocks.append((start, end))

    if not valid_blocks:
        # Fallback if segments are shorter than block_length: take whole segments
        for seg_id in np.unique(segments):
            seg_indices = np.where(segments == seg_id)[0]
            if len(seg_indices) > 0:
                valid_blocks.append((seg_indices[0], seg_indices[-1] + 1))

    return valid_blocks


def paired_moving_block_bootstrap(
    y_true: np.ndarray,
    y_pred_joint: np.ndarray,
    y_pred_baseline: np.ndarray,
    segments: np.ndarray,
    block_length: int = 24,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes paired moving-block bootstrap for Delta_RMSE and Delta_MAE:
    Delta_RMSE = RMSE_Joint - RMSE_Baseline (negative values indicate Joint is superior).

    Args:
        y_true: Ground truth target array (N,).
        y_pred_joint: Model 1 (Joint TCN) predictions (N,).
        y_pred_baseline: Model 2 (Baseline) predictions (N,).
        segments: Segment ID array (N,) enforcing temporal continuity.
        block_length: Temporal block length in hours (default 24h for diurnal cycle).
        n_bootstraps: Number of bootstrap iterations.
        confidence_level: Nominal confidence level (e.g. 0.95 for 95% CI).
        random_seed: Reproducible seed.

    Returns:
        Dict with point estimates, confidence intervals, p-values, and bootstrap distributions.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_j = np.asarray(y_pred_joint, dtype=float)
    y_b = np.asarray(y_pred_baseline, dtype=float)
    segs = np.asarray(segments)

    # Filter out any non-finite pairs
    valid = np.isfinite(y_t) & np.isfinite(y_j) & np.isfinite(y_b)
    if np.sum(valid) < 10:
        return {
            "error": "Insufficient valid observations for bootstrap",
            "n_valid": int(np.sum(valid)),
        }

    y_t = y_t[valid]
    y_j = y_j[valid]
    y_b = y_b[valid]
    segs = segs[valid]

    n_samples = len(y_t)
    valid_blocks = get_segment_block_indices(segs, block_length=block_length)

    if not valid_blocks:
        valid_blocks = [(0, n_samples)]

    # Point estimates
    rmse_j = float(np.sqrt(np.mean((y_t - y_j) ** 2)))
    rmse_b = float(np.sqrt(np.mean((y_t - y_b) ** 2)))
    mae_j = float(np.mean(np.abs(y_t - y_j)))
    mae_b = float(np.mean(np.abs(y_t - y_b)))

    delta_rmse_point = rmse_j - rmse_b
    delta_mae_point = mae_j - mae_b

    rng = np.random.default_rng(random_seed)
    n_blocks_needed = int(np.ceil(n_samples / block_length))

    boot_delta_rmse = np.zeros(n_bootstraps, dtype=float)
    boot_delta_mae = np.zeros(n_bootstraps, dtype=float)

    for k in range(n_bootstraps):
        # Sample blocks with replacement
        chosen_block_indices = rng.choice(len(valid_blocks), size=n_blocks_needed, replace=True)
        resampled_indices = []
        for b_idx in chosen_block_indices:
            start, end = valid_blocks[b_idx]
            resampled_indices.extend(range(start, end))

        resampled_indices = np.array(resampled_indices[:n_samples])

        yt_k = y_t[resampled_indices]
        yj_k = y_j[resampled_indices]
        yb_k = y_b[resampled_indices]

        rmse_j_k = np.sqrt(np.mean((yt_k - yj_k) ** 2))
        rmse_b_k = np.sqrt(np.mean((yt_k - yb_k) ** 2))
        mae_j_k = np.mean(np.abs(yt_k - yj_k))
        mae_b_k = np.mean(np.abs(yt_k - yb_k))

        boot_delta_rmse[k] = rmse_j_k - rmse_b_k
        boot_delta_mae[k] = mae_j_k - mae_b_k

    # Percentile confidence intervals
    alpha_tail = (1.0 - confidence_level) / 2.0
    ci_lower_pct = alpha_tail * 100.0
    ci_upper_pct = (1.0 - alpha_tail) * 100.0

    delta_rmse_ci = (
        float(np.percentile(boot_delta_rmse, ci_lower_pct)),
        float(np.percentile(boot_delta_rmse, ci_upper_pct)),
    )
    delta_mae_ci = (
        float(np.percentile(boot_delta_mae, ci_lower_pct)),
        float(np.percentile(boot_delta_mae, ci_upper_pct)),
    )

    # Empirical p-value: proportion of bootstrap iterations where Joint is NOT superior (Delta >= 0)
    p_val_rmse = float(np.mean(boot_delta_rmse >= 0.0))
    p_val_mae = float(np.mean(boot_delta_mae >= 0.0))

    return {
        "n_samples": n_samples,
        "block_length_hours": block_length,
        "n_bootstraps": n_bootstraps,
        "confidence_level": confidence_level,
        "point_estimates": {
            "rmse_joint": round(rmse_j, 3),
            "rmse_baseline": round(rmse_b, 3),
            "delta_rmse": round(delta_rmse_point, 3),
            "mae_joint": round(mae_j, 3),
            "mae_baseline": round(mae_b, 3),
            "delta_mae": round(delta_mae_point, 3),
        },
        "delta_rmse_ci": (round(delta_rmse_ci[0], 3), round(delta_rmse_ci[1], 3)),
        "delta_mae_ci": (round(delta_mae_ci[0], 3), round(delta_mae_ci[1], 3)),
        "p_value_rmse": round(p_val_rmse, 4),
        "p_value_mae": round(p_val_mae, 4),
        "statistically_significant_05": bool(delta_rmse_ci[1] < 0.0 or delta_rmse_ci[0] > 0.0),
    }


def block_length_sensitivity_sweep(
    y_true: np.ndarray,
    y_pred_joint: np.ndarray,
    y_pred_baseline: np.ndarray,
    segments: np.ndarray,
    block_lengths: Optional[List[int]] = None,
    n_bootstraps: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Runs moving-block bootstrap across multiple block lengths [12, 24, 48, 72]."""
    if block_lengths is None:
        block_lengths = [12, 24, 48, 72]

    records = []
    for b in block_lengths:
        res = paired_moving_block_bootstrap(
            y_true=y_true,
            y_pred_joint=y_pred_joint,
            y_pred_baseline=y_pred_baseline,
            segments=segments,
            block_length=b,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
        )
        if "error" not in res:
            pe = res["point_estimates"]
            ci = res["delta_rmse_ci"]
            records.append({
                "block_length_hours": b,
                "delta_rmse": pe["delta_rmse"],
                "ci_lower_95": ci[0],
                "ci_upper_95": ci[1],
                "ci_width": round(ci[1] - ci[0], 3),
                "p_value": res["p_value_rmse"],
                "significant": res["statistically_significant_05"],
            })
    return pd.DataFrame(records)
