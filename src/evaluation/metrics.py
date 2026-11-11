"""
Module: src.evaluation.metrics
Calculates standard air quality calibration and reconstruction metrics.
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import r2_score


def compute_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, R^2, Mean Bias Error (MBE), and Relative Bias (%).

    Args:
        y_true: Ground truth target array (reference BAM PM2.5).
        y_pred: Predicted / calibrated / reconstructed array.

    Returns:
        Dictionary with mae, rmse, r2, mbe, relative_bias_pct.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    # Filter any NaNs if present
    valid_mask = ~(np.isnan(yt) | np.isnan(yp))
    if not np.any(valid_mask):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mbe": np.nan, "relative_bias_pct": np.nan}

    yt = yt[valid_mask]
    yp = yp[valid_mask]

    errors = yp - yt
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mbe = float(np.mean(errors))
    mean_true = float(np.mean(yt))
    rel_bias = (mbe / mean_true) * 100.0 if mean_true != 0 else np.nan

    # R^2 score
    r2 = float(r2_score(yt, yp))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mbe": round(mbe, 2),
        "relative_bias_pct": round(rel_bias, 2),
    }
