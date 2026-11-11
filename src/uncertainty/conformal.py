"""
Module: src.uncertainty.conformal
Implements standard inductive split conformal prediction for continuous regression:
- Absolute non-conformity scores: s_i = |y_i - y_hat_i|
- Exact finite-sample quantile calculation: k = ceil((n + 1)(1 - alpha))
- Construction of prediction intervals at 90% (alpha=0.10) and 95% (alpha=0.05) nominal levels
- Evaluation metrics: empirical coverage, interval width, coverage error, and Winkler interval score
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd


def compute_conformal_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Computes the exact finite-sample conformal quantile for non-conformity scores.
    Order statistic rank: k = ceil((n + 1) * (1 - alpha))

    Args:
        scores: 1D array of calibration non-conformity scores (absolute residuals).
        alpha: Significance level in (0, 1), e.g., 0.10 for 90% coverage.

    Returns:
        conformal_quantile: The k-th order statistic.
    """
    scores_clean = np.asarray(scores, dtype=float)
    scores_clean = scores_clean[~np.isnan(scores_clean)]
    n = len(scores_clean)

    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty scores array.")

    # Sort calibration scores in ascending order
    sorted_scores = np.sort(scores_clean)

    # Rank formula: k = ceil((n + 1) * (1 - alpha))
    k = int(np.ceil((n + 1) * (1.0 - alpha)))

    # Handle boundary edge cases
    if k > n:
        # Conservative finite-sample fallback: return max observed score
        return float(sorted_scores[-1])
    elif k < 1:
        return float(sorted_scores[0])

    # 1-based index k corresponds to 0-based index k - 1
    return float(sorted_scores[k - 1])


def compute_winkler_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Computes the Winkler Interval Score (IS_alpha):
    IS_alpha = (Upper - Lower) + (2 / alpha) * (Lower - Y) * I(Y < Lower)
                               + (2 / alpha) * (Y - Upper) * I(Y > Upper)
    Penalizes both wider intervals and coverage violations. Lower score is better.
    """
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)

    width = u - l
    under_penalty = np.where(y < l, (2.0 / alpha) * (l - y), 0.0)
    over_penalty = np.where(y > u, (2.0 / alpha) * (y - u), 0.0)

    total_score = width + under_penalty + over_penalty
    return float(np.mean(total_score))


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    """
    Computes full set of uncertainty metrics for prediction intervals:
    - Empirical coverage (%)
    - Nominal coverage (%)
    - Coverage error (%)
    - Mean interval width (ug/m3)
    - Median interval width (ug/m3)
    - Winkler Interval Score
    """
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)

    is_covered = (y >= l) & (y <= u)
    emp_coverage = float(np.mean(is_covered) * 100.0)
    nom_coverage = float((1.0 - alpha) * 100.0)
    cov_error = float(emp_coverage - nom_coverage)

    widths = u - l
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    winkler = compute_winkler_interval_score(y, l, u, alpha)

    # Point regression metrics
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y)))
    mbe = float(np.mean(y_pred - y))

    return {
        "n_samples": int(len(y)),
        "nominal_coverage": nom_coverage,
        "empirical_coverage": round(emp_coverage, 2),
        "coverage_error": round(cov_error, 2),
        "mean_interval_width": round(mean_width, 2),
        "median_interval_width": round(median_width, 2),
        "winkler_score": round(winkler, 2),
        "point_rmse": round(rmse, 2),
        "point_mae": round(mae, 2),
        "point_mbe": round(mbe, 2),
    }


class ConformalCalibrator:
    """
    Standard Split Conformal Calibrator for continuous scalar predictions.
    Stores calibration non-conformity scores and computes prediction intervals.
    """

    def __init__(self, name: str = "StandardSplitConformal"):
        self.name = name
        self.residuals: Optional[np.ndarray] = None
        self.quantiles: Dict[float, float] = {}
        self.is_fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, alphas: List[float] = [0.10, 0.05]):
        """
        Fits calibration non-conformity scores on hold-out calibration data.
        s_i = |y_i - y_hat_i|
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        valid_idx = ~np.isnan(y_t) & ~np.isnan(y_p)
        self.residuals = np.abs(y_t[valid_idx] - y_p[valid_idx])

        for alpha in alphas:
            q = compute_conformal_quantile(self.residuals, alpha)
            self.quantiles[alpha] = q

        self.is_fitted = True
        return self

    def predict_interval(
        self,
        y_pred: np.ndarray,
        alpha: float = 0.10,
        clip_zero: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs prediction intervals: [y_hat - q, y_hat + q]
        """
        if not self.is_fitted:
            raise RuntimeError("ConformalCalibrator must be fit on calibration data before predict_interval.")

        if alpha not in self.quantiles:
            q = compute_conformal_quantile(self.residuals, alpha)
            self.quantiles[alpha] = q
        else:
            q = self.quantiles[alpha]

        y_p = np.asarray(y_pred, dtype=float)
        lower = y_p - q
        upper = y_p + q

        if clip_zero:
            lower = np.maximum(0.0, lower)

        return lower, upper
