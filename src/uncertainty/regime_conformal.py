"""
Module: src.uncertainty.regime_conformal
Implements Regime-Conditioned Conformal Prediction:
- Humidity regimes: Low (<40%), Medium (40-65%), High (>65%)
- Predicted pollution regimes: Low (<=60 ug/m3), Medium (60-120 ug/m3), High (>120 ug/m3)
  (Operational prediction-time regimes strictly based on y_pred, NEVER true BAM)
- Site-conditioned calibration: Delhi, Hamirpur, Bangalore
- Minimum group calibration sample threshold with documented fallback
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd

from src.uncertainty.conformal import compute_conformal_quantile, compute_uncertainty_metrics


def assign_humidity_regime(rh_values: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Assigns prediction-time humidity regime based on available ambient RH:
    - 'Low (<40% RH)': RH < 40.0
    - 'Medium (40-65% RH)': 40.0 <= RH <= 65.0
    - 'High (>65% RH)': RH > 65.0
    - Fallback 'Unknown RH' if missing.
    """
    rh = np.asarray(rh_values, dtype=float)
    regimes = np.empty(len(rh), dtype=object)

    low_mask = rh < 40.0
    med_mask = (rh >= 40.0) & (rh <= 65.0)
    high_mask = rh > 65.0
    nan_mask = np.isnan(rh)

    regimes[low_mask] = "Low (<40% RH)"
    regimes[med_mask] = "Medium (40-65% RH)"
    regimes[high_mask] = "High (>65% RH)"
    regimes[nan_mask] = "Unknown RH"

    return regimes


def assign_predicted_pollution_regime(y_pred_values: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Assigns operational prediction-time pollution regime strictly based on model predicted PM2.5.
    CRITICAL: Never uses true ground-truth BAM for regime assignment!
    - 'Low (<=60 ug/m3)': y_pred <= 60.0
    - 'Moderate (60-120 ug/m3)': 60.0 < y_pred <= 120.0
    - 'Severe (>120 ug/m3)': y_pred > 120.0
    """
    y_pred = np.asarray(y_pred_values, dtype=float)
    regimes = np.empty(len(y_pred), dtype=object)

    low_mask = y_pred <= 60.0
    med_mask = (y_pred > 60.0) & (y_pred <= 120.0)
    high_mask = y_pred > 120.0

    regimes[low_mask] = "Low (<=60 ug/m3)"
    regimes[med_mask] = "Moderate (60-120 ug/m3)"
    regimes[high_mask] = "Severe (>120 ug/m3)"

    return regimes


class RegimeConformalCalibrator:
    """
    Regime-Conditioned Conformal Calibrator.
    Partitions calibration non-conformity scores by environmental or operational regimes.
    Enforces minimum calibration group size (min_group_size) and falls back to pooled quantiles when needed.
    """

    def __init__(
        self,
        conditioning_variable: str,
        min_group_size: int = 30,
        alphas: List[float] = [0.10, 0.05],
    ):
        self.conditioning_variable = conditioning_variable
        self.min_group_size = min_group_size
        self.alphas = alphas

        self.group_residuals: Dict[str, np.ndarray] = {}
        self.group_quantiles: Dict[str, Dict[float, float]] = {}
        self.group_counts: Dict[str, int] = {}
        self.fallback_flags: Dict[str, bool] = {}

        self.pooled_residuals: Optional[np.ndarray] = None
        self.pooled_quantiles: Dict[float, float] = {}
        self.is_fitted = False

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regimes: np.ndarray,
    ):
        """
        Fits regime-specific conformal quantiles.
        Falls back to pooled residuals if a regime's sample size < min_group_size.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        reg = np.asarray(regimes, dtype=object)

        valid_idx = ~np.isnan(y_t) & ~np.isnan(y_p)
        y_t = y_t[valid_idx]
        y_p = y_p[valid_idx]
        reg = reg[valid_idx]

        # 1. Fit pooled baseline as universal fallback
        self.pooled_residuals = np.abs(y_t - y_p)
        for alpha in self.alphas:
            self.pooled_quantiles[alpha] = compute_conformal_quantile(self.pooled_residuals, alpha)

        # 2. Fit group-specific quantiles
        unique_groups = np.unique(reg)
        for g in unique_groups:
            g_mask = reg == g
            g_res = np.abs(y_t[g_mask] - y_p[g_mask])
            n_g = len(g_res)
            self.group_counts[g] = n_g
            self.group_residuals[g] = g_res
            self.group_quantiles[g] = {}

            if n_g >= self.min_group_size:
                self.fallback_flags[g] = False
                for alpha in self.alphas:
                    self.group_quantiles[g][alpha] = compute_conformal_quantile(g_res, alpha)
            else:
                # Fallback to pooled quantile due to insufficient sample size
                self.fallback_flags[g] = True
                for alpha in self.alphas:
                    self.group_quantiles[g][alpha] = self.pooled_quantiles[alpha]

        self.is_fitted = True
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        regimes: np.ndarray,
        alpha: float = 0.10,
        clip_zero: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates regime-conditioned prediction intervals for test data.
        Returns:
            lower: 1D array of lower prediction bounds
            upper: 1D array of upper prediction bounds
            fallback_applied: 1D boolean array indicating whether pooled fallback was used
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeConformalCalibrator must be fitted before predict_intervals.")

        y_p = np.asarray(y_pred, dtype=float)
        reg = np.asarray(regimes, dtype=object)
        n = len(y_p)

        lower = np.empty(n, dtype=float)
        upper = np.empty(n, dtype=float)
        fallback_applied = np.zeros(n, dtype=bool)

        for i in range(n):
            g = reg[i]
            if g in self.group_quantiles and not self.fallback_flags.get(g, False):
                q = self.group_quantiles[g][alpha]
                fallback_applied[i] = False
            else:
                # Use pooled quantile for unknown regime or under-sampled regime
                q = self.pooled_quantiles[alpha]
                fallback_applied[i] = True

            lower[i] = y_p[i] - q
            upper[i] = y_p[i] + q

        if clip_zero:
            lower = np.maximum(0.0, lower)

        return lower, upper, fallback_applied
