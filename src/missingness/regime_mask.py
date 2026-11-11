"""
Module: src.missingness.regime_mask
Simulates Missing at Random (MAR) condition-dependent dropout patterns based on
environmental humidity regimes and sensor pollution intensity.
"""

from typing import Optional
import numpy as np
import pandas as pd


def generate_humidity_dependent_mask(
    df: pd.DataFrame,
    rh_col: str = "relative_humidity",
    missing_rate: float = 0.20,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulates humidity-conditioned missingness where high-humidity conditions
    exhibit significantly higher dropout probability (e.g., optical condensation / hygroscopic fogging).

    Calculates dropout probability as a sigmoid function scaled by RH, adjusted to yield the desired overall missing rate.
    """
    n_samples = len(df)
    rh_vals = df[rh_col].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    # Normalize RH to [0, 1]
    rh_norm = (rh_vals - np.nanmin(rh_vals)) / (np.nanmax(rh_vals) - np.nanmin(rh_vals) + 1e-6)

    # Sigmoid propensity centered around 65% RH
    propensity = 1.0 / (1.0 + np.exp(-6.0 * (rh_norm - 0.55)))
    propensity = np.clip(propensity, 0.05, 0.95)

    # Calibrate threshold to achieve exact target missing rate
    target_k = int(round(n_samples * missing_rate))
    noise = rng.uniform(0.0, 1.0, size=n_samples)
    scores = propensity * noise
    masked_indices = np.argsort(scores)[-target_k:]

    mask = np.ones(n_samples, dtype=bool)
    mask[masked_indices] = False
    return mask


def generate_lag_dependent_mask(
    df: pd.DataFrame,
    pm_col: str = "pa_cf1",
    missing_rate: float = 0.20,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulates Missing at Random (MAR) where dropout probability at time t depends
    strictly on the observed particulate concentration at the previous timestep t-1 (X_{t-1}).
    """
    n_samples = len(df)
    pm_vals = df[pm_col].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    # Shift by 1 step to obtain past observed concentration X_{t-1}
    pm_lag = np.roll(pm_vals, 1)
    pm_lag[0] = pm_vals[0]

    pm_log = np.log1p(np.maximum(pm_lag, 0.0))
    pm_norm = (pm_log - np.nanmin(pm_log)) / (np.nanmax(pm_log) - np.nanmin(pm_log) + 1e-6)

    propensity = 1.0 / (1.0 + np.exp(-5.0 * (pm_norm - 0.50)))
    propensity = np.clip(propensity, 0.05, 0.95)

    target_k = int(round(n_samples * missing_rate))
    noise = rng.uniform(0.0, 1.0, size=n_samples)
    scores = propensity * noise
    masked_indices = np.argsort(scores)[-target_k:]

    mask = np.ones(n_samples, dtype=bool)
    mask[masked_indices] = False
    return mask



def generate_pollution_dependent_mask(
    df: pd.DataFrame,
    pm_col: str = "pa_cf1",
    missing_rate: float = 0.20,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulates pollution-conditioned missingness where extreme particulate episodes
    increase sensor dropout risk (e.g. sensor fan saturation, optical threshold clipping).
    """
    n_samples = len(df)
    pm_vals = df[pm_col].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    # Log-transformed PM intensity
    pm_log = np.log1p(np.maximum(pm_vals, 0.0))
    pm_norm = (pm_log - np.nanmin(pm_log)) / (np.nanmax(pm_log) - np.nanmin(pm_log) + 1e-6)

    propensity = 1.0 / (1.0 + np.exp(-5.0 * (pm_norm - 0.50)))
    propensity = np.clip(propensity, 0.05, 0.95)

    target_k = int(round(n_samples * missing_rate))
    noise = rng.uniform(0.0, 1.0, size=n_samples)
    scores = propensity * noise
    masked_indices = np.argsort(scores)[-target_k:]

    mask = np.ones(n_samples, dtype=bool)
    mask[masked_indices] = False
    return mask


def generate_lag_dependent_mask(
    df: pd.DataFrame,
    pm_col: str = "pa_cf1",
    missing_rate: float = 0.20,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulates Missing at Random (MAR) where dropout probability at time t depends
    strictly on the observed particulate concentration at the previous timestep t-1 (X_{t-1}).
    """
    n_samples = len(df)
    pm_vals = df[pm_col].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    # Shift by 1 step to obtain past observed concentration X_{t-1}
    pm_lag = np.roll(pm_vals, 1)
    pm_lag[0] = pm_vals[0]

    pm_log = np.log1p(np.maximum(pm_lag, 0.0))
    pm_norm = (pm_log - np.nanmin(pm_log)) / (np.nanmax(pm_log) - np.nanmin(pm_log) + 1e-6)

    propensity = 1.0 / (1.0 + np.exp(-5.0 * (pm_norm - 0.50)))
    propensity = np.clip(propensity, 0.05, 0.95)

    target_k = int(round(n_samples * missing_rate))
    noise = rng.uniform(0.0, 1.0, size=n_samples)
    scores = propensity * noise
    masked_indices = np.argsort(scores)[-target_k:]

    mask = np.ones(n_samples, dtype=bool)
    mask[masked_indices] = False
    return mask

