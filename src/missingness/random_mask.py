"""
Module: src.missingness.random_mask
Simulates Missing Completely at Random (MCAR) dropout patterns.
"""

from typing import Optional
import numpy as np


def generate_mcar_mask(
    n_samples: int,
    missing_rate: float = 0.20,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generates a 1D boolean observation mask under Missing Completely At Random (MCAR).

    Args:
        n_samples: Total number of observations in sequence.
        missing_rate: Proportion of observations to hide (0.0 to 1.0).
        random_seed: Random seed for reproducibility.

    Returns:
        mask: 1D boolean array where True (1) = observed, False (0) = hidden.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    mask = rng.uniform(0.0, 1.0, size=n_samples) >= missing_rate
    return mask.astype(bool)
