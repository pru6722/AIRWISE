"""
Module: src.missingness.block_mask
Simulates consecutive block outages of fixed temporal duration (e.g., 1, 3, 6, 12, 24 hours),
strictly respecting sequence segment boundaries.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def generate_block_outage_mask(
    df: pd.DataFrame,
    gap_hours: int = 6,
    missing_rate: float = 0.20,
    segment_col: str = "segment_id",
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generates a consecutive block missingness mask where blocks of length `gap_hours`
    are hidden until the cumulative missing rate approaches `missing_rate`.

    Blocks are placed strictly within contiguous segments (`segment_col`), preventing
    simulated outages from straddling real natural gaps.

    Args:
        df: Input DataFrame containing the sequence and `segment_col`.
        gap_hours: Duration of consecutive outage block in steps.
        missing_rate: Desired fraction of total points to mask (e.g. 0.20).
        segment_col: Column indicating contiguous temporal segment.
        random_seed: Reproducible seed.

    Returns:
        mask: 1D boolean array where True (1) = observed, False (0) = hidden.
    """
    n_total = len(df)
    target_masked = int(round(n_total * missing_rate))
    mask = np.ones(n_total, dtype=bool)

    if target_masked == 0:
        return mask

    rng = np.random.default_rng(random_seed)

    # Find valid starting indices for blocks within each segment
    segments = df[segment_col].to_numpy() if segment_col in df.columns else np.zeros(n_total, dtype=int)
    valid_starts = []

    for i in range(n_total - gap_hours + 1):
        # All points in the block must belong to the exact same segment
        if segments[i] == segments[i + gap_hours - 1]:
            valid_starts.append(i)

    if not valid_starts:
        # Fallback if segments are shorter than gap_hours: mask available contiguous sub-blocks
        valid_starts = list(range(max(1, n_total - gap_hours + 1)))

    rng.shuffle(valid_starts)

    currently_masked = 0
    for start_idx in valid_starts:
        end_idx = start_idx + gap_hours
        # Check how many unmasked points would be masked
        slice_unmasked = np.sum(mask[start_idx:end_idx])
        if slice_unmasked > 0:
            mask[start_idx:end_idx] = False
            currently_masked += slice_unmasked

        if currently_masked >= target_masked:
            break

    return mask
