"""
Module: src.uncertainty.block_conformal
Implements Non-Overlapping Temporal Block-Based Conformal Prediction:
- Partitions calibration and test time series into contiguous blocks of duration B in {6, 12, 24} hours.
- Strictly respects continuous segment_id boundaries and site boundaries (never bridges natural missing gaps).
- Block non-conformity score: S_b = max_{t in Block_b} |y_t - y_hat_t|
- Conformal quantile calculated across blocks: k = ceil((M + 1)(1 - alpha))
- Evaluates simultaneous interval coverage and sensitivity to block duration.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

from src.uncertainty.conformal import compute_conformal_quantile, compute_uncertainty_metrics


def construct_temporal_blocks(
    df: pd.DataFrame,
    block_size: int = 12,
    segment_col: str = "segment_id",
    site_col: str = "site",
) -> List[np.ndarray]:
    """
    Constructs non-overlapping temporal blocks of size block_size strictly
    within contiguous segments and sites. Discards incomplete residual blocks at segment ends.

    Returns:
        blocks: List of 1D integer index arrays, each of length block_size.
    """
    blocks = []
    # Group by site and continuous segment
    for (_, _), seg_df in df.groupby([site_col, segment_col], observed=False, sort=False):
        indices = seg_df.index.to_numpy()
        n_seg = len(indices)
        n_blocks = n_seg // block_size

        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            blocks.append(indices[start:end])

    return blocks


class TemporalBlockConformalCalibrator:
    """
    Temporal Block-Based Conformal Calibrator.
    Accounts for serial autocorrelation by grouping consecutive hourly residuals into blocks.
    Score: Maximum absolute residual within the block.
    Target: Simultaneous block protection.
    """

    def __init__(
        self,
        block_size: int = 12,
        alphas: List[float] = [0.10, 0.05],
    ):
        self.block_size = block_size
        self.alphas = alphas
        self.block_scores: Optional[np.ndarray] = None
        self.block_quantiles: Dict[float, float] = {}
        self.n_calib_blocks: int = 0
        self.is_fitted = False

    def fit(
        self,
        calib_df: pd.DataFrame,
        y_true_col: str = "bam_reference",
        y_pred_col: str = "y_pred",
        segment_col: str = "segment_id",
        site_col: str = "site",
    ):
        """
        Constructs calibration blocks and computes block maximum absolute residuals.
        """
        blocks = construct_temporal_blocks(calib_df, block_size=self.block_size, segment_col=segment_col, site_col=site_col)
        self.n_calib_blocks = len(blocks)

        if self.n_calib_blocks == 0:
            raise ValueError(f"No valid temporal blocks of size {self.block_size} could be formed from calibration data.")

        y_true = calib_df[y_true_col].to_numpy(dtype=float)
        y_pred = calib_df[y_pred_col].to_numpy(dtype=float)
        res = np.abs(y_true - y_pred)

        scores = []
        for blk_idx in blocks:
            blk_max = float(np.max(res[blk_idx]))
            scores.append(blk_max)

        self.block_scores = np.array(scores, dtype=float)

        for alpha in self.alphas:
            self.block_quantiles[alpha] = compute_conformal_quantile(self.block_scores, alpha)

        self.is_fitted = True
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        alpha: float = 0.10,
        clip_zero: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies block-conformal quantile q_block to construct prediction intervals:
        [y_hat - q_block, y_hat + q_block]
        """
        if not self.is_fitted:
            raise RuntimeError("TemporalBlockConformalCalibrator must be fitted before predict_intervals.")

        if alpha not in self.block_quantiles:
            q = compute_conformal_quantile(self.block_scores, alpha)
            self.block_quantiles[alpha] = q
        else:
            q = self.block_quantiles[alpha]

        y_p = np.asarray(y_pred, dtype=float)
        lower = y_p - q
        upper = y_p + q

        if clip_zero:
            lower = np.maximum(0.0, lower)

        return lower, upper
