"""
Module: src.uncertainty.split_manager
Manages the four-way chronological partition for conformal prediction:
- 60% Training
- 10% Model-Selection Validation
- 10% Conformal Calibration
- 20% Final Evaluation Test
Preserves exact original Checkpoint 3 training (60%) and test (20%) boundaries.
Splits the previous 20% validation partition chronologically into two halves.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

from src.data.feature_engineering import add_temporal_cyclical_features


def create_conformal_splits(
    df: pd.DataFrame,
    site_col: str = "site",
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Creates four strictly chronological partitions per site:
    - train_df: 60% (original training partition)
    - val_model_df: 10% (first half of original validation partition)
    - calib_conformal_df: 10% (second half of original validation partition)
    - test_df: 20% (original untouched test partition)

    Returns:
        train_df, val_model_df, calib_conformal_df, test_df, split_info
    """
    df_featured = add_temporal_cyclical_features(df)
    train_dfs = []
    val_model_dfs = []
    calib_conformal_dfs = []
    test_dfs = []
    split_info = {}

    for site_name, site_group in df_featured.groupby(site_col, observed=False, sort=False):
        site_sorted = site_group.sort_values(timestamp_col).reset_index(drop=True)
        n = len(site_sorted)

        # 1. Original 60/20/20 boundaries
        n_train = int(np.floor(n * 0.60))
        n_orig_val = int(np.floor(n * 0.20))
        # remainder goes to test
        n_test = n - n_train - n_orig_val

        # 2. Divide original validation partition into two chronological halves
        n_val_model = int(np.floor(n_orig_val * 0.50))
        n_calib_conf = n_orig_val - n_val_model

        # Slices
        train_s = site_sorted.iloc[:n_train].copy()
        val_model_s = site_sorted.iloc[n_train : n_train + n_val_model].copy()
        calib_conf_s = site_sorted.iloc[n_train + n_val_model : n_train + n_orig_val].copy()
        test_s = site_sorted.iloc[n_train + n_orig_val :].copy()

        train_dfs.append(train_s)
        val_model_dfs.append(val_model_s)
        calib_conformal_dfs.append(calib_conf_s)
        test_dfs.append(test_s)

        split_info[site_name] = {
            "total_count": n,
            "train_count": len(train_s),
            "train_start": str(train_s[timestamp_col].min()),
            "train_end": str(train_s[timestamp_col].max()),
            "val_model_count": len(val_model_s),
            "val_model_start": str(val_model_s[timestamp_col].min()),
            "val_model_end": str(val_model_s[timestamp_col].max()),
            "calib_conformal_count": len(calib_conf_s),
            "calib_conformal_start": str(calib_conf_s[timestamp_col].min()),
            "calib_conformal_end": str(calib_conf_s[timestamp_col].max()),
            "test_count": len(test_s),
            "test_start": str(test_s[timestamp_col].min()),
            "test_end": str(test_s[timestamp_col].max()),
        }

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_model_df = pd.concat(val_model_dfs, ignore_index=True)
    calib_conformal_df = pd.concat(calib_conformal_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    summary = {
        "n_total": len(df),
        "n_train": len(train_df),
        "n_val_model": len(val_model_df),
        "n_calib_conformal": len(calib_conformal_df),
        "n_test": len(test_df),
        "sites": split_info,
    }

    return train_df, val_model_df, calib_conformal_df, test_df, summary
