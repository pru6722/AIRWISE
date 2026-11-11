"""
Module: src.data.feature_engineering
Extracts cyclical temporal features, constructs strict chronological train/val/test splits,
and fits standard scalers strictly on training partitions without data leakage.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_temporal_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cyclical sine and cosine encodings for hour of day and day of year.
    Preserves all existing columns and avoids timestamp ordering alteration.
    """
    df_out = df.copy()
    if not np.issubdtype(df_out["timestamp"].dtype, np.datetime64):
        df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])

    hours = df_out["timestamp"].dt.hour + df_out["timestamp"].dt.minute / 60.0
    df_out["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df_out["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)

    day_of_year = df_out["timestamp"].dt.dayofyear
    df_out["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25).astype(np.float32)
    df_out["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25).astype(np.float32)

    return df_out


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partitions the dataset chronologically within each measurement site.
    Ensures that validation and test sets strictly follow training observations in time.

    Args:
        df: Input DataFrame containing 'site' and 'timestamp'.
        train_ratio: Fraction of records for training (default 0.60).
        val_ratio: Fraction of records for validation (default 0.20).
        test_ratio: Fraction of records for test (default 0.20).

    Returns:
        train_df, val_df, test_df, split_metadata
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    df_featured = add_temporal_cyclical_features(df)
    train_dfs = []
    val_dfs = []
    test_dfs = []
    split_info: Dict[str, Any] = {}

    for site_name, group in df_featured.groupby("site", observed=False, sort=False):
        g = group.sort_values("timestamp").reset_index(drop=True)
        n = len(g)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        g_train = g.iloc[:n_train].copy()
        g_val = g.iloc[n_train : n_train + n_val].copy()
        g_test = g.iloc[n_train + n_val :].copy()

        g_train["split"] = "train"
        g_val["split"] = "val"
        g_test["split"] = "test"

        train_dfs.append(g_train)
        val_dfs.append(g_val)
        test_dfs.append(g_test)

        split_info[str(site_name)] = {
            "total_count": n,
            "train_count": len(g_train),
            "val_count": len(g_val),
            "test_count": len(g_test),
            "train_start": str(g_train["timestamp"].iloc[0]),
            "train_end": str(g_train["timestamp"].iloc[-1]),
            "val_start": str(g_val["timestamp"].iloc[0]),
            "val_end": str(g_val["timestamp"].iloc[-1]),
            "test_start": str(g_test["timestamp"].iloc[0]),
            "test_end": str(g_test["timestamp"].iloc[-1]),
        }

    train_combined = pd.concat(train_dfs, ignore_index=True)
    val_combined = pd.concat(val_dfs, ignore_index=True)
    test_combined = pd.concat(test_dfs, ignore_index=True)

    split_info["pooled"] = {
        "total_count": len(df_featured),
        "train_count": len(train_combined),
        "val_count": len(val_combined),
        "test_count": len(test_combined),
    }

    return train_combined, val_combined, test_combined, split_info


class FeatureScaler:
    """
    StandardScaler wrapper that fits strictly on training partitions
    and transforms validation / test sets without data leakage.
    """

    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureScaler":
        X = train_df[self.feature_columns].to_numpy(dtype=np.float32)
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted on training data before transform.")
        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        return self.scaler.transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled)
