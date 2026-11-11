"""
Module: src.data.preprocessing
Standardizes raw air quality data, attaches quality-control flags and block segment IDs,
and saves the canonical cleaned dataset as Parquet.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def preprocess_and_flag_dataset(
    df: pd.DataFrame,
    max_step_hours_for_continuity: float = 1.0,
) -> pd.DataFrame:
    """
    Applies data standardization, flags suspicious dew-point values, and attaches
    contiguous sequence identifiers for time-series modeling.

    Args:
        df: Input DataFrame with standardized column names.
        max_step_hours_for_continuity: Maximum gap to consider two consecutive readings contiguous.

    Returns:
        Cleaned, flagged DataFrame.
    """
    df_clean = df.copy()

    # Ensure timestamp is datetime and sort by site, then timestamp
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    df_clean = df_clean.sort_values(["site", "timestamp"]).reset_index(drop=True)

    # 1. Dew point physical consistency flag
    # If dew_point_raw > temperature, it is physically inconsistent in Celsius
    df_clean["is_dew_point_suspicious"] = (df_clean["dew_point_raw"] > df_clean["temperature"]).astype(int)

    # 2. Sequential continuity flags and block segmentation per site
    df_clean["time_diff_hours"] = np.nan
    df_clean["is_contiguous_step"] = 0
    df_clean["segment_id"] = 0

    processed_sites = []
    global_segment_counter = 0

    for site_name, group in df_clean.groupby("site", sort=False):
        g = group.sort_values("timestamp").copy()
        time_diffs = (g["timestamp"].diff().dt.total_seconds() / 3600.0).fillna(1.0)
        g["time_diff_hours"] = time_diffs.round(2)

        # Flag contiguous step
        is_contig = (time_diffs <= max_step_hours_for_continuity).astype(int)
        # First row is contiguous by definition for starting a block
        is_contig.iloc[0] = 1
        g["is_contiguous_step"] = is_contig

        # Assign contiguous block ID: increments whenever there is a time gap > max_step_hours
        new_block = (time_diffs > max_step_hours_for_continuity).astype(int)
        site_block_ids = new_block.cumsum() + global_segment_counter
        g["segment_id"] = site_block_ids
        global_segment_counter = site_block_ids.max() + 1

        processed_sites.append(g)

    final_df = pd.concat(processed_sites, ignore_index=True)

    # Enforce efficient and explicit dtypes
    dtypes = {
        "site": "category",
        "bam_reference": "float32",
        "pa_cf1": "float32",
        "pa_atm": "float32",
        "pa_alt": "float32",
        "relative_humidity": "float32",
        "temperature": "float32",
        "dew_point_raw": "float32",
        "is_dew_point_suspicious": "int8",
        "time_diff_hours": "float32",
        "is_contiguous_step": "int8",
        "segment_id": "int32",
    }
    for col, dtype in dtypes.items():
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(dtype)

    return final_df


def save_cleaned_dataset(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "data/processed/cleaned_data.parquet",
) -> Path:
    """Saves the preprocessed DataFrame to an Apache Parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def load_cleaned_dataset(
    parquet_path: Union[str, Path] = "data/processed/cleaned_data.parquet",
) -> pd.DataFrame:
    """Loads the preprocessed Parquet dataset."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cleaned parquet not found at {parquet_path.resolve()}")
    return pd.read_parquet(parquet_path, engine="pyarrow")
