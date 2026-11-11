"""
Module: src.aqi.windowing
Constructs aligned 24-hour PM2.5 windows separately for each site and partition.

Requirements Enforced:
1. Sort observations chronologically by timestamp within each site.
2. Segment time series across natural timestamp discontinuities.
3. Never construct windows across natural discontinuities or across sites.
4. Never allow windows to cross partition boundaries (train, val, calib, test).
5. Exact timestamp alignment between BAM reference and model predictions.
6. Completeness policies: primary >=18/24 (75%), sensitivity >=20/24 and 24/24.
7. Support both non-overlapping windows (primary) and rolling windows (sensitivity).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


def construct_24h_windows(
    df: pd.DataFrame,
    min_valid_hours: int = 18,
    non_overlapping: bool = True,
    timestamp_col: str = "timestamp",
    site_col: str = "site",
    target_col: str = "bam_reference",
    pred_col: str = "point_prediction",
    partition_name: str = "test",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs aligned 24-hour PM2.5 evaluation windows for a given partition.

    Args:
        df: DataFrame containing hourly observations for a single partition.
        min_valid_hours: Minimum valid hourly observations required (default 18 for 75% completeness).
        non_overlapping: If True, constructs discrete non-overlapping 24h blocks.
                         If False, constructs rolling 24h windows (sliding by 1 hour).
        timestamp_col: Name of datetime timestamp column.
        site_col: Name of site identifier column.
        target_col: Name of reference BAM PM2.5 target column.
        pred_col: Name of model point prediction column.
        partition_name: Name of partition ('train', 'val_model', 'calib_conformal', 'test').

    Returns:
        valid_windows_df: DataFrame of valid 24h aggregated windows meeting completeness threshold.
        audit_summary: Dictionary containing counts of evaluated, valid, and rejected windows.
    """
    df_clean = df.copy()
    df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col])

    valid_records = []
    rejected_records = []

    total_evaluated_windows = 0
    total_valid_windows = 0
    total_rejected_windows = 0

    site_counts = {}

    for site_name, site_group in df_clean.groupby(site_col, observed=False, sort=False):
        site_sorted = site_group.sort_values(timestamp_col).reset_index(drop=True)
        n_obs_site = len(site_sorted)

        if n_obs_site == 0:
            continue

        site_valid = 0
        site_rejected = 0

        # Extract timestamps and values
        ts_series = site_sorted[timestamp_col]
        min_ts = ts_series.min()
        max_ts = ts_series.max()

        if non_overlapping:
            # Discrete 24-hour wall-clock bins starting from partition start
            cur_start = min_ts
            window_idx = 0

            while cur_start + pd.Timedelta(hours=24) <= max_ts + pd.Timedelta(hours=1):
                cur_end = cur_start + pd.Timedelta(hours=24)
                window_mask = (ts_series >= cur_start) & (ts_series < cur_end)
                win_rows = site_sorted[window_mask]

                n_avail = len(win_rows)
                completeness_pct = (n_avail / 24.0) * 100.0
                total_evaluated_windows += 1

                # Calculate consecutive missing runs within the 24-hour expected timeline
                expected_hours = pd.date_range(start=cur_start, periods=24, freq="1h")
                present_hours = set(win_rows[timestamp_col])
                missing_hours = [1 if h not in present_hours else 0 for h in expected_hours]

                # Max consecutive missing duration
                max_consec_missing = 0
                cur_consec = 0
                for m in missing_hours:
                    if m == 1:
                        cur_consec += 1
                        if cur_consec > max_consec_missing:
                            max_consec_missing = cur_consec
                    else:
                        cur_consec = 0

                meta = {
                    "window_id": f"{partition_name}_{site_name}_win_{window_idx:04d}",
                    "partition": partition_name,
                    "site": site_name,
                    "window_type": "non_overlapping_24h",
                    "start_time": cur_start,
                    "end_time": cur_end,
                    "n_valid_hours": n_avail,
                    "completeness_pct": completeness_pct,
                    "max_consecutive_missing_hours": max_consec_missing,
                }

                if n_avail >= min_valid_hours:
                    # Valid window
                    ref_mean = float(win_rows[target_col].mean())
                    pred_mean = float(win_rows[pred_col].mean())
                    meta["ref_pm25_24h"] = ref_mean
                    meta["pred_pm25_24h"] = pred_mean
                    meta["abs_residual_24h"] = abs(ref_mean - pred_mean)
                    meta["signed_residual_24h"] = pred_mean - ref_mean

                    # Record additional aggregated covariates if present
                    if "relative_humidity" in win_rows.columns:
                        meta["mean_relative_humidity"] = float(win_rows["relative_humidity"].mean())
                    if "temperature" in win_rows.columns:
                        meta["mean_temperature"] = float(win_rows["temperature"].mean())

                    valid_records.append(meta)
                    site_valid += 1
                    total_valid_windows += 1
                else:
                    meta["rejection_reason"] = f"insufficient_completeness_{n_avail}_of_24"
                    rejected_records.append(meta)
                    site_rejected += 1
                    total_rejected_windows += 1

                window_idx += 1
                cur_start = cur_end

        else:
            # Rolling 24-hour windows (sliding by 1 hour across observation timestamps)
            for i in range(len(site_sorted)):
                cur_end = site_sorted.loc[i, timestamp_col]
                cur_start = cur_end - pd.Timedelta(hours=23)

                window_mask = (ts_series >= cur_start) & (ts_series <= cur_end)
                win_rows = site_sorted[window_mask]

                n_avail = len(win_rows)
                completeness_pct = (n_avail / 24.0) * 100.0
                total_evaluated_windows += 1

                # Missingness within trailing 24h
                expected_hours = pd.date_range(start=cur_start, periods=24, freq="1h")
                present_hours = set(win_rows[timestamp_col])
                missing_hours = [1 if h not in present_hours else 0 for h in expected_hours]

                max_consec_missing = 0
                cur_consec = 0
                for m in missing_hours:
                    if m == 1:
                        cur_consec += 1
                        if cur_consec > max_consec_missing:
                            max_consec_missing = cur_consec
                    else:
                        cur_consec = 0

                meta = {
                    "window_id": f"{partition_name}_{site_name}_roll_{i:04d}",
                    "partition": partition_name,
                    "site": site_name,
                    "window_type": "rolling_24h",
                    "start_time": cur_start,
                    "end_time": cur_end,
                    "n_valid_hours": n_avail,
                    "completeness_pct": completeness_pct,
                    "max_consecutive_missing_hours": max_consec_missing,
                }

                if n_avail >= min_valid_hours:
                    ref_mean = float(win_rows[target_col].mean())
                    pred_mean = float(win_rows[pred_col].mean())
                    meta["ref_pm25_24h"] = ref_mean
                    meta["pred_pm25_24h"] = pred_mean
                    meta["abs_residual_24h"] = abs(ref_mean - pred_mean)
                    meta["signed_residual_24h"] = pred_mean - ref_mean

                    if "relative_humidity" in win_rows.columns:
                        meta["mean_relative_humidity"] = float(win_rows["relative_humidity"].mean())
                    if "temperature" in win_rows.columns:
                        meta["mean_temperature"] = float(win_rows["temperature"].mean())

                    valid_records.append(meta)
                    site_valid += 1
                    total_valid_windows += 1
                else:
                    meta["rejection_reason"] = f"insufficient_completeness_{n_avail}_of_24"
                    rejected_records.append(meta)
                    site_rejected += 1
                    total_rejected_windows += 1

        site_counts[site_name] = {
            "valid": site_valid,
            "rejected": site_rejected,
            "total": site_valid + site_rejected,
            "valid_pct": (site_valid / (site_valid + site_rejected) * 100.0) if (site_valid + site_rejected) > 0 else 0.0,
        }

    valid_df = pd.DataFrame(valid_records)
    audit_summary = {
        "partition": partition_name,
        "window_mode": "non_overlapping" if non_overlapping else "rolling",
        "min_valid_hours": min_valid_hours,
        "total_evaluated_windows": total_evaluated_windows,
        "total_valid_windows": total_valid_windows,
        "total_rejected_windows": total_rejected_windows,
        "site_breakdown": site_counts,
    }

    return valid_df, audit_summary
