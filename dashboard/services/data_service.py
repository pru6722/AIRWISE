"""
AIRWISE Dashboard: Data Service
Manages cached loading and slicing of datasets, prediction outputs, and report tables.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import (
    CLEANED_PARQUET_PATH,
    RAW_EXCEL_PATH,
    CANONICAL_MASK_PATH,
    REPORTS_DIR,
    CALIBRATION_PREDICTIONS_CSV,
    RECONSTRUCTION_PREDICTIONS_CSV,
    JOINT_TCN_PREDICTIONS_CSV,
    CONFORMAL_PREDICTIONS_CSV,
    CHECKPOINT5_24H_PREDICTIONS_CSV,
    CHECKPOINT5_24H_INTERVALS_CSV,
    CHECKPOINT5_CATEGORY_ASSESSMENT_CSV,
    BASELINE_SUMMARY_CSV,
    JOINT_TCN_METRICS_CSV,
    JOINT_TCN_ABLATION_CSV,
    CONFORMAL_COVERAGE_CSV,
    CONFORMAL_SITE_METRICS_CSV,
    CONFORMAL_SPLIT_SUMMARY_PATH,
    CHECKPOINT5_24H_POINT_METRICS_CSV,
    CHECKPOINT5_24H_COVERAGE_METRICS_CSV,
    CHECKPOINT5_CATEGORY_METRICS_CSV,
    CHECKPOINT5_CATEGORY_STABILITY_CSV,
    CHECKPOINT5_BOUNDARY_ANALYSIS_CSV,
    CHECKPOINT5_SITE_RELIABILITY_CSV,
    CHECKPOINT6_SCALING_BENCHMARK_CSV,
    CHECKPOINT6_STORAGE_FORMATS_CSV,
    CHECKPOINT6_PARTITIONING_CSV,
    CHECKPOINT6_CACHING_JSON,
    CHECKPOINT6_SQL_QUERIES_CSV,
    CORRECTED_RECONSTRUCTION_METRICS_CSV,
    CORRECTED_RECONSTRUCTION_PREDICTIONS_CSV,
    CONFORMAL_COVERAGE_RECONCILIATION_CSV,
    CONFORMAL_SIMULTANEOUS_BLOCK_COVERAGE_CSV,
    CPCB_CONTROLLED_COMPARISON_CSV,
    CPCB_STABILITY_ASSESSMENT_CSV,
    CALIBRATION_BY_POLLUTION_RANGE_CSV,
    HUMIDITY_METRICS_CSV,
    DATASET_AUDIT_CSV,
    SENSOR_BIAS_SUMMARY_CSV,
    BENCHMARK_PREDICTIONS_PARQUET,
    SCENARIO_METRICS_OVERALL_CSV,
    SCENARIO_METRICS_MISSING_PM_CSV,
    SCENARIO_METRICS_COMMON_SUPPORT_CSV,
    BOOTSTRAP_SIGNIFICANCE_CSV,
    BOOTSTRAP_SENSITIVITY_CSV,
    SITE_STRATIFIED_METRICS_CSV,
    JOINT_TCN_V2_METADATA_PATH,
    JOINT_TCN_V2_CHECKPOINT_PATH,
)


@st.cache_data(show_spinner=False)
def load_cleaned_dataset() -> pd.DataFrame:
    """Loads the authoritative 18,584-row cleaned and segmented Parquet dataset."""
    df = pd.read_parquet(CLEANED_PARQUET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def get_raw_sheet_names() -> List[str]:
    """Returns available sheets in the raw Campmier Excel workbook."""
    import openpyxl
    wb = openpyxl.load_workbook(RAW_EXCEL_PATH, read_only=True)
    return wb.sheetnames


@st.cache_data(show_spinner=False)
def load_raw_sheet(sheet_name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Loads a specific sheet from the raw Excel workbook."""
    return pd.read_excel(RAW_EXCEL_PATH, sheet_name=sheet_name, nrows=nrows)


@st.cache_data(show_spinner=False)
def load_canonical_mask() -> np.ndarray:
    """Loads the canonical 20% MCAR benchmark mask for the 3,720-row test split."""
    if CANONICAL_MASK_PATH.exists():
        return np.load(CANONICAL_MASK_PATH)
    return np.ones(3720, dtype=bool)


@st.cache_data(show_spinner=False)
def load_predictions(name: str) -> pd.DataFrame:
    """Loads precomputed prediction tables by checkpoint name."""
    mapping = {
        "calibration": CALIBRATION_PREDICTIONS_CSV,
        "reconstruction": RECONSTRUCTION_PREDICTIONS_CSV,
        "reconstruction_corrected": CORRECTED_RECONSTRUCTION_PREDICTIONS_CSV,
        "joint_tcn": JOINT_TCN_PREDICTIONS_CSV,
        "conformal": CONFORMAL_PREDICTIONS_CSV,
        "checkpoint5_24h_predictions": CHECKPOINT5_24H_PREDICTIONS_CSV,
        "checkpoint5_24h_intervals": CHECKPOINT5_24H_INTERVALS_CSV,
        "checkpoint5_category_assessment": CHECKPOINT5_CATEGORY_ASSESSMENT_CSV,
        "cpcb_stability_assessment": CPCB_STABILITY_ASSESSMENT_CSV,
    }
    path = mapping.get(name)
    if path and path.exists():
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics_table(name: str) -> pd.DataFrame:
    """Loads authoritative experiment metrics and report summaries."""
    mapping = {
        "baseline_summary": BASELINE_SUMMARY_CSV,
        "joint_tcn_metrics": JOINT_TCN_METRICS_CSV,
        "joint_tcn_ablation": JOINT_TCN_ABLATION_CSV,
        "conformal_coverage": CONFORMAL_COVERAGE_CSV,
        "conformal_site_metrics": CONFORMAL_SITE_METRICS_CSV,
        "conformal_block_comparison": REPORTS_DIR / "conformal_block_comparison.csv",
        "conformal_humidity_metrics": REPORTS_DIR / "conformal_humidity_metrics.csv",
        "conformal_pollution_metrics": REPORTS_DIR / "conformal_pollution_metrics.csv",
        "conformal_extreme_pollution": REPORTS_DIR / "conformal_extreme_pollution.csv",
        "checkpoint5_point_metrics": CHECKPOINT5_24H_POINT_METRICS_CSV,
        "checkpoint5_24h_point_metrics": CHECKPOINT5_24H_POINT_METRICS_CSV,
        "checkpoint5_coverage_metrics": CHECKPOINT5_24H_COVERAGE_METRICS_CSV,
        "checkpoint5_24h_coverage_metrics": CHECKPOINT5_24H_COVERAGE_METRICS_CSV,
        "checkpoint5_category_metrics": CHECKPOINT5_CATEGORY_METRICS_CSV,
        "checkpoint5_category_stability": CHECKPOINT5_CATEGORY_STABILITY_CSV,
        "checkpoint5_boundary_analysis": CHECKPOINT5_BOUNDARY_ANALYSIS_CSV,
        "checkpoint5_site_reliability": CHECKPOINT5_SITE_RELIABILITY_CSV,
        "checkpoint6_scaling": CHECKPOINT6_SCALING_BENCHMARK_CSV,
        "checkpoint6_storage_formats": CHECKPOINT6_STORAGE_FORMATS_CSV,
        "checkpoint6_storage": CHECKPOINT6_STORAGE_FORMATS_CSV,
        "checkpoint6_partitioning": CHECKPOINT6_PARTITIONING_CSV,
        "checkpoint6_sql_queries": CHECKPOINT6_SQL_QUERIES_CSV,
        "corrected_reconstruction_metrics": CORRECTED_RECONSTRUCTION_METRICS_CSV,
        "conformal_coverage_reconciliation": CONFORMAL_COVERAGE_RECONCILIATION_CSV,
        "conformal_simultaneous_block_coverage": CONFORMAL_SIMULTANEOUS_BLOCK_COVERAGE_CSV,
        "cpcb_controlled_comparison": CPCB_CONTROLLED_COMPARISON_CSV,
        "calibration_by_pollution_range": CALIBRATION_BY_POLLUTION_RANGE_CSV,
        "humidity_metrics": HUMIDITY_METRICS_CSV,
        "dataset_audit": DATASET_AUDIT_CSV,
        "sensor_bias_summary": SENSOR_BIAS_SUMMARY_CSV,
        "missing_timestamp_report": REPORTS_DIR / "missing_timestamp_report.csv",
        "dew_point_investigation": REPORTS_DIR / "dew_point_investigation.csv",
    }
    path = mapping.get(name)
    if path and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_split_summary() -> Dict[str, Any]:
    """Loads per-site chronological split counts and exact boundaries from split_summary.json."""
    if CONFORMAL_SPLIT_SUMMARY_PATH.exists():
        with open(CONFORMAL_SPLIT_SUMMARY_PATH, "r") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_caching_benchmark() -> Dict[str, Any]:
    """Loads Spark SQL RAM caching benchmark timings from checkpoint6_caching.json."""
    if CHECKPOINT6_CACHING_JSON.exists():
        with open(CHECKPOINT6_CACHING_JSON, "r") as f:
            return json.load(f)
    return {}


def filter_dataset(
    df: pd.DataFrame,
    site: Optional[str] = None,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
) -> pd.DataFrame:
    """Slices dataset safely by site and date range without modifying original data."""
    filtered = df.copy()
    if site and site != "All Sites":
        filtered = filtered[filtered["site"] == site]
    if start_date:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[filtered["timestamp"] <= end_ts]
    return filtered


# =============================================================================
# Phase 3C Benchmark Data Services (Four Pipelines & Nine Scenarios)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_benchmark_predictions() -> pd.DataFrame:
    """
    Loads the authoritative Phase 3C benchmark predictions (3,720 test rows x 49 columns).
    Includes ground-truth BAM references, binary scenario masks, and predictions for
    Pipelines A, B, C, D across all 9 missingness scenarios.
    """
    if not BENCHMARK_PREDICTIONS_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(BENCHMARK_PREDICTIONS_PARQUET)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_phase3c_metrics(table_key: str) -> pd.DataFrame:
    """
    Loads precomputed Phase 3C benchmark metric tables.
    Valid keys:
      - 'overall': scenario_metrics_overall.csv (Full test population N=3,720)
      - 'missing_pm': scenario_metrics_missing_pm.csv (Reconstruction on missing intervals)
      - 'common_support': scenario_metrics_common_support.csv (Matched calibration on m_t = 1)
      - 'bootstrap_significance': bootstrap_significance_results.csv (Canonical b=24h bootstrap)
      - 'bootstrap_sensitivity': bootstrap_sensitivity_sweep.csv (b in {12, 24, 48, 72}h sweep)
      - 'site_stratified': site_stratified_metrics.csv (Site-level breakdown across BLR, DEL, HAM)
    """
    mapping = {
        "overall": SCENARIO_METRICS_OVERALL_CSV,
        "missing_pm": SCENARIO_METRICS_MISSING_PM_CSV,
        "common_support": SCENARIO_METRICS_COMMON_SUPPORT_CSV,
        "bootstrap_significance": BOOTSTRAP_SIGNIFICANCE_CSV,
        "bootstrap_sensitivity": BOOTSTRAP_SENSITIVITY_CSV,
        "site_stratified": SITE_STRATIFIED_METRICS_CSV,
    }
    path = mapping.get(table_key)
    if path and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_joint_tcn_v2_metadata() -> Dict[str, Any]:
    """
    Loads verified Joint TCN v2 training metadata, 16-epoch loss history,
    and 144 causality verification checks from joint_tcn_v2_metadata.json.
    """
    if JOINT_TCN_V2_METADATA_PATH.exists():
        with open(JOINT_TCN_V2_METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


def filter_benchmark_predictions(
    df: pd.DataFrame,
    scenario_id: str,
    pipeline_code: Optional[str] = None,
    site: Optional[str] = None,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    population_mode: str = "overall",
) -> pd.DataFrame:
    """
    Slices benchmark predictions safely by scenario, pipeline, site, date, and population mode.
    Population modes:
      - 'overall': all timestamps in selection
      - 'missing_pm': strictly where the nephelometer mask was False (simulated outage)
      - 'common_support': strictly where the nephelometer mask was True (observed sensor)
    """
    if df.empty:
        return df

    filtered = df.copy()

    # Geographical filtering
    if site and site != "All Sites":
        filtered = filtered[filtered["site"] == site]

    # Chronological filtering
    if start_date:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[filtered["timestamp"] <= end_ts]

    # Population filtering based on scenario mask
    mask_col = f"mask_{scenario_id}"
    if mask_col in filtered.columns:
        if population_mode == "missing_pm":
            filtered = filtered[~filtered[mask_col]]
        elif population_mode == "common_support":
            filtered = filtered[filtered[mask_col]]

    return filtered


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes standard regression metrics (RMSE, MAE, R², MBE, Relative Bias %)
    safely handling NaNs and edge cases.
    """
    # Remove pairs with any NaN
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return {
            "n_samples": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mbe": np.nan,
            "rel_bias_pct": np.nan,
        }

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    n = len(yt)

    diff = yp - yt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mbe = float(np.mean(diff))

    mean_yt = float(np.mean(yt))
    rel_bias_pct = float((mbe / mean_yt) * 100.0) if abs(mean_yt) > 1e-6 else np.nan

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((yt - mean_yt) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else np.nan

    return {
        "n_samples": n,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mbe": mbe,
        "rel_bias_pct": rel_bias_pct,
    }

