"""
AIRWISE Dashboard: CPCB Category Reliability Service
Provides official CPCB regulatory PM2.5 classification, distance-to-boundary calculations,
and uncertainty-aware category stability diagnostics.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.aqi.cpcb_categories import (
    classify_pm25,
    classify_pm25_vectorized,
    get_distance_to_nearest_boundary,
    get_crossed_boundaries,
    assess_interval_category_stability,
    CPCB_CATEGORIES,
    CPCB_BOUNDARY_VALUES,
    CPCB_BREAKPOINTS,
)
from src.aqi.windowing import construct_24h_windows


def evaluate_cpcb_status(
    point_pred: float,
    lower: float,
    upper: float,
    ref_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates regulatory CPCB category and stability for a predicted PM2.5 value.
    """
    dist, nearest_b = get_distance_to_nearest_boundary(point_pred)
    stability = assess_interval_category_stability(
        point_pred=point_pred,
        lower=lower,
        upper=upper,
        ref_value=ref_value,
    )
    stability["distance_to_nearest_boundary"] = dist
    stability["nearest_boundary_value"] = nearest_b
    return stability


def compute_daily_aggregates_from_hourly(
    df_hourly: pd.DataFrame,
    min_hours: int = 18,
) -> pd.DataFrame:
    """
    Applies the regulatory 75% completeness rule (>=18 valid hours out of 24)
    to aggregate hourly time series into daily 24h compliance windows.
    """
    windows_df, audit_summary = construct_24h_windows(
        df=df_hourly,
        min_valid_hours=min_hours,
        non_overlapping=True,
        timestamp_col="timestamp",
        site_col="site",
        target_col="bam_reference",
        pred_col="point_prediction" if "point_prediction" in df_hourly.columns else "pa_atm",
    )
    return windows_df
