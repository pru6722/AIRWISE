"""
Module: src.aqi.cpcb_categories
Authoritative implementation of CPCB PM2.5 Air Quality Category Breakpoints
and Uncertainty-Aware Category Stability Assessment.

Authoritative Source Documentation:
- Issuing Authority: Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India.
- Document Title: National Air Quality Index (NAQI)
- Publication Date: October 2014
- Document Reference: CPCB Report Control of Urban Pollution Series (CUPS/82/2014-15)
- Official URL: https://cpcb.nic.in/displaypdf.php?id=bmFxaS9uYXFpX3JlcG9ydC5wZGY=
- Accessed: November 2026

Official 24-Hour PM2.5 Breakpoints (continuous concentration boundaries):
- Good:          [0, 30] µg/m³
- Satisfactory:  (30, 60] µg/m³
- Moderate:      (60, 90] µg/m³
- Poor:          (90, 120] µg/m³
- Very Poor:     (120, 250] µg/m³
- Severe:        (250, inf) µg/m³
"""

from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd


CPCB_CATEGORIES: List[str] = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]

CPCB_CATEGORY_CODES: Dict[str, int] = {
    cat: idx for idx, cat in enumerate(CPCB_CATEGORIES)
}

CPCB_CODE_TO_CATEGORY: Dict[int, str] = {
    idx: cat for idx, cat in enumerate(CPCB_CATEGORIES)
}

# Continuous boundary thresholds between categories
CPCB_BOUNDARY_VALUES: List[float] = [30.0, 60.0, 90.0, 120.0, 250.0]

CPCB_BREAKPOINTS: List[float] = [0.0, 30.0, 60.0, 90.0, 120.0, 250.0]

CPCB_CATEGORY_RANGES: Dict[str, Tuple[float, float]] = {
    "Good": (0.0, 30.0),
    "Satisfactory": (30.0, 60.0),
    "Moderate": (60.0, 90.0),
    "Poor": (90.0, 120.0),
    "Very Poor": (120.0, 250.0),
    "Severe": (250.0, float("inf")),
}

CPCB_CATEGORY_COLORS: Dict[str, str] = {
    "Good": "#009966",          # Green
    "Satisfactory": "#99CC33",  # Light Green / Chartreuse
    "Moderate": "#FFDE33",      # Yellow
    "Poor": "#FF9933",          # Orange
    "Very Poor": "#CC0033",     # Red
    "Severe": "#7E0023",        # Maroon / Dark Red
}


def classify_pm25(value: Union[float, int]) -> str:
    """
    Classifies a continuous 24-hour PM2.5 mean concentration into official CPCB categories.

    Continuous boundary conventions:
    - value <= 30.0: Good
    - 30.0 < value <= 60.0: Satisfactory
    - 60.0 < value <= 90.0: Moderate
    - 90.0 < value <= 120.0: Poor
    - 120.0 < value <= 250.0: Very Poor
    - value > 250.0: Severe

    Args:
        value: Scalar PM2.5 concentration in ug/m3.

    Returns:
        category: String name of the CPCB category, or 'Unknown' if NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"

    v = float(value)
    # Handle physical floor: ambient PM2.5 cannot be strictly negative
    if v <= 30.0:
        return "Good"
    elif v <= 60.0:
        return "Satisfactory"
    elif v <= 90.0:
        return "Moderate"
    elif v <= 120.0:
        return "Poor"
    elif v <= 250.0:
        return "Very Poor"
    else:
        return "Severe"


def classify_pm25_vectorized(values: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Vectorized classification of PM2.5 concentration values into CPCB categories.
    """
    arr = np.asarray(values, dtype=float)
    categories = np.empty(len(arr), dtype=object)

    nan_mask = np.isnan(arr)
    categories[nan_mask] = "Unknown"

    categories[~nan_mask & (arr <= 30.0)] = "Good"
    categories[~nan_mask & (arr > 30.0) & (arr <= 60.0)] = "Satisfactory"
    categories[~nan_mask & (arr > 60.0) & (arr <= 90.0)] = "Moderate"
    categories[~nan_mask & (arr > 90.0) & (arr <= 120.0)] = "Poor"
    categories[~nan_mask & (arr > 120.0) & (arr <= 250.0)] = "Very Poor"
    categories[~nan_mask & (arr > 250.0)] = "Severe"

    return categories


def get_category_code(category_name: str) -> int:
    """Returns integer code (0 to 5) for CPCB category, or -1 if unknown."""
    return CPCB_CATEGORY_CODES.get(category_name, -1)


def get_category_from_code(code: int) -> str:
    """Returns CPCB category name from integer code (0 to 5)."""
    return CPCB_CODE_TO_CATEGORY.get(code, "Unknown")


def get_distance_to_nearest_boundary(value: float) -> Tuple[float, float]:
    """
    Computes absolute distance from a PM2.5 value to the nearest CPCB category boundary
    among [30.0, 60.0, 90.0, 120.0, 250.0].

    Returns:
        min_distance: Absolute difference |value - boundary| in ug/m3.
        nearest_boundary: The boundary value closest to the input.
    """
    if value is None or np.isnan(value):
        return float("nan"), float("nan")

    v = float(value)
    distances = [abs(v - b) for b in CPCB_BOUNDARY_VALUES]
    min_idx = int(np.argmin(distances))
    return float(distances[min_idx]), float(CPCB_BOUNDARY_VALUES[min_idx])


def get_crossed_boundaries(lower: float, upper: float) -> List[float]:
    """
    Identifies all CPCB category boundaries strictly crossed by the prediction interval [lower, upper].
    A boundary b is crossed if lower < b < upper.
    """
    if lower is None or upper is None or np.isnan(lower) or np.isnan(upper):
        return []

    l = float(lower)
    u = float(upper)
    if l > u:
        l, u = u, l

    crossed = [b for b in CPCB_BOUNDARY_VALUES if l < b < u]
    return crossed


def assess_interval_category_stability(
    point_pred: float,
    lower: float,
    upper: float,
    ref_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Assesses uncertainty-aware category stability for a single 24-hour prediction interval.

    Definitions:
    - Stable Category: The entire interval [lower, upper] lies within exactly ONE CPCB category.
      (i.e., classify_pm25(lower) == classify_pm25(upper))
    - Ambiguous Category: The interval crosses at least one category boundary.
    - Compatible Category: The reference ground-truth category falls within the span
      of categories covered by [lower, upper].

    Args:
        point_pred: 24h point prediction mean (ug/m3)
        lower: Conformal lower bound (ug/m3)
        upper: Conformal upper bound (ug/m3)
        ref_value: Optional reference 24h BAM mean (ug/m3)

    Returns:
        Dictionary with full stability metadata.
    """
    # Enforce non-negative physical floor on bounds for category classification
    l_clean = max(0.0, float(lower))
    u_clean = max(0.0, float(upper))
    p_clean = max(0.0, float(point_pred))

    cat_point = classify_pm25(p_clean)
    cat_lower = classify_pm25(l_clean)
    cat_upper = classify_pm25(u_clean)

    crossed_bounds = get_crossed_boundaries(l_clean, u_clean)
    num_crossed = len(crossed_bounds)
    is_stable = (cat_lower == cat_upper)

    code_lower = get_category_code(cat_lower)
    code_upper = get_category_code(cat_upper)

    # All categories covered by the interval
    if code_lower != -1 and code_upper != -1:
        min_code = min(code_lower, code_upper)
        max_code = max(code_lower, code_upper)
        covered_categories = [CPCB_CODE_TO_CATEGORY[c] for c in range(min_code, max_code + 1)]
    else:
        covered_categories = [cat_point]

    result: Dict[str, Any] = {
        "point_prediction": p_clean,
        "lower_bound": l_clean,
        "upper_bound": u_clean,
        "interval_width": u_clean - l_clean,
        "cat_point": cat_point,
        "cat_lower": cat_lower,
        "cat_upper": cat_upper,
        "is_stable": is_stable,
        "stability_label": "Stable" if is_stable else "Ambiguous",
        "num_crossed_boundaries": num_crossed,
        "crossed_boundaries": crossed_bounds,
        "covered_categories": covered_categories,
        "num_covered_categories": len(covered_categories),
    }

    if ref_value is not None and not np.isnan(ref_value):
        r_clean = max(0.0, float(ref_value))
        cat_ref = classify_pm25(r_clean)
        is_point_correct = (cat_point == cat_ref)
        is_ref_compatible = (cat_ref in covered_categories)

        result["ref_value"] = r_clean
        result["cat_ref"] = cat_ref
        result["is_point_correct"] = is_point_correct
        result["is_ref_compatible"] = is_ref_compatible
        result["category_error_distance"] = abs(get_category_code(cat_point) - get_category_code(cat_ref))
    else:
        result["ref_value"] = None
        result["cat_ref"] = "Unknown"
        result["is_point_correct"] = None
        result["is_ref_compatible"] = None
        result["category_error_distance"] = None

    return result
