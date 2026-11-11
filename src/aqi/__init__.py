"""
Package: src.aqi
Modules for CPCB PM2.5 air quality category assessment, 24-hour windowing,
and uncertainty-aware category stability analysis.
"""

from src.aqi.cpcb_categories import (
    CPCB_CATEGORIES,
    CPCB_BREAKPOINTS,
    CPCB_BOUNDARY_VALUES,
    CPCB_CATEGORY_RANGES,
    classify_pm25,
    classify_pm25_vectorized,
    get_category_code,
    get_category_from_code,
    get_distance_to_nearest_boundary,
    get_crossed_boundaries,
    assess_interval_category_stability,
)

__all__ = [
    "CPCB_CATEGORIES",
    "CPCB_BREAKPOINTS",
    "CPCB_BOUNDARY_VALUES",
    "CPCB_CATEGORY_RANGES",
    "classify_pm25",
    "classify_pm25_vectorized",
    "get_category_code",
    "get_category_from_code",
    "get_distance_to_nearest_boundary",
    "get_crossed_boundaries",
    "assess_interval_category_stability",
]
