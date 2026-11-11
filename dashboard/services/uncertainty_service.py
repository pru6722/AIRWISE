"""
AIRWISE Dashboard: Uncertainty Service
Manages conformal prediction quantiles, symmetric prediction intervals,
and prediction-time regime conditioning without reference target leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import CONFORMAL_QUANTILES_PATH
from src.uncertainty.regime_conformal import (
    assign_humidity_regime,
    assign_predicted_pollution_regime,
)


@st.cache_data(show_spinner=False)
def load_conformal_quantiles() -> Dict[str, float]:
    """Loads precomputed finite-sample conformal quantiles."""
    if CONFORMAL_QUANTILES_PATH.exists():
        with open(CONFORMAL_QUANTILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Default fallbacks matching Checkpoint 4 audit findings
    return {
        "cal_head_q90": 24.499,
        "cal_head_q95": 29.159,
        "rec_head_q90": 30.297,
        "rec_head_q95": 36.729,
        "composite_q90": 25.179,
        "composite_q95": 30.828,
        "block_6h_q90": 36.210,
        "block_12h_q90": 44.336,
        "block_12h_q95": 100.422,
        "block_24h_q90": 100.422,
    }


def compute_prediction_interval(
    y_pred: float,
    alpha: float = 0.10,
    head_type: str = "composite",
) -> Tuple[float, float, float]:
    """
    Computes finite-sample symmetric conformal prediction interval.
    Args:
        y_pred: Point prediction in µg/m³
        alpha: Miscoverage level (0.10 for 90%, 0.05 for 95%)
        head_type: 'cal_head', 'rec_head', 'composite', or 'block_12h'
    Returns:
        lower: Lower bound clipped to 0.0 µg/m³
        upper: Upper bound
        quantile: Conformal quantile used
    """
    quantiles = load_conformal_quantiles()
    suffix = "q90" if alpha >= 0.08 else "q95"
    key = f"{head_type}_{suffix}"
    q = quantiles.get(key, 25.18)

    lower = max(0.0, y_pred - q)
    upper = y_pred + q
    return lower, upper, q
