"""
AIRWISE Dashboard: Calibration Service
Loads serialized Scikit-learn and XGBoost models and executes live CPU inference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import ctypes
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Preload libomp on macOS ARM64 for XGBoost if present
try:
    if os.path.exists("/opt/homebrew/opt/libomp/lib/libomp.dylib"):
        ctypes.cdll.LoadLibrary("/opt/homebrew/opt/libomp/lib/libomp.dylib")
except Exception:
    pass

from dashboard.utils.paths import CALIBRATION_MODELS_DIR
from dashboard.config import FEATURE_SETS


ALGORITHMS = ["Random Forest", "XGBoost", "Linear Regression"]
PARADIGMS = ["Pooled", "Delhi", "Hamirpur", "Bangalore"]
FEATURE_SET_NAMES = ["all_channels", "atm_only", "cf1_only", "alt_only"]


def build_model_filename(paradigm: str, algorithm: str, feature_set: str) -> str:
    """Constructs the standard serialized filename for a calibration model."""
    prefix = paradigm.lower()
    algo_slug = algorithm.replace(" ", "_").lower()
    return f"{prefix}_{algo_slug}_{feature_set}.joblib"


@st.cache_resource(show_spinner=False)
def load_calibration_model(paradigm: str, algorithm: str, feature_set: str) -> Optional[Any]:
    """Loads a specific calibration model checkpoint with Streamlit caching."""
    filename = build_model_filename(paradigm, algorithm, feature_set)
    filepath = CALIBRATION_MODELS_DIR / filename
    if filepath.exists():
        return joblib.load(filepath)
    return None


def run_live_calibration_inference(
    paradigm: str,
    algorithm: str,
    feature_set: str,
    pa_cf1: float,
    pa_atm: float,
    pa_alt: float,
    relative_humidity: float,
    temperature: float,
) -> Tuple[Optional[float], List[str]]:
    """
    Executes actual live inference on CPU using the chosen trained model checkpoint.
    Returns:
        calibrated_pm25: Predicted reference concentration in µg/m³
        feature_order: List of features fed into the model
    """
    model = load_calibration_model(paradigm, algorithm, feature_set)
    if model is None:
        return None, []

    feature_cols = FEATURE_SETS[feature_set]
    input_dict = {
        "pa_cf1": pa_cf1,
        "pa_atm": pa_atm,
        "pa_alt": pa_alt,
        "relative_humidity": relative_humidity,
        "temperature": temperature,
    }

    feature_vector = [input_dict[col] for col in feature_cols]
    X = np.array([feature_vector], dtype=np.float32)

    pred = model.predict(X)
    return float(pred[0]), feature_cols
