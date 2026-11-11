"""
Module: src.models.calibration
Implements Linear Regression, Random Forest, and XGBoost calibration models
across single-channel and multi-channel feature configurations, with pooled and site-specific training.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import ctypes
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Pre-load libomp for XGBoost on macOS if needed
try:
    if os.path.exists("/opt/homebrew/opt/libomp/lib/libomp.dylib"):
        ctypes.cdll.LoadLibrary("/opt/homebrew/opt/libomp/lib/libomp.dylib")
except Exception:
    pass

from xgboost import XGBRegressor

from src.evaluation.metrics import compute_regression_metrics


FEATURE_SETS = {
    "cf1_only": ["pa_cf1", "relative_humidity", "temperature"],
    "atm_only": ["pa_atm", "relative_humidity", "temperature"],
    "alt_only": ["pa_alt", "relative_humidity", "temperature"],
    "all_channels": ["pa_cf1", "pa_atm", "pa_alt", "relative_humidity", "temperature"],
}


def build_model(model_name: str, random_state: int = 42) -> Any:
    """Instantiates the specified calibration regressor."""
    if model_name == "Linear Regression":
        return LinearRegression(fit_intercept=True)
    elif model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_and_evaluate_calibration_model(
    model_name: str,
    feature_set_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "bam_reference",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains a calibration model on train_df and evaluates on train, val, and test partitions.
    """
    feature_cols = FEATURE_SETS[feature_set_name]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.float32)

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_col].to_numpy(dtype=np.float32)

    model = build_model(model_name, random_state=random_state)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    train_metrics = compute_regression_metrics(y_train, train_preds)
    val_metrics = compute_regression_metrics(y_val, val_preds)
    test_metrics = compute_regression_metrics(y_test, test_preds)

    return {
        "model_name": model_name,
        "feature_set": feature_set_name,
        "feature_cols": feature_cols,
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "predictions": {
            "val": val_preds,
            "test": test_preds,
        },
    }


def run_all_calibration_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_save_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Executes the full factorial calibration benchmark:
    - 3 Algorithms (Linear Regression, Random Forest, XGBoost)
    - 4 Feature subsets (cf1_only, atm_only, alt_only, all_channels)
    - 2 Spatial paradigms (Pooled training, Site-specific training)

    Returns:
        results_df: Tabular evaluation summary across all configurations.
        models_dict: Trained model instances.
        best_model_bundle: Dictionary containing the best performing model and its metadata.
    """
    if models_save_dir is not None:
        models_save_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["Linear Regression", "Random Forest", "XGBoost"]
    feature_sets = ["cf1_only", "atm_only", "alt_only", "all_channels"]

    rows = []
    models_dict = {}

    # 1. Pooled Calibration Experiments
    print("  Executing Pooled Calibration Experiments...")
    for algo in algorithms:
        for fset in feature_sets:
            res = train_and_evaluate_calibration_model(
                algo, fset, train_df, val_df, test_df, random_state=random_state
            )
            key = f"pooled_{algo.replace(' ', '_').lower()}_{fset}"
            models_dict[key] = res["model"]

            if models_save_dir is not None:
                joblib.dump(res["model"], models_save_dir / f"{key}.joblib")

            tm = res["test_metrics"]
            vm = res["val_metrics"]
            trm = res["train_metrics"]
            rows.append({
                "paradigm": "Pooled",
                "site": "All Sites",
                "algorithm": algo,
                "feature_set": fset,
                "train_rmse": trm["rmse"],
                "train_r2": trm["r2"],
                "val_mae": vm["mae"],
                "val_rmse": vm["rmse"],
                "val_r2": vm["r2"],
                "val_mbe": vm["mbe"],
                "test_mae": tm["mae"],
                "test_rmse": tm["rmse"],
                "test_r2": tm["r2"],
                "test_mbe": tm["mbe"],
                "test_rel_bias_pct": tm["relative_bias_pct"],
            })

    # 2. Site-Specific Calibration Experiments
    sites = train_df["site"].unique()
    print(f"  Executing Site-Specific Calibration Experiments for: {list(sites)}...")
    for site in sites:
        t_site = train_df[train_df["site"] == site]
        v_site = val_df[val_df["site"] == site]
        te_site = test_df[test_df["site"] == site]

        for algo in algorithms:
            for fset in feature_sets:
                res = train_and_evaluate_calibration_model(
                    algo, fset, t_site, v_site, te_site, random_state=random_state
                )
                key = f"{site.lower()}_{algo.replace(' ', '_').lower()}_{fset}"
                models_dict[key] = res["model"]

                if models_save_dir is not None:
                    joblib.dump(res["model"], models_save_dir / f"{key}.joblib")

                tm = res["test_metrics"]
                vm = res["val_metrics"]
                trm = res["train_metrics"]
                rows.append({
                    "paradigm": "Site-Specific",
                    "site": site,
                    "algorithm": algo,
                    "feature_set": fset,
                    "train_rmse": trm["rmse"],
                    "train_r2": trm["r2"],
                    "val_mae": vm["mae"],
                    "val_rmse": vm["rmse"],
                    "val_r2": vm["r2"],
                    "val_mbe": vm["mbe"],
                    "test_mae": tm["mae"],
                    "test_rmse": tm["rmse"],
                    "test_r2": tm["r2"],
                    "test_mbe": tm["mbe"],
                    "test_rel_bias_pct": tm["relative_bias_pct"],
                })

    results_df = pd.DataFrame(rows)

    # Select best model based on validation RMSE (to avoid test leakage)
    # Filter for pooled models to use as universal stage-1 calibrator
    pooled_df = results_df[results_df["paradigm"] == "Pooled"]
    best_row = pooled_df.sort_values("val_rmse").iloc[0]
    best_key = f"pooled_{best_row['algorithm'].replace(' ', '_').lower()}_{best_row['feature_set']}"
    best_model = models_dict[best_key]

    best_bundle = {
        "key": best_key,
        "algorithm": best_row["algorithm"],
        "feature_set": best_row["feature_set"],
        "feature_cols": FEATURE_SETS[best_row["feature_set"]],
        "model": best_model,
        "val_rmse": best_row["val_rmse"],
        "test_rmse": best_row["test_rmse"],
        "test_mae": best_row["test_mae"],
        "test_r2": best_row["test_r2"],
    }

    return results_df, models_dict, best_bundle
