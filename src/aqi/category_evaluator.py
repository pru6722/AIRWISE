"""
Module: src.aqi.category_evaluator
Comprehensive evaluation of 24-hour PM2.5 predictions, CPCB category classification,
conformal uncertainty intervals, and category stability.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    r2_score,
)

from src.aqi.cpcb_categories import (
    CPCB_CATEGORIES,
    CPCB_CATEGORY_CODES,
    CPCB_CODE_TO_CATEGORY,
    CPCB_BOUNDARY_VALUES,
    classify_pm25,
    classify_pm25_vectorized,
    get_category_code,
    get_distance_to_nearest_boundary,
    get_crossed_boundaries,
    assess_interval_category_stability,
)
from src.uncertainty.conformal import (
    compute_conformal_quantile,
    compute_winkler_interval_score,
    compute_uncertainty_metrics,
)


def compute_24h_point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes regression performance metrics for 24-hour PM2.5 aggregated predictions.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    valid = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[valid]
    y_p = y_p[valid]
    n = len(y_t)

    if n == 0:
        return {
            "n_samples": 0, "mae": np.nan, "rmse": np.nan,
            "r2": np.nan, "mbe": np.nan, "rel_bias_pct": np.nan, "med_ae": np.nan,
        }

    err = y_p - y_t
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mbe = float(np.mean(err))
    med_ae = float(np.median(abs_err))

    mean_ref = float(np.mean(y_t))
    rel_bias_pct = (mbe / mean_ref * 100.0) if mean_ref != 0 else np.nan

    r2 = float(r2_score(y_t, y_p)) if n > 1 else np.nan

    return {
        "n_samples": n,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mbe": round(mbe, 2),
        "rel_bias_pct": round(rel_bias_pct, 2),
        "med_ae": round(med_ae, 2),
    }


def compute_cpcb_classification_metrics(
    y_true_cat: List[str],
    y_pred_cat: List[str],
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive discrete classification metrics for CPCB PM2.5 categories.
    """
    if categories is None:
        categories = CPCB_CATEGORIES

    y_t = list(y_true_cat)
    y_p = list(y_pred_cat)
    n = len(y_t)

    if n == 0:
        return {"n_samples": 0, "accuracy": 0.0}

    # Integer codes
    codes_t = [get_category_code(c) for c in y_t]
    codes_p = [get_category_code(c) for c in y_p]

    accuracy = float(np.mean(np.array(codes_t) == np.array(codes_p)) * 100.0)
    bal_acc = float(balanced_accuracy_score(codes_t, codes_p) * 100.0)
    kappa_unweighted = float(cohen_kappa_score(codes_t, codes_p))
    kappa_linear = float(cohen_kappa_score(codes_t, codes_p, weights="linear"))
    kappa_quad = float(cohen_kappa_score(codes_t, codes_p, weights="quadratic"))

    # Macro and weighted precision, recall, F1
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        codes_t, codes_p, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        codes_t, codes_p, average="weighted", zero_division=0
    )

    # Category-specific precision, recall, F1, support
    p_per_cat, r_per_cat, f1_per_cat, supp_per_cat = precision_recall_fscore_support(
        codes_t, codes_p, labels=list(range(len(categories))), zero_division=0
    )

    # 6x6 Confusion Matrix
    cm = confusion_matrix(codes_t, codes_p, labels=list(range(len(categories))))

    # Adjacent vs Large Errors
    diffs = np.abs(np.array(codes_t) - np.array(codes_p))
    n_exact = int(np.sum(diffs == 0))
    n_adjacent = int(np.sum(diffs == 1))
    n_large = int(np.sum(diffs >= 2))

    pct_exact = (n_exact / n) * 100.0
    pct_adjacent = (n_adjacent / n) * 100.0
    pct_large = (n_large / n) * 100.0

    cat_breakdown = {}
    for idx, cname in enumerate(categories):
        cat_breakdown[cname] = {
            "support": int(supp_per_cat[idx]),
            "precision": round(float(p_per_cat[idx]) * 100.0, 2),
            "recall": round(float(r_per_cat[idx]) * 100.0, 2),
            "f1_score": round(float(f1_per_cat[idx]) * 100.0, 2),
        }

    return {
        "n_samples": n,
        "accuracy": round(accuracy, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "cohen_kappa_unweighted": round(kappa_unweighted, 4),
        "cohen_kappa_linear": round(kappa_linear, 4),
        "cohen_kappa_quadratic": round(kappa_quad, 4),
        "macro_precision": round(float(p_mac) * 100.0, 2),
        "macro_recall": round(float(r_mac) * 100.0, 2),
        "macro_f1": round(float(f1_mac) * 100.0, 2),
        "weighted_f1": round(float(f1_wt) * 100.0, 2),
        "n_exact": n_exact,
        "n_adjacent_error": n_adjacent,
        "n_large_error": n_large,
        "pct_exact": round(pct_exact, 2),
        "pct_adjacent_error": round(pct_adjacent, 2),
        "pct_large_error": round(pct_large, 2),
        "category_metrics": cat_breakdown,
        "confusion_matrix": cm,
    }


def evaluate_uncertainty_aware_stability(
    test_windows_df: pd.DataFrame,
    lower_col: str = "lower_90",
    upper_col: str = "upper_90",
    pred_col: str = "pred_pm25_24h",
    ref_col: str = "ref_pm25_24h",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates category stability and ambiguity across all 24-hour test windows.
    """
    df = test_windows_df.copy()
    stability_list = []

    for _, row in df.iterrows():
        p_val = row[pred_col]
        l_val = row[lower_col]
        u_val = row[upper_col]
        r_val = row[ref_col] if ref_col in row and not pd.isna(row[ref_col]) else None

        res = assess_interval_category_stability(
            point_pred=p_val,
            lower=l_val,
            upper=u_val,
            ref_value=r_val,
        )
        stability_list.append(res)

    stab_df = pd.DataFrame(stability_list)
    for col in stab_df.columns:
        if col not in df.columns:
            df[col] = stab_df[col].values

    n_total = len(df)
    n_stable = int(np.sum(df["is_stable"]))
    n_ambiguous = n_total - n_stable

    pct_stable = (n_stable / n_total * 100.0) if n_total > 0 else 0.0
    pct_ambiguous = (n_ambiguous / n_total * 100.0) if n_total > 0 else 0.0

    # Accuracy in stable vs ambiguous subsets
    stable_sub = df[df["is_stable"]]
    ambig_sub = df[~df["is_stable"]]

    acc_stable = float(np.mean(stable_sub["is_point_correct"]) * 100.0) if len(stable_sub) > 0 else np.nan
    acc_ambig = float(np.mean(ambig_sub["is_point_correct"]) * 100.0) if len(ambig_sub) > 0 else np.nan
    acc_overall = float(np.mean(df["is_point_correct"]) * 100.0) if n_total > 0 else np.nan

    ref_compatible_pct = float(np.mean(df["is_ref_compatible"]) * 100.0) if n_total > 0 else np.nan

    summary = {
        "n_total": n_total,
        "n_stable": n_stable,
        "pct_stable": round(pct_stable, 2),
        "n_ambiguous": n_ambiguous,
        "pct_ambiguous": round(pct_ambiguous, 2),
        "point_accuracy_overall": round(acc_overall, 2),
        "point_accuracy_stable": round(acc_stable, 2),
        "point_accuracy_ambiguous": round(acc_ambig, 2),
        "ref_category_compatibility_pct": round(ref_compatible_pct, 2),
    }

    return df, summary
