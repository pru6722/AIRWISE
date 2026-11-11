"""
AIRWISE Dashboard: Joint TCN Deep Learning Service
Loads serialized PyTorch state dictionaries and normalizers, prepares sliding causal window tensors,
and executes real-time CPU sequence inference without lookahead leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.models.joint_tcn import JointTCN
from src.data.feature_engineering import add_temporal_cyclical_features
from src.training.train_joint_tcn import DataNormalizer, INPUT_FEATURE_COLS
from src.uncertainty.split_manager import create_conformal_splits
from src.evaluation.evaluate_joint_tcn import predict_joint_tcn_full_sequence
from dashboard.utils.paths import (
    CLEANED_PARQUET_PATH,
    JOINT_TCN_BEST_PATH,
    JOINT_TCN_FROZEN_PATH,
    JOINT_TCN_V2_CHECKPOINT_PATH,
    JOINT_TCN_NORMALIZER_PATH,
    CONFORMAL_NORMALIZER_PATH,
    JOINT_TCN_MODELS_DIR,
)


@st.cache_resource(show_spinner=False)
def _fit_v2_train_normalizer() -> Optional[DataNormalizer]:
    """Fits the v2 DataNormalizer strictly on the canonical 11,149-row training partition."""
    if not CLEANED_PARQUET_PATH.exists():
        return None
    df = pd.read_parquet(CLEANED_PARQUET_PATH)
    train_df, _, _, _, _ = create_conformal_splits(df)
    if "hour_sin" not in train_df.columns:
        train_df = add_temporal_cyclical_features(train_df)
    feature_cols = [c for c in INPUT_FEATURE_COLS if c in train_df.columns]
    return DataNormalizer(feature_cols).fit(train_df, target_col="bam_reference")


@st.cache_resource(show_spinner=False)
def load_joint_tcn_model(model_variant: str = "v2") -> Tuple[Optional[JointTCN], Optional[Any]]:
    """
    Loads frozen Joint TCN model checkpoint and its corresponding DataNormalizer.
    Variants:
        - 'v2': Phase 3C Causality-Safe Epoch-6 Checkpoint (ChannelLayerNorm1d)
        - 'best': Historical validation checkpoint (BatchNorm)
        - 'frozen': Historical conformal checkpoint (BatchNorm)
        - 'cal1_rec5': Historical ablation checkpoint lambda_cal=1.0, lambda_rec=5.0
        - 'cal1_rec10': Historical ablation checkpoint lambda_cal=1.0, lambda_rec=10.0
        - 'cal1_rec20': Historical ablation checkpoint lambda_cal=1.0, lambda_rec=20.0
    """
    if model_variant == "v2":
        if not JOINT_TCN_V2_CHECKPOINT_PATH.exists():
            return None, None
        normalizer = _fit_v2_train_normalizer()
        if normalizer is None:
            return None, None
        model = JointTCN(
            input_dim=11,
            hidden_channels=32,
            dilations=[1, 2, 4, 8],
            kernel_size=3,
            dropout=0.15,
            norm_type="layer",
        )
        state_dict = torch.load(JOINT_TCN_V2_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model, normalizer

    ckpt_paths = {
        "best": (JOINT_TCN_BEST_PATH, JOINT_TCN_NORMALIZER_PATH),
        "frozen": (JOINT_TCN_FROZEN_PATH, CONFORMAL_NORMALIZER_PATH),
        "cal1_rec5": (JOINT_TCN_MODELS_DIR / "joint_tcn_cal1_rec5.pt", JOINT_TCN_NORMALIZER_PATH),
        "cal1_rec10": (JOINT_TCN_MODELS_DIR / "joint_tcn_cal1_rec10.pt", JOINT_TCN_NORMALIZER_PATH),
        "cal1_rec20": (JOINT_TCN_MODELS_DIR / "joint_tcn_cal1_rec20.pt", JOINT_TCN_NORMALIZER_PATH),
    }

    ckpt_path, norm_path = ckpt_paths.get(model_variant, (JOINT_TCN_BEST_PATH, JOINT_TCN_NORMALIZER_PATH))
    if not ckpt_path.exists() or not norm_path.exists():
        return None, None

    # Instantiate architecture matching historical Checkpoint 3 specifications
    model = JointTCN(
        input_dim=11,
        hidden_channels=32,
        dilations=[1, 2, 4, 8],
        kernel_size=3,
        dropout=0.15,
        norm_type="batch",
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    normalizer = joblib.load(norm_path)
    return model, normalizer


def run_sequence_inference(
    df_window: pd.DataFrame,
    mask: np.ndarray,
    model_variant: str = "v2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes true causal sequence inference on a DataFrame window using the Joint TCN.
    Ensures input features include cyclical hour and day-of-year features.
    
    Returns:
        Tuple of (y_cal, y_rec, y_final) in original BAM µg/m³ scale.
    """
    model, normalizer = load_joint_tcn_model(model_variant)
    if model is None or normalizer is None:
        raise FileNotFoundError("Joint TCN model checkpoint or normalizer artifact missing.")

    # Prepare stripped copy guaranteeing bam_reference never enters feature extraction
    df_prepared = df_window.copy()
    if "hour_sin" not in df_prepared.columns:
        df_prepared = add_temporal_cyclical_features(df_prepared)
    # Normalizer.transform requires target_col column presence but output y_norm is discarded;
    # overwrite with constant 0.0 so test BAM ground truth cannot reach inference memory.
    df_prepared["bam_reference"] = 0.0

    y_cal, y_rec, y_final = predict_joint_tcn_full_sequence(
        model=model,
        normalizer=normalizer,
        df=df_prepared,
        mask=mask,
        target_col="bam_reference",
        use_mask_feature=True,
        device="cpu",
    )

    return y_cal, y_rec, y_final
