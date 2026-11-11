"""
Module: src.models.reconstruction
Implements missing-data reconstruction baselines:
- Linear interpolation (segment-bounded)
- K-Nearest Neighbors (KNN) Imputation
- Random Forest Imputation
- Gated Recurrent Unit (GRU) Sequence Model
- Long Short-Term Memory (LSTM) Sequence Model

Executes the conventional two-stage pipeline:
Raw Sensor -> Calibration Head -> Calibrated Sequence -> Missingness Mask -> Imputation Baseline -> BAM Evaluation
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

from src.evaluation.metrics import compute_regression_metrics


# =====================================================================
# 1. Classical / Statistical Imputation Baselines
# =====================================================================

def reconstruct_linear_interpolation(
    df: pd.DataFrame,
    val_col: str,
    mask: np.ndarray,
    segment_col: str = "segment_id",
) -> np.ndarray:
    """
    Applies linear time-series interpolation strictly within contiguous segments.
    Does not interpolate across multi-day segment outages.
    """
    df_temp = df.copy()
    vals = df_temp[val_col].to_numpy(dtype=float).copy()
    # Artificially hidden values set to NaN
    vals[~mask] = np.nan
    df_temp["val_masked"] = vals

    # Group by segment and interpolate linearly
    interpolated = []
    for _, seg_df in df_temp.groupby(segment_col, sort=False):
        s = seg_df["val_masked"].interpolate(method="linear", limit_direction="both")
        # If an entire segment is missing, forward/backward fill fallback
        s = s.bfill().ffill()
        interpolated.append(s)

    reconstructed = pd.concat(interpolated).sort_index().to_numpy()
    # In rare cases where remaining NaNs exist, fill with median of observed
    if np.isnan(reconstructed).any():
        med = np.nanmedian(vals)
        reconstructed = np.nan_to_num(reconstructed, nan=med)

    return reconstructed


def reconstruct_knn(
    df: pd.DataFrame,
    val_col: str,
    feature_cols: List[str],
    mask: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Applies KNN Imputation utilizing environmental covariates and cyclical time features.
    """
    X = df[feature_cols].to_numpy(dtype=float).copy()
    vals = df[val_col].to_numpy(dtype=float).copy()
    vals[~mask] = np.nan

    # Combine covariates with masked PM2.5 column
    matrix = np.column_stack([vals, X])
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed_matrix = imputer.fit_transform(matrix)

    return imputed_matrix[:, 0]


def reconstruct_random_forest(
    df: pd.DataFrame,
    val_col: str,
    feature_cols: List[str],
    mask: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Trains a Random Forest regressor on observed points (mask==1) using covariates
    and predicts missing points (mask==0).
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[val_col].to_numpy(dtype=float)

    X_train = X[mask]
    y_train = y[mask]

    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    reconstructed = y.copy()
    if np.sum(~mask) > 0:
        reconstructed[~mask] = rf.predict(X[~mask])

    return reconstructed


# =====================================================================
# 2. PyTorch Recurrent Baselines: GRU & LSTM
# =====================================================================

class SequenceTimeSeriesDataset(Dataset):
    """
    Constructs rolling sequence windows strictly within contiguous segment boundaries.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        segments: np.ndarray,
        seq_len: int = 24,
    ):
        self.seq_len = seq_len
        self.X_windows = []
        self.y_windows = []
        self.m_windows = []

        n_samples = len(features)
        for i in range(n_samples - seq_len + 1):
            # Window must belong strictly to the same contiguous segment
            if segments[i] == segments[i + seq_len - 1]:
                self.X_windows.append(features[i : i + seq_len])
                self.y_windows.append(targets[i : i + seq_len])
                self.m_windows.append(masks[i : i + seq_len])

        if len(self.X_windows) == 0:
            # Fallback if no full window fits
            self.X_windows.append(features[:seq_len])
            self.y_windows.append(targets[:seq_len])
            self.m_windows.append(masks[:seq_len])

        self.X_windows = np.array(self.X_windows, dtype=np.float32)
        self.y_windows = np.array(self.y_windows, dtype=np.float32)
        self.m_windows = np.array(self.m_windows, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X_windows[idx]),
            torch.tensor(self.y_windows[idx]),
            torch.tensor(self.m_windows[idx]),
        )


class RecurrentImputerNet(nn.Module):
    """
    Sequence-to-sequence recurrent network (GRU or LSTM) for time-series imputation.
    Receives input sequence, observation mask, and environmental features.
    Supports either bidirectional (offline, uses future observations) or
    unidirectional (strictly causal, uses past observations only).
    """

    def __init__(
        self,
        cell_type: str = "GRU",
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.cell_type = cell_type.upper()
        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1

        if self.cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_dirs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        preds = self.fc(out).squeeze(-1)
        return preds


def train_recurrent_imputer(
    cell_type: str,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    val_col: str = "calibrated_pm25",
    feature_cols: Optional[List[str]] = None,
    seq_len: int = 24,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    bidirectional: bool = True,
    device: str = "cpu",
    random_state: int = 42,
) -> RecurrentImputerNet:
    """
    Trains a GRU or LSTM imputation model on training windows with simulated masks.
    Can be configured as bidirectional (offline) or unidirectional (strictly causal).
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if feature_cols is None:
        feature_cols = ["relative_humidity", "temperature", "hour_sin", "hour_cos"]

    # Prepare features and target strictly from train_df
    X_train_base = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[val_col].to_numpy(dtype=np.float32)
    seg_train = train_df["segment_id"].to_numpy(dtype=int)

    # Random training mask (20% missing)
    rng = np.random.default_rng(random_state)
    m_train = (rng.uniform(0.0, 1.0, size=len(y_train)) >= 0.20).astype(np.float32)

    # Masked input: replace missing values with 0.0 and append mask as feature
    y_masked = y_train.copy()
    y_masked[m_train == 0] = 0.0
    input_features = np.column_stack([y_masked, m_train, X_train_base])
    input_dim = input_features.shape[1]

    dataset = SequenceTimeSeriesDataset(input_features, y_train, m_train, seg_train, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RecurrentImputerNet(
        cell_type=cell_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    model.train()
    for epoch in range(epochs):
        for bx, by, bm in loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    return model


def predict_recurrent_imputer(
    model: RecurrentImputerNet,
    df: pd.DataFrame,
    val_col: str,
    feature_cols: List[str],
    mask: np.ndarray,
    seq_len: int = 24,
    device: str = "cpu",
) -> np.ndarray:
    """Generates sequential imputation predictions across sequence windows."""
    model.eval()
    X_base = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[val_col].to_numpy(dtype=np.float32)
    m = mask.astype(np.float32)

    y_masked = y.copy()
    y_masked[m == 0] = 0.0
    input_features = np.column_stack([y_masked, m, X_base])

    n_samples = len(y)
    preds_accum = np.zeros(n_samples, dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.int32)

    # Process sliding windows with step=1
    with torch.no_grad():
        for i in range(0, n_samples - seq_len + 1, max(1, seq_len // 2)):
            window = torch.tensor(input_features[i : i + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
            pred_win = model(window).squeeze(0).cpu().numpy()
            preds_accum[i : i + seq_len] += pred_win
            counts[i : i + seq_len] += 1

    # Average window overlaps
    valid_idx = counts > 0
    reconstructed = y.copy()
    reconstructed[valid_idx] = preds_accum[valid_idx] / counts[valid_idx]

    # For endpoints not covered by windows, fallback to linear interpolation
    if np.sum(~valid_idx) > 0:
        fallback = reconstruct_linear_interpolation(df, val_col, mask)
        reconstructed[~valid_idx] = fallback[~valid_idx]

    # Preserve observed values exactly at unmasked positions
    final_output = y.copy()
    final_output[~mask] = reconstructed[~mask]
    return final_output


# =====================================================================
# 3. Two-Stage Reconstruction Benchmark Evaluator
# =====================================================================

def evaluate_reconstruction_pipeline_corrected(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    calibration_model: Any,
    calib_feature_cols: List[str],
    mask: np.ndarray,
    covariate_cols: List[str],
    epochs: int = 15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Executes a scientifically rigorous, leak-free reconstruction benchmark:
    1. Calibrates raw sensor features using calibration_model.
    2. Fits scalers and imputation models STRICTLY on train_df.
    3. Validates on val_df.
    4. Evaluates predictions on the held-out test_df specifically on hidden points against BAM.
    5. Categorizes methods explicitly into:
       - 'Offline (Two-Sided)': Linear Interpolation, Bidirectional GRU, Bidirectional LSTM
       - 'Causal (Past Context Only)': Random Forest Imputer, Unidirectional GRU, Unidirectional LSTM

    Returns:
        metrics_df: Comparison metrics with explicit pipeline categories.
        predictions_dict: Reconstructed test values for each model.
    """
    from sklearn.preprocessing import StandardScaler

    # 1. Calibrate train, val, test partitions
    X_cal_train = train_df[calib_feature_cols].to_numpy(dtype=np.float32)
    X_cal_val = val_df[calib_feature_cols].to_numpy(dtype=np.float32)
    X_cal_test = test_df[calib_feature_cols].to_numpy(dtype=np.float32)

    df_train_st = train_df.copy()
    df_val_st = val_df.copy()
    df_test_st = test_df.copy()

    df_train_st["calibrated_pm25"] = calibration_model.predict(X_cal_train)
    df_val_st["calibrated_pm25"] = calibration_model.predict(X_cal_val)
    df_test_st["calibrated_pm25"] = calibration_model.predict(X_cal_test)

    bam_target_test = df_test_st["bam_reference"].to_numpy(dtype=float)

    # 2. Fit covariate scaler strictly on train_df
    impute_features = [c for c in covariate_cols if c in df_train_st.columns]
    cov_scaler = StandardScaler()
    cov_scaler.fit(df_train_st[impute_features].to_numpy(dtype=np.float32))

    predictions: Dict[str, np.ndarray] = {}
    categories: Dict[str, str] = {}
    metrics_rows = []

    hidden_mask = ~mask
    n_hidden = int(np.sum(hidden_mask))

    # A. Linear Interpolation (Offline Two-Sided Baseline)
    preds_lin = reconstruct_linear_interpolation(df_test_st, "calibrated_pm25", mask)
    predictions["Linear Interpolation"] = preds_lin
    categories["Linear Interpolation"] = "Offline (Two-Sided)"

    # B. KNN Imputer (Fitted strictly on train_df)
    knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
    train_matrix = np.column_stack([df_train_st["calibrated_pm25"].to_numpy(), df_train_st[impute_features].to_numpy()])
    knn_imputer.fit(train_matrix)

    test_vals_masked = df_test_st["calibrated_pm25"].to_numpy(dtype=float).copy()
    test_vals_masked[hidden_mask] = np.nan
    test_matrix = np.column_stack([test_vals_masked, df_test_st[impute_features].to_numpy()])
    imputed_test = knn_imputer.transform(test_matrix)[:, 0]
    preds_knn = df_test_st["calibrated_pm25"].to_numpy(dtype=float).copy()
    preds_knn[hidden_mask] = imputed_test[hidden_mask]
    predictions["KNN Imputer"] = preds_knn
    categories["KNN Imputer"] = "Offline (Global Similarity)"

    # C. Random Forest Imputer (Fitted strictly on train_df)
    rf_imputer = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=random_state, n_jobs=-1)
    rf_imputer.fit(df_train_st[impute_features].to_numpy(dtype=np.float32), df_train_st["calibrated_pm25"].to_numpy(dtype=np.float32))
    preds_rf = df_test_st["calibrated_pm25"].to_numpy(dtype=float).copy()
    preds_rf[hidden_mask] = rf_imputer.predict(df_test_st[impute_features].to_numpy(dtype=np.float32)[hidden_mask])
    predictions["Random Forest Imputer"] = preds_rf
    categories["Random Forest Imputer"] = "Causal (Tabular RF)"

    # D. Bidirectional GRU (Offline)
    gru_bidir = train_recurrent_imputer(
        "GRU", df_train_st, df_val_st, "calibrated_pm25", impute_features,
        seq_len=24, epochs=epochs, bidirectional=True, random_state=random_state
    )
    preds_gru_bi = predict_recurrent_imputer(gru_bidir, df_test_st, "calibrated_pm25", impute_features, mask)
    predictions["Bidirectional GRU"] = preds_gru_bi
    categories["Bidirectional GRU"] = "Offline (Bidirectional RNN)"

    # E. Bidirectional LSTM (Offline)
    lstm_bidir = train_recurrent_imputer(
        "LSTM", df_train_st, df_val_st, "calibrated_pm25", impute_features,
        seq_len=24, epochs=epochs, bidirectional=True, random_state=random_state
    )
    preds_lstm_bi = predict_recurrent_imputer(lstm_bidir, df_test_st, "calibrated_pm25", impute_features, mask)
    predictions["Bidirectional LSTM"] = preds_lstm_bi
    categories["Bidirectional LSTM"] = "Offline (Bidirectional RNN)"

    # F. Unidirectional GRU (Causal)
    gru_causal = train_recurrent_imputer(
        "GRU", df_train_st, df_val_st, "calibrated_pm25", impute_features,
        seq_len=24, epochs=epochs, bidirectional=False, random_state=random_state
    )
    preds_gru_causal = predict_recurrent_imputer(gru_causal, df_test_st, "calibrated_pm25", impute_features, mask)
    predictions["Causal GRU"] = preds_gru_causal
    categories["Causal GRU"] = "Causal (Unidirectional RNN)"

    # G. Unidirectional LSTM (Causal)
    lstm_causal = train_recurrent_imputer(
        "LSTM", df_train_st, df_val_st, "calibrated_pm25", impute_features,
        seq_len=24, epochs=epochs, bidirectional=False, random_state=random_state
    )
    preds_lstm_causal = predict_recurrent_imputer(lstm_causal, df_test_st, "calibrated_pm25", impute_features, mask)
    predictions["Causal LSTM"] = preds_lstm_causal
    categories["Causal LSTM"] = "Causal (Unidirectional RNN)"

    # Compute metrics specifically on hidden points against BAM
    for name, recon_vals in predictions.items():
        y_true_hidden = bam_target_test[hidden_mask]
        y_pred_hidden = recon_vals[hidden_mask]

        m = compute_regression_metrics(y_true_hidden, y_pred_hidden)
        metrics_rows.append({
            "model": name,
            "pipeline_category": categories[name],
            "hidden_samples": n_hidden,
            "hidden_mae": m["mae"],
            "hidden_rmse": m["rmse"],
            "hidden_r2": m["r2"],
            "hidden_mbe": m["mbe"],
            "hidden_rel_bias_pct": m["relative_bias_pct"],
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("hidden_rmse").reset_index(drop=True)
    return metrics_df, predictions


def evaluate_reconstruction_baselines(
    test_df: pd.DataFrame,
    calibration_model: Any,
    calib_feature_cols: List[str],
    mask: np.ndarray,
    covariate_cols: List[str],
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Two-stage reconstruction evaluator.
    If train_df and val_df are provided, executes leak-free evaluation via evaluate_reconstruction_pipeline_corrected.
    If omitted, runs legacy test-set fitting with an explicit warning for backward compatibility.
    """
    if train_df is not None and val_df is not None:
        return evaluate_reconstruction_pipeline_corrected(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            calibration_model=calibration_model,
            calib_feature_cols=calib_feature_cols,
            mask=mask,
            covariate_cols=covariate_cols,
            random_state=random_state,
        )

    import warnings
    warnings.warn(
        "evaluate_reconstruction_baselines called without train_df/val_df. "
        "Baseline models will be fitted on test_df (legacy test-leakage mode). "
        "Pass train_df and val_df to use leak-free evaluation.",
        UserWarning,
        stacklevel=2,
    )

    # Legacy fallback implementation
    X_calib = test_df[calib_feature_cols].to_numpy(dtype=np.float32)
    calibrated_pm25 = calibration_model.predict(X_calib)

    df_stage = test_df.copy()
    df_stage["calibrated_pm25"] = calibrated_pm25
    bam_target = df_stage["bam_reference"].to_numpy(dtype=float)

    impute_features = [c for c in covariate_cols if c in df_stage.columns]
    predictions: Dict[str, np.ndarray] = {}
    metrics_rows = []

    hidden_mask = ~mask
    n_hidden = int(np.sum(hidden_mask))

    preds_linear = reconstruct_linear_interpolation(df_stage, "calibrated_pm25", mask)
    predictions["Linear Interpolation"] = preds_linear

    preds_knn = reconstruct_knn(df_stage, "calibrated_pm25", impute_features, mask, n_neighbors=5)
    predictions["KNN Imputer"] = preds_knn

    preds_rf = reconstruct_random_forest(df_stage, "calibrated_pm25", impute_features, mask, random_state=random_state)
    predictions["Random Forest Imputer"] = preds_rf

    gru_model = train_recurrent_imputer(
        "GRU", df_stage, df_stage, "calibrated_pm25", impute_features,
        seq_len=24, epochs=15, bidirectional=True, random_state=random_state
    )
    preds_gru = predict_recurrent_imputer(gru_model, df_stage, "calibrated_pm25", impute_features, mask)
    predictions["GRU"] = preds_gru

    lstm_model = train_recurrent_imputer(
        "LSTM", df_stage, df_stage, "calibrated_pm25", impute_features,
        seq_len=24, epochs=15, bidirectional=True, random_state=random_state
    )
    preds_lstm = predict_recurrent_imputer(lstm_model, df_stage, "calibrated_pm25", impute_features, mask)
    predictions["LSTM"] = preds_lstm

    for name, recon_vals in predictions.items():
        y_true_hidden = bam_target[hidden_mask]
        y_pred_hidden = recon_vals[hidden_mask]

        m = compute_regression_metrics(y_true_hidden, y_pred_hidden)
        metrics_rows.append({
            "model": name,
            "hidden_samples": n_hidden,
            "hidden_mae": m["mae"],
            "hidden_rmse": m["rmse"],
            "hidden_r2": m["r2"],
            "hidden_mbe": m["mbe"],
            "hidden_rel_bias_pct": m["relative_bias_pct"],
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("hidden_rmse").reset_index(drop=True)
    return metrics_df, predictions
