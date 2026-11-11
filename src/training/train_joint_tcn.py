"""
Module: src.training.train_joint_tcn
Implements data loaders, dynamic window masking, early-stopping training routines,
and checkpoint management for the Joint Calibration-and-Reconstruction TCN.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.models.joint_tcn import JointTCN, JointMultiTaskLoss
from src.missingness.random_mask import generate_mcar_mask
from src.missingness.block_mask import generate_block_outage_mask


SENSOR_COLS = [
    "pa_cf1",
    "pa_atm",
    "pa_alt",
]

INPUT_FEATURE_COLS = [
    "pa_cf1",
    "pa_atm",
    "pa_alt",
    "relative_humidity",
    "temperature",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "time_diff_hours",
]


class TCNWindowDataset(Dataset):
    """
    Sliding sequence window dataset for TCN training and evaluation.
    Ensures that windows strictly reside within the same contiguous segment_id
    and never cross partition boundaries.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        segments: np.ndarray,
        seq_len: int = 24,
        is_training: bool = True,
        missing_rate_range: Tuple[float, float] = (0.05, 0.30),
        use_mask_feature: bool = True,
        stride: int = 1,
        random_seed: int = 42,
    ):
        self.features = features  # (N, D)
        self.targets = targets    # (N,)
        self.segments = segments  # (N,)
        self.seq_len = seq_len
        self.is_training = is_training
        self.missing_rate_range = missing_rate_range
        self.use_mask_feature = use_mask_feature
        self.stride = stride
        self.rng = np.random.default_rng(random_seed)

        # Precompute valid window start indices strictly within the same segment
        self.valid_starts = []
        n_samples = len(features)
        for i in range(0, n_samples - seq_len + 1, stride):
            if segments[i] == segments[i + seq_len - 1]:
                self.valid_starts.append(i)

        if not self.valid_starts:
            self.valid_starts = [0]

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.valid_starts[idx]
        end = start + self.seq_len

        feat_win = self.features[start:end].copy()  # (L, D)
        targ_win = self.targets[start:end].copy()    # (L,)

        # Generate observation mask for this window
        # Mask shape: (L,) where 1 = observed, 0 = missing
        if self.is_training:
            rate = self.rng.uniform(self.missing_rate_range[0], self.missing_rate_range[1])
            # 50% chance random MCAR, 50% chance block outage (1 to 6 hours)
            if self.rng.random() < 0.5:
                mask_win = (self.rng.uniform(0.0, 1.0, size=self.seq_len) >= rate).astype(np.float32)
            else:
                block_len = self.rng.integers(1, min(7, self.seq_len // 2))
                mask_win = np.ones(self.seq_len, dtype=np.float32)
                block_start = self.rng.integers(0, self.seq_len - block_len + 1)
                mask_win[block_start : block_start + block_len] = 0.0
        else:
            # Deterministic 20% MCAR mask for validation
            rng_val = np.random.default_rng(42 + idx)
            mask_win = (rng_val.uniform(0.0, 1.0, size=self.seq_len) >= 0.20).astype(np.float32)

        # Zero out low-cost PM2.5 channels (indices 0, 1, 2) where mask == 0
        # Meteorology (RH, Temp) and cyclical time features remain intact
        feat_win[mask_win == 0, :3] = 0.0

        # Append mask as an explicit feature column -> total input channels = D + 1
        if self.use_mask_feature:
            mask_col = mask_win[:, np.newaxis]
        else:
            mask_col = np.ones((self.seq_len, 1), dtype=np.float32)
        input_tensor = np.concatenate([feat_win, mask_col], axis=-1)  # (L, D+1)

        # Convert to PyTorch tensors with shape (Channels, Length)
        x_t = torch.tensor(input_tensor.T, dtype=torch.float32)
        m_t = torch.tensor(mask_win, dtype=torch.float32)
        y_t = torch.tensor(targ_win, dtype=torch.float32)

        return x_t, m_t, y_t


class DataNormalizer:
    """
    Fits standard scalers strictly on training data features and BAM targets.
    Provides leak-free forward and inverse scaling transforms.
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.feat_scaler = StandardScaler()
        self.targ_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, target_col: str = "bam_reference"):
        X_train = train_df[self.feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df[[target_col]].to_numpy(dtype=np.float32)

        self.feat_scaler.fit(X_train)
        self.targ_scaler.fit(y_train)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, target_col: str = "bam_reference") -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted on training data before transform.")
        X = self.feat_scaler.transform(df[self.feature_cols].to_numpy(dtype=np.float32))
        y = self.targ_scaler.transform(df[[target_col]].to_numpy(dtype=np.float32)).squeeze(-1)
        return X, y

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        if y_scaled.ndim == 1:
            y_2d = y_scaled[:, np.newaxis]
            return self.targ_scaler.inverse_transform(y_2d).squeeze(-1)
        return self.targ_scaler.inverse_transform(y_scaled)


def train_joint_tcn_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "bam_reference",
    lambda_cal: float = 1.0,
    lambda_rec: float = 1.0,
    use_mask_feature: bool = True,
    hidden_channels: int = 32,
    dilations: Optional[List[int]] = None,
    kernel_size: int = 3,
    dropout: float = 0.15,
    norm_type: str = "batch",
    seq_len: int = 24,
    stride: int = 1,
    batch_size: int = 64,
    epochs: int = 35,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    random_seed: int = 42,
) -> Tuple[JointTCN, DataNormalizer, pd.DataFrame]:
    """
    Trains Joint TCN with joint multi-task Huber loss, learning-rate scheduling,
    and validation early stopping.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if feature_cols is None:
        feature_cols = [c for c in INPUT_FEATURE_COLS if c in train_df.columns]

    if dilations is None:
        dilations = [1, 2, 4, 8]

    # 1. Fit normalizer on training set strictly
    normalizer = DataNormalizer(feature_cols).fit(train_df, target_col=target_col)
    X_train, y_train = normalizer.transform(train_df, target_col=target_col)
    X_val, y_val = normalizer.transform(val_df, target_col=target_col)

    seg_train = train_df["segment_id"].to_numpy(dtype=int)
    seg_val = val_df["segment_id"].to_numpy(dtype=int)

    # 2. Datasets and DataLoaders
    train_dataset = TCNWindowDataset(X_train, y_train, seg_train, seq_len=seq_len, is_training=True, use_mask_feature=use_mask_feature, stride=stride, random_seed=random_seed)
    val_dataset = TCNWindowDataset(X_val, y_val, seg_val, seq_len=seq_len, is_training=False, use_mask_feature=use_mask_feature, stride=stride, random_seed=random_seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Input dim: len(feature_cols) + 1 (observation mask)
    input_dim = len(feature_cols) + 1

    model = JointTCN(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        dilations=dilations,
        kernel_size=kernel_size,
        dropout=dropout,
        norm_type=norm_type,
    ).to(device)

    criterion = JointMultiTaskLoss(lambda_cal=lambda_cal, lambda_rec=lambda_rec, delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history = []
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_accum = 0.0
        train_cal_accum = 0.0
        train_rec_accum = 0.0
        n_train_batches = 0

        for bx, bm, by in train_loader:
            bx, bm, by = bx.to(device), bm.to(device), by.to(device)
            optimizer.zero_grad()

            y_cal, y_rec, _ = model(bx, bm)
            total_loss, loss_dict = criterion(y_cal, y_rec, by, bm)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_accum += loss_dict["total_loss"]
            train_cal_accum += loss_dict["cal_loss"]
            train_rec_accum += loss_dict["rec_loss"]
            n_train_batches += 1

        avg_train_loss = train_loss_accum / max(1, n_train_batches)
        avg_train_cal = train_cal_accum / max(1, n_train_batches)
        avg_train_rec = train_rec_accum / max(1, n_train_batches)

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_cal_accum = 0.0
        val_rec_accum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for bx, bm, by in val_loader:
                bx, bm, by = bx.to(device), bm.to(device), by.to(device)
                y_cal, y_rec, _ = model(bx, bm)
                v_loss, v_dict = criterion(y_cal, y_rec, by, bm)

                val_loss_accum += v_dict["total_loss"]
                val_cal_accum += v_dict["cal_loss"]
                val_rec_accum += v_dict["rec_loss"]
                n_val_batches += 1

        avg_val_loss = val_loss_accum / max(1, n_val_batches)
        avg_val_cal = val_cal_accum / max(1, n_val_batches)
        avg_val_rec = val_rec_accum / max(1, n_val_batches)

        scheduler.step(avg_val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_cal_loss": round(avg_train_cal, 4),
            "train_rec_loss": round(avg_train_rec, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_cal_loss": round(avg_val_cal, 4),
            "val_rec_loss": round(avg_val_rec, 4),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        # Check for improvement
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state_dict, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # Early stop triggered
                break

    # Restore best validation checkpoint
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    history_df = pd.DataFrame(history)
    return model, normalizer, history_df
