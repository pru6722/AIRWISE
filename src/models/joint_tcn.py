"""
Module: src.models.joint_tcn
Implements the Joint Calibration-and-Reconstruction Temporal Convolutional Network (Joint TCN).
Features a shared dilated causal temporal encoder, a calibration prediction head,
a reconstruction prediction head, and a joint multi-task loss function.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn_blocks import TemporalEncoder


class CalibrationHead(nn.Module):
    """
    Pointwise prediction head that maps shared temporal representations
    to reference BAM PM2.5 concentrations at observed timestamps.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden representation of shape (Batch, hidden_dim, Length)
        Returns:
            y_cal: Tensor of shape (Batch, Length)
        """
        # Transpose to (Batch, Length, hidden_dim) for linear layer
        h_t = h.transpose(1, 2)
        out = self.mlp(h_t).squeeze(-1)
        return out


class ReconstructionHead(nn.Module):
    """
    Pointwise prediction head that reconstructs reference BAM PM2.5 concentrations
    at artificially masked timestamps using temporal context and observation state.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Input dimension: hidden representation + calibrated estimate + mask indicator
        in_dim = hidden_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        y_cal: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: Hidden representation of shape (Batch, hidden_dim, Length)
            y_cal: Calibrated estimates of shape (Batch, Length)
            mask: Observation mask of shape (Batch, 1, Length)
        Returns:
            y_rec: Tensor of shape (Batch, Length)
        """
        h_t = h.transpose(1, 2)  # (Batch, Length, hidden_dim)
        y_cal_t = y_cal.unsqueeze(-1)  # (Batch, Length, 1)
        mask_t = mask.transpose(1, 2)  # (Batch, Length, 1)

        # Zero out calibration estimate where observation is missing to prevent target leakage
        y_cal_observed = y_cal_t * mask_t

        combined = torch.cat([h_t, y_cal_observed, mask_t], dim=-1)
        out = self.mlp(combined).squeeze(-1)
        return out


class JointTCN(nn.Module):
    """
    Full Joint Calibration and Reconstruction TCN.
    Consists of:
    1. Shared Dilated Causal Temporal Encoder
    2. Calibration Prediction Head (observed positions)
    3. Reconstruction Prediction Head (hidden positions)
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_channels: int = 32,
        dilations: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.15,
        norm_type: str = "batch",
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.dilations = dilations
        self.norm_type = norm_type

        self.shared_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
            norm_type=norm_type,
        )

        self.calibration_head = CalibrationHead(hidden_dim=hidden_channels)
        self.reconstruction_head = ReconstructionHead(hidden_dim=hidden_channels)

    @property
    def receptive_field(self) -> int:
        """Effective temporal receptive field in hours from the shared temporal encoder."""
        return self.shared_encoder.receptive_field

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature tensor of shape (Batch, input_dim, Length)
            mask: Binary observation mask of shape (Batch, 1, Length)
                  1 = observed, 0 = artificially masked.
        Returns:
            Tuple of (y_cal, y_rec, y_final):
            - y_cal: (Batch, Length) calibration predictions
            - y_rec: (Batch, Length) reconstruction predictions
            - y_final: (Batch, Length) composite prediction: y_cal where mask==1, y_rec where mask==0
        """
        # Ensure mask is broadcastable (Batch, 1, Length)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        h = self.shared_encoder(x)
        y_cal = self.calibration_head(h)
        y_rec = self.reconstruction_head(h, y_cal, mask)

        # Composite output: calibration where observed, reconstruction where hidden
        m_squeeze = mask.squeeze(1).bool()
        y_final = torch.where(m_squeeze, y_cal, y_rec)

        return y_cal, y_rec, y_final


class JointMultiTaskLoss(nn.Module):
    """
    Joint multi-task Huber loss with configurable task weights:
    L_joint = lambda_cal * L_cal + lambda_rec * L_rec + lambda_reg * L_reg
    """

    def __init__(
        self,
        lambda_cal: float = 1.0,
        lambda_rec: float = 1.0,
        delta: float = 1.0,
    ):
        super().__init__()
        self.lambda_cal = lambda_cal
        self.lambda_rec = lambda_rec
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(
        self,
        y_cal: torch.Tensor,
        y_rec: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        valid_target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates independent calibration and reconstruction losses:
        - L_cal evaluated exclusively on observed positions (mask == 1)
        - L_rec evaluated exclusively on artificially hidden positions (mask == 0)

        Args:
            y_cal: (Batch, Length) calibration predictions
            y_rec: (Batch, Length) reconstruction predictions
            y_true: (Batch, Length) ground truth BAM values
            mask: (Batch, Length) or (Batch, 1, Length) binary mask
            valid_target_mask: Optional mask indicating valid non-NaN ground-truth targets
        """
        if mask.dim() == 3:
            mask = mask.squeeze(1)

        if valid_target_mask is None:
            valid_target_mask = ~torch.isnan(y_true)

        obs_mask = (mask > 0.5) & valid_target_mask
        hid_mask = (mask <= 0.5) & valid_target_mask

        # Pointwise Huber losses
        loss_cal_all = self.huber(y_cal, y_true)
        loss_rec_all = self.huber(y_rec, y_true)

        n_obs = obs_mask.sum()
        n_hid = hid_mask.sum()

        l_cal = loss_cal_all[obs_mask].mean() if n_obs > 0 else torch.tensor(0.0, device=y_cal.device)
        l_rec = loss_rec_all[hid_mask].mean() if n_hid > 0 else torch.tensor(0.0, device=y_rec.device)

        total_loss = self.lambda_cal * l_cal + self.lambda_rec * l_rec

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "cal_loss": float(l_cal.item()),
            "rec_loss": float(l_rec.item()),
            "n_obs": int(n_obs.item()),
            "n_hid": int(n_hid.item()),
        }

        return total_loss, loss_dict


def verify_full_model_causality(
    model: JointTCN,
    seq_len: int = 64,
    cut_positions: Optional[List[int]] = None,
    perturbation_magnitudes: Optional[List[float]] = None,
    tolerance: float = 1e-6,
    batch_size: int = 4,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes comprehensive full-model future-perturbation causality verification:
    - Test A: Future-value perturbation across multiple magnitudes with fixed observation mask.
    - Test B: Future-mask perturbation with fixed input features.
    - Evaluates:
      * Multiple perturbation magnitudes (default: [0.1, 10.0, 100.0, 500.0, -50.0])
      * Multiple future cut positions (default: [16, 32, 48, 60])
      * Both output heads (y_cal, y_rec) and composite output (y_final)
      * Both evaluation (model.eval()) and training (model.train()) execution modes
      * Float32 machine precision tolerance bound (1.0e-6)
      * Verifies that normalization and convolution operations preserve temporal causality.

    Returns:
        Dict containing:
        - "passed": bool
        - "max_deviation": float
        - "tolerance": float
        - "test_a_results": List[Dict]
        - "test_b_results": List[Dict]
        - "total_checks": int
    """
    if cut_positions is None:
        cut_positions = [16, 32, 48, 60]
    if perturbation_magnitudes is None:
        perturbation_magnitudes = [0.1, 10.0, 100.0, 500.0, -50.0]

    torch.manual_seed(random_seed)

    # Cache and zero out dropout rates to isolate architectural causality from stochastic dropout masks
    orig_dropouts = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            orig_dropouts[name] = module.p
            module.p = 0.0

    B = batch_size
    D = model.input_dim
    L = seq_len
    x_base = torch.randn(B, D, L)
    mask_base = torch.ones(B, 1, L)

    test_a_records = []
    test_b_records = []
    max_dev = 0.0
    all_passed = True
    total_checks = 0

    try:
        for mode in ["eval", "train"]:
            if mode == "train":
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(mode == "train"):
                y_cal_base, y_rec_base, y_fin_base = model(x_base.clone(), mask_base.clone())

                for t0 in cut_positions:
                    # ---------------------------------------------------------
                    # TEST A: Future-Value Perturbation across Multiple Magnitudes
                    # ---------------------------------------------------------
                    for delta in perturbation_magnitudes:
                        x_pert_a = x_base.clone()
                        x_pert_a[:, :, t0:] += delta
                        y_cal_a, y_rec_a, y_fin_a = model(x_pert_a, mask_base.clone())

                        dev_cal = torch.max(torch.abs(y_cal_base[:, :t0] - y_cal_a[:, :t0])).item()
                        dev_rec = torch.max(torch.abs(y_rec_base[:, :t0] - y_rec_a[:, :t0])).item()
                        dev_fin = torch.max(torch.abs(y_fin_base[:, :t0] - y_fin_a[:, :t0])).item()

                        step_max = max(dev_cal, dev_rec, dev_fin)
                        max_dev = max(max_dev, step_max)
                        passed_step = step_max <= tolerance
                        if not passed_step:
                            all_passed = False

                        test_a_records.append({
                            "mode": mode,
                            "cut_t0": t0,
                            "delta": delta,
                            "dev_cal": dev_cal,
                            "dev_rec": dev_rec,
                            "dev_fin": dev_fin,
                            "passed": passed_step,
                        })
                        total_checks += 3

                    # ---------------------------------------------------------
                    # TEST B: Future-Mask Perturbation with Fixed Features
                    # ---------------------------------------------------------
                    mask_pert_b = mask_base.clone()
                    mask_pert_b[:, :, t0:] = 0.0
                    y_cal_b, y_rec_b, y_fin_b = model(x_base.clone(), mask_pert_b)

                    dev_cal_b = torch.max(torch.abs(y_cal_base[:, :t0] - y_cal_b[:, :t0])).item()
                    dev_rec_b = torch.max(torch.abs(y_rec_base[:, :t0] - y_rec_b[:, :t0])).item()
                    dev_fin_b = torch.max(torch.abs(y_fin_base[:, :t0] - y_fin_b[:, :t0])).item()

                    step_max_b = max(dev_cal_b, dev_rec_b, dev_fin_b)
                    max_dev = max(max_dev, step_max_b)
                    passed_step_b = step_max_b <= tolerance
                    if not passed_step_b:
                        all_passed = False

                    test_b_records.append({
                        "mode": mode,
                        "cut_t0": t0,
                        "dev_cal": dev_cal_b,
                        "dev_rec": dev_rec_b,
                        "dev_fin": dev_fin_b,
                        "passed": passed_step_b,
                    })
                    total_checks += 3

    finally:
        # Restore original dropout rates
        for name, module in model.named_modules():
            if name in orig_dropouts:
                module.p = orig_dropouts[name]

    return {
        "passed": all_passed,
        "max_deviation": max_dev,
        "tolerance": tolerance,
        "test_a_results": test_a_records,
        "test_b_results": test_b_records,
        "total_checks": total_checks,
    }

