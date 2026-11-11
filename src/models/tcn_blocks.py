"""
Module: src.models.tcn_blocks
Implements core causal dilated temporal convolutional network (TCN) residual blocks.
Enforces strict temporal causality: outputs at time step t depend exclusively on past and current inputs (<= t).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Convolution with causal left-padding to prevent future information leakage.
    Output sequence length exactly matches input sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad_len = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Length)
        Returns:
            Tensor of shape (Batch, Channels, Length)
        """
        # Pad only on the left (past) dimension: (pad_left, pad_right) = (pad_len, 0)
        x_padded = F.pad(x, (self.pad_len, 0))
        return self.conv(x_padded)


class ChannelLayerNorm1d(nn.Module):
    """
    Channel-wise Layer Normalization for 1D sequence tensors of shape (Batch, Channels, Length).
    Normalizes across channels at each individual time step independently.
    Guarantees strict temporal causality in BOTH training and evaluation modes
    (zero future temporal leakage across the Length dimension).
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, L) -> transpose to (B, L, C) -> LayerNorm -> transpose back to (B, C, L)
        x_t = x.transpose(1, 2)
        out_t = self.ln(x_t)
        return out_t.transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    """
    Dilated residual block for Temporal Convolutional Networks.
    Consists of two dilated causal convolutions, normalization, GELU activation,
    dropout, and a 1x1 residual skip connection.
    Supports either 'batch' (nn.BatchNorm1d, checkpoint-compatible) or
    'layer' (ChannelLayerNorm1d, strictly causal in train and eval modes).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.15,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        if norm_type == "layer":
            self.norm1 = ChannelLayerNorm1d(out_channels)
        else:
            self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        if norm_type == "layer":
            self.norm2 = ChannelLayerNorm1d(out_channels)
        else:
            self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # 1x1 conv residual connection if channel dimensions change
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        self.final_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return self.final_act(out + res)


class TemporalEncoder(nn.Module):
    """
    Shared Temporal Feature Encoder stacking dilated residual blocks.
    Exponentially expands the temporal receptive field to capture diurnal
    and multi-day synoptic meteorological patterns.
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
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        # Initial input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, kernel_size=1)

        # Stack of dilated residual blocks
        layers = []
        for d in dilations:
            layers.append(
                TemporalResidualBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )
        self.network = nn.Sequential(*layers)

    @property
    def receptive_field(self) -> int:
        """
        Effective temporal receptive field in hours based on kernel size, dilations,
        and two causal convolutions per residual block:
        RF = 1 + 2 * (kernel_size - 1) * sum(dilations)
        For K=3 and dilations=[1, 2, 4, 8]:
        RF = 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 1 + 2 * 2 * 15 = 61 hours.
        """
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, input_dim, Length)
        Returns:
            h: Hidden representation of shape (Batch, hidden_channels, Length)
        """
        proj = self.input_proj(x)
        h = self.network(proj)
        return h
