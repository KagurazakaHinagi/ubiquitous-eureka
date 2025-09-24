"""
Module for model layers.
"""

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Fixed 3D convolutional block with proper normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.1,
        activation: Literal["relu", "leaky_relu", "elu", "gelu"] = "relu",
        use_attention: bool = False,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout)

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)

        # Optional attention
        self.attention = MultiScaleAttention3D(out_channels) if use_attention else nn.Identity()

        # Residual connection handling
        self.residual_conv = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        # Main path
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.activation(self.bn2(self.conv2(out)))

        # Apply attention
        out = self.attention(out)

        # Residual connection
        out = out + residual

        return out


class MultiScaleAttention3D(nn.Module):
    """Fixed multi-scale spatial attention mechanism for 3D volumes."""

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels

        # Multi-scale convolutions with proper channel handling
        self.scale_convs = nn.ModuleList(
            [
                nn.Conv3d(channels, channels // num_scales, kernel_size=2 * i + 1, padding=i, bias=False)
                for i in range(num_scales)
            ]
        )

        # Combine multi-scale features
        self.fusion_conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

        # Attention mechanism
        self.attention_conv = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: [B, C, D, H, W]

        # Multi-scale feature extraction
        scale_features = []
        for conv in self.scale_convs:
            scale_feat = F.relu(conv(x))
            scale_features.append(scale_feat)

        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=1)

        # Restore channel dimension
        fused = self.fusion_conv(multi_scale)

        # Generate attention map
        attention = torch.sigmoid(self.attention_conv(fused))

        return x * attention


class AttentionGate3D(nn.Module):
    """Fixed 3D attention gate for skip connections."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()

        self.gate_conv = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.skip_conv = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.attention_conv = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm3d(inter_channels)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Feature from decoder path (lower resolution)
            skip: Feature from encoder path (higher resolution)
        """
        # Upsample gate to match skip resolution if needed
        if gate.shape[2:] != skip.shape[2:]:
            gate_upsampled = F.interpolate(gate, size=skip.shape[2:], mode="trilinear", align_corners=False)
        else:
            gate_upsampled = gate

        # Compute attention weights
        gate_feat = self.gate_conv(gate_upsampled)
        skip_feat = self.skip_conv(skip)
        combined = F.relu(self.bn(gate_feat + skip_feat))
        attention = torch.sigmoid(self.attention_conv(combined))

        # Apply attention to skip connection
        return skip * attention
