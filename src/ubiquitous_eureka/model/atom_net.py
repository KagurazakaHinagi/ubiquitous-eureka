"""
Module for the AtomNet model.
"""

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from ubiquitous_eureka.model.components.layers import AttentionGate3D, ConvBlock3D, MultiScaleAttention3D
from ubiquitous_eureka.model.components.losses import PhysicsInformedLoss
from ubiquitous_eureka.util.constants import ATOM_TYPES


class AtomNet(nn.Module):
    """
    UNet for atom segmentation and type prediction from cryo-EM density maps.
    Preprocess density maps to a preliminary atom map for diffusion model initialization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        atom_types: list[str] | None = None,
        use_attention_gates: bool = True,
        dropout: float = 0.1,
        activation: Literal["relu", "leaky_relu", "elu", "gelu"] = "relu",
        device: str | None = None,
    ):
        super().__init__()

        if atom_types is None:
            atom_types = list(ATOM_TYPES)

        self.atom_types = atom_types
        self.num_atom_types = len(atom_types)
        self.use_attention_gates = use_attention_gates
        self.num_levels = num_levels

        # Calculate channel sizes for each level
        channels = [base_channels * (2**i) for i in range(num_levels)]

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        current_channels = in_channels
        for i, ch in enumerate(channels):
            use_attn = i >= 2  # Use attention in deeper layers
            self.encoder_blocks.append(
                ConvBlock3D(current_channels, ch, dropout=dropout, activation=activation, use_attention=use_attn)
            )
            if i < num_levels - 1:  # No pooling after last encoder block
                self.encoder_pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = ch

        # Bottleneck
        self.bottleneck = ConvBlock3D(
            channels[-1], channels[-1] * 2, dropout=dropout, activation=activation, use_attention=True
        )

        # Decoder path
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        if use_attention_gates:
            self.attention_gates = nn.ModuleList()

        decoder_channels = channels[::-1]  # Reverse for decoder
        current_channels = channels[-1] * 2

        for i, ch in enumerate(decoder_channels):
            # Upsample
            self.decoder_upsamples.append(nn.ConvTranspose3d(current_channels, ch, kernel_size=2, stride=2, bias=False))

            # Attention gate
            if use_attention_gates:
                self.attention_gates.append(AttentionGate3D(ch, ch, ch // 2))

            # Decoder block (input: upsampled + skip connection)
            use_attn = i < 2  # Use attention in shallower decoder layers
            self.decoder_blocks.append(
                ConvBlock3D(ch * 2, ch, dropout=dropout, activation=activation, use_attention=use_attn)
            )
            current_channels = ch

        # Output heads
        final_channels = channels[0]

        # Atom presence prediction (binary segmentation)
        self.atom_presence_head = nn.Sequential(
            nn.Conv3d(final_channels, final_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(final_channels // 2, 1, kernel_size=1),
        )

        # Atom type prediction (multi-class classification)
        self.atom_type_head = nn.Sequential(
            nn.Conv3d(final_channels, final_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(final_channels // 2, self.num_atom_types, kernel_size=1),
        )

        # Feature head for DiT conditioning
        self.feature_head = nn.Sequential(
            nn.Conv3d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(final_channels),
            nn.ReLU(inplace=True),
            MultiScaleAttention3D(final_channels),
            nn.Conv3d(final_channels, final_channels, kernel_size=1),
        )

        # Confidence/uncertainty head
        self.confidence_head = nn.Sequential(
            nn.Conv3d(final_channels, final_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(final_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(final_channels // 4, 1, kernel_size=1),
        )

        # Initialize weights
        self._initialize_weights()

        # Physics constraints
        self.physics_loss = PhysicsInformedLoss(device=device or "cpu")

    def _initialize_weights(self):
        """Initialize network weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass of the UNet.

        Args:
            x: Input density map [B, 1, D, H, W]

        Returns:
            Dictionary containing:
            - atom_presence: Atom presence probability [B, 1, D, H, W]
            - atom_types: Atom type logits [B, num_atom_types, D, H, W]
            - features: Feature maps for DiT conditioning [B, channels, D, H, W]
            - confidence: Prediction confidence [B, 1, D, H, W]
            - encoder_features: Multi-scale encoder features for analysis
        """
        # Store encoder features and skip connections
        encoder_features = []
        skip_connections = []

        # Encoder path
        current = x
        for i, (block, pool) in enumerate(zip(self.encoder_blocks[:-1], self.encoder_pools)):
            current = block(current)
            encoder_features.append(current)
            skip_connections.append(current)
            current = pool(current)

        # Last encoder block (no pooling)
        current = self.encoder_blocks[-1](current)
        encoder_features.append(current)
        skip_connections.append(current)

        # Bottleneck
        current = self.bottleneck(current)

        # Decoder path
        decoder_features = []
        for i, (upsample, block) in enumerate(zip(self.decoder_upsamples, self.decoder_blocks)):
            # Upsample
            current = upsample(current)

            # Get corresponding skip connection
            skip = skip_connections[-(i + 1)]

            # Apply attention gate if enabled
            if self.use_attention_gates:
                skip = self.attention_gates[i](current, skip)

            # Concatenate and process
            current = torch.cat([current, skip], dim=1)
            current = block(current)
            decoder_features.append(current)

        # Generate outputs
        outputs = {
            "atom_presence": torch.sigmoid(self.atom_presence_head(current)),
            "atom_types": self.atom_type_head(current),
            "features": self.feature_head(current),
            "confidence": torch.sigmoid(self.confidence_head(current)),
            "encoder_features": encoder_features,
            "decoder_features": decoder_features,
        }

        return outputs

    def get_atom_predictions(
        self, outputs: dict[str, torch.Tensor], presence_threshold: float = 0.5, confidence_threshold: float = 0.3
    ) -> dict[str, torch.Tensor]:
        """
        Extract atom predictions from model outputs.

        Args:
            outputs: Model output dictionary
            presence_threshold: Threshold for atom presence
            confidence_threshold: Minimum confidence threshold

        Returns:
            Dictionary with processed predictions
        """
        atom_presence = outputs["atom_presence"] > presence_threshold
        atom_type_probs = F.softmax(outputs["atom_types"], dim=1)
        atom_type_pred = torch.argmax(atom_type_probs, dim=1, keepdim=True)
        confidence_mask = outputs["confidence"] > confidence_threshold

        # Combine masks
        valid_atoms = atom_presence & confidence_mask

        return {
            "atom_positions": valid_atoms,
            "atom_types": atom_type_pred,
            "atom_type_probs": atom_type_probs,
            "confidence": outputs["confidence"],
            "valid_atoms": valid_atoms,
        }

    def extract_atom_coordinates(
        self,
        predictions: dict[str, torch.Tensor],
        voxel_size: float = 1.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, list]:
        """
        Extract 3D coordinates of predicted atoms.

        Args:
            predictions: Output from get_atom_predictions
            voxel_size: Voxel size in Angstroms
            origin: Origin coordinates

        Returns:
            Dictionary with atom coordinates and types
        """
        valid_atoms = predictions["valid_atoms"].squeeze(1)  # Remove channel dimension
        atom_types = predictions["atom_types"].squeeze(1)

        batch_coords = []
        batch_types = []

        for b in range(valid_atoms.shape[0]):
            # Get coordinates of valid atoms
            coords = torch.nonzero(valid_atoms[b], as_tuple=False).float()

            if len(coords) > 0:
                # Convert voxel coordinates to real coordinates
                coords = coords * voxel_size
                coords[:, 0] += origin[2]  # z
                coords[:, 1] += origin[1]  # y
                coords[:, 2] += origin[0]  # x

                # Get corresponding atom types
                atom_type_indices = atom_types[b][valid_atoms[b]]

                batch_coords.append(coords)
                batch_types.append(atom_type_indices)
            else:
                batch_coords.append(torch.empty(0, 3, device=valid_atoms.device))
                batch_types.append(torch.empty(0, dtype=torch.long, device=valid_atoms.device))

        return {
            "coordinates": batch_coords,
            "atom_type_indices": batch_types,
            "atom_type_names": [[self.atom_types[idx.item()] for idx in types] for types in batch_types],
        }

    def compute_physics_loss(
        self, coordinates_batch: list[torch.Tensor], atom_types_batch: list[list[str]]
    ) -> torch.Tensor:
        """Compute physics-informed constraints loss."""
        return self.physics_loss(coordinates_batch, atom_types_batch)
