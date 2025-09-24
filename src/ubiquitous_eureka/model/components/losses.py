"""
Module for model losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ubiquitous_eureka.model.atom_net import AtomNet
from ubiquitous_eureka.model.components.constraints import PhysicsConstraints


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss combining multiple molecular constraints."""

    def __init__(
        self,
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        vdw_weight: float = 0.3,
        rdkit_weight: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.vdw_weight = vdw_weight
        self.rdkit_weight = rdkit_weight

        self.constraints = PhysicsConstraints(device=device)

    def forward(self, coordinates_batch: list[torch.Tensor], atom_types_batch: list[list[str]]) -> torch.Tensor:
        """Compute physics-informed loss for a batch of predictions."""
        total_loss = torch.tensor(0.0, device=next(iter(coordinates_batch)).device if coordinates_batch else "cpu")
        valid_structures = 0

        for coordinates, atom_types in zip(coordinates_batch, atom_types_batch):
            if len(coordinates) == 0:
                continue

            # Bond length constraints
            bond_loss = self.constraints.bond_length_constraint(coordinates, atom_types)

            # Bond angle constraints
            angle_loss = self.constraints.angle_constraint(coordinates)

            # Van der Waals constraints
            vdw_loss = self.constraints.van_der_waals_constraint(coordinates, atom_types)

            # RDKit energy constraints (if available)
            rdkit_loss = self.constraints.rdkit_energy_constraint(coordinates, atom_types)

            # Combine losses
            structure_loss = (
                self.bond_weight * bond_loss
                + self.angle_weight * angle_loss
                + self.vdw_weight * vdw_loss
                + self.rdkit_weight * rdkit_loss
            )

            total_loss += structure_loss
            valid_structures += 1

        return total_loss / max(valid_structures, 1)


class AtomNetLoss(nn.Module):
    """AtomNet loss with physics constraints."""

    def __init__(
        self,
        presence_weight: float = 1.0,
        type_weight: float = 1.0,
        confidence_weight: float = 0.5,
        physics_weight: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.presence_weight = presence_weight
        self.type_weight = type_weight
        self.confidence_weight = confidence_weight
        self.physics_weight = physics_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        eps = 1e-8
        pred = torch.clamp(pred, eps, 1 - eps)

        ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
        return (focal_weight * ce_loss).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        model: AtomNet | None = None,
        coordinates_batch: list[torch.Tensor] | None = None,
        atom_types_batch: list[list[str]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate multi-task loss with physics constraints.

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            model: Model instance for physics loss computation
            coordinates_batch: Predicted atom coordinates
            atom_types_batch: Predicted atom types
        """
        losses = {}
        device = outputs["atom_presence"].device

        # Atom presence loss (focal loss to handle imbalance)
        presence_loss = self.focal_loss(outputs["atom_presence"], targets["atom_presence"].float())
        losses["presence"] = presence_loss * self.presence_weight

        # Atom type loss (only where atoms are present)
        if "atom_types" in targets:
            atom_mask = targets["atom_presence"].bool().squeeze(1)  # [B, D, H, W]

            if atom_mask.sum() > 0:
                # Get predictions and targets for atom locations
                pred_types = outputs["atom_types"]  # [B, num_classes, D, H, W]
                true_types = targets["atom_types"]  # [B, D, H, W]

                # Flatten spatial dimensions and apply mask
                pred_flat = pred_types.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, num_classes]
                pred_flat = pred_flat[atom_mask]  # [N_atoms, num_classes]
                true_flat = true_types[atom_mask]  # [N_atoms]

                # Apply label smoothing
                type_loss = F.cross_entropy(pred_flat, true_flat, label_smoothing=self.label_smoothing)
                losses["type"] = type_loss * self.type_weight
            else:
                losses["type"] = torch.tensor(0.0, device=device)

        # Confidence loss (encourage high confidence for correct predictions)
        if "confidence" in outputs:
            # Create confidence targets based on prediction correctness
            with torch.no_grad():
                presence_correct = (outputs["atom_presence"] > 0.5) == targets["atom_presence"].bool()
                confidence_target = presence_correct.float()

            confidence_loss = F.binary_cross_entropy(outputs["confidence"], confidence_target)
            losses["confidence"] = confidence_loss * self.confidence_weight

        # Physics-informed loss
        if (
            model is not None
            and coordinates_batch is not None
            and atom_types_batch is not None
            and self.physics_weight > 0
        ):
            physics_loss = model.compute_physics_loss(coordinates_batch, atom_types_batch)
            losses["physics"] = physics_loss * self.physics_weight

        # Total loss
        losses["total"] = sum(losses.values())

        return losses