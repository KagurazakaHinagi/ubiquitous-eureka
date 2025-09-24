"""
Module for model constraints.
"""

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Geometry import Point3D

from ubiquitous_eureka.util.constants import (
    COMMON_BOND_ANGLES,
    COMMON_BOND_LENGTHS,
    COMMON_COORDINATION_NUMBERS,
    COMMON_VAN_DER_WAALS_RADII,
)


class PhysicsConstraints:
    """Physics-informed constraints for molecular validation."""

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Standard bond lengths (Angstroms) from crystallographic data
        self.bond_lengths = COMMON_BOND_LENGTHS

        # Standard bond angles (degrees)
        self.bond_angles = COMMON_BOND_ANGLES

        # Coordination numbers
        self.coordination_numbers = COMMON_COORDINATION_NUMBERS

    def bond_length_constraint(
        self, coordinates: torch.Tensor, atom_types: list[str], distance_threshold: float = 3.0
    ) -> torch.Tensor:
        """Enforce realistic bond lengths between atoms."""
        if len(coordinates) < 2:
            return torch.tensor(0.0, device=self.device)

        # Calculate pairwise distances
        dists = torch.cdist(coordinates.unsqueeze(0), coordinates.unsqueeze(0)).squeeze(0)

        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = dists[i, j]

                if distance < distance_threshold:  # Only check nearby atoms
                    type_i = atom_types[i] if i < len(atom_types) else "C"
                    type_j = atom_types[j] if j < len(atom_types) else "C"

                    bond_key = (type_i, type_j) if type_i <= type_j else (type_j, type_i)
                    if bond_key in self.bond_lengths:
                        min_dist, max_dist = self.bond_lengths[bond_key]

                        if distance < min_dist:
                            loss += (min_dist - distance) ** 2
                        elif distance > max_dist and distance < max_dist * 1.2:
                            loss += (distance - max_dist) ** 2

                        count += 1

        return loss / max(count, 1)

    def angle_constraint(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Enforce realistic bond angles for triplets of nearby atoms."""
        if len(coordinates) < 3:
            return torch.tensor(0.0, device=self.device)

        angle_violations = torch.tensor(0.0, device=self.device)
        count = 0

        # Check angles for triplets of atoms
        for i in range(len(coordinates) - 2):
            v1 = coordinates[i + 1] - coordinates[i]
            v2 = coordinates[i + 2] - coordinates[i + 1]

            # Calculate angle using dot product
            cos_angle = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
            angle = torch.acos(cos_angle) * 180.0 / torch.pi

            # Penalize very sharp or very flat angles (unphysical)
            if angle < 60.0 or angle > 150.0:
                angle_violations += (angle - 109.5) ** 2  # Tetrahedral angle as reference
                count += 1

        return angle_violations / max(count, 1)

    def van_der_waals_constraint(self, coordinates: torch.Tensor, atom_types: list[str]) -> torch.Tensor:
        """Simple van der Waals repulsion to prevent atom overlap."""
        if len(coordinates) < 2:
            return torch.tensor(0.0, device=self.device)

        # Van der Waals radii (Angstroms)
        vdw_radii = COMMON_VAN_DER_WAALS_RADII

        dists = torch.cdist(coordinates.unsqueeze(0), coordinates.unsqueeze(0)).squeeze(0)

        vdw_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                type_i = atom_types[i] if i < len(atom_types) else "C"
                type_j = atom_types[j] if j < len(atom_types) else "C"

                radius_i = vdw_radii.get(type_i, 1.70)
                radius_j = vdw_radii.get(type_j, 1.70)
                min_distance = radius_i + radius_j

                actual_distance = dists[i, j]
                if actual_distance < min_distance:
                    # Soft repulsion potential
                    vdw_loss += (min_distance - actual_distance) ** 2
                    count += 1

        return vdw_loss / max(count, 1)

    def rdkit_energy_constraint(self, coordinates: torch.Tensor, atom_types: list[str]) -> torch.Tensor:
        """RDKit-based molecular energy constraint."""
        if len(coordinates) < 2:
            return torch.tensor(0.0, device=self.device)

        try:
            mol = self._create_rdkit_mol(coordinates, atom_types)
            if mol is None:
                return torch.tensor(0.0, device=self.device)

            # Calculate force field energy
            ff = UFFGetMoleculeForceField(mol)
            if ff is not None:
                energy = ff.CalcEnergy()
                # Normalize energy (typical range 0-1000 kcal/mol)
                normalized_energy = min(energy / 1000.0, 10.0)
                return torch.tensor(normalized_energy, device=self.device)

        except Exception as e:
            raise RuntimeError(f"Failed to calculate RDKit energy: {e}")

        return torch.tensor(0.0, device=self.device)

    def _create_rdkit_mol(self, coordinates: torch.Tensor, atom_types: list[str]):
        """Helper to create RDKit molecule from coordinates."""

        try:
            mol = Chem.RWMol()

            # Add atoms
            atom_indices = {}
            coords_np = coordinates.detach().cpu().numpy()

            for i, atom_type in enumerate(atom_types):
                if atom_type not in ["UNKNOWN", "SOLVENT"]:
                    atom = Chem.Atom(atom_type)
                    atom_idx = mol.AddAtom(atom)
                    atom_indices[i] = atom_idx

            # Add bonds based on distance
            for i in range(len(coords_np)):
                if i not in atom_indices:
                    continue
                for j in range(i + 1, len(coords_np)):
                    if j not in atom_indices:
                        continue

                    distance = np.linalg.norm(coords_np[i] - coords_np[j])
                    if distance < 2.0:  # Conservative bonding threshold
                        mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)

            # Set 3D coordinates
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, coord in enumerate(coords_np):
                if i in atom_indices:
                    atom_idx = atom_indices[i]
                    conf.SetAtomPosition(atom_idx, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))

            mol.AddConformer(conf)
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol

        except Exception as e:
            raise RuntimeError(f"Failed to create RDKit molecule from coordinates: {e}")
