"""
Module for handling molecular structures from PDB/mmCIF files.
"""

import warnings
from os import PathLike
from pathlib import Path
from typing import Literal

import biotite.structure.io.pdb as pdb
import numpy as np
import torch
from atomworks.io import parser
from atomworks.io.transforms.atom_array import remove_nan_coords
from atomworks.io.utils.io_utils import load_any, to_cif_file
from biotite import structure as struc
from biotite.structure import AtomArray, AtomArrayStack

from ubiquitous_eureka.data.density import DensityMap
from ubiquitous_eureka.util.constants import BACKBONE_ATOMS


class Structure:
    """
    Class for handling molecular structures from PDB/mmCIF files.
    """

    def __init__(
        self,
        filepath: PathLike | None = None,
        atom_array: AtomArray | None = None,
        device: str | torch.device | None = None,
    ):
        """Initialize Structure.

        Args:
            filepath (PathLike | None, optional): Path to PDB or mmCIF file. Defaults to None.
            atom_array (AtomArray | None, optional): Biotite AtomArray object. Defaults to None.
            device (str | torch.device | None, optional): Device for tensor operations. Defaults to None.
        """

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._atom_array = None
        self._filepath = None
        self._metadata = {}

        self._chain_info = None
        self._ligand_info = None
        self._asym_unit = None

        if filepath:
            self.load(filepath)
        elif atom_array is not None:
            self._atom_array = atom_array.copy()

    @property
    def atom_array(self) -> AtomArray | None:
        """Get the underlying AtomArray."""
        assert isinstance(self._atom_array, (AtomArray, type(None)))
        return self._atom_array

    @property
    def filepath(self) -> Path | None:
        """Get the file path of the loaded structure."""
        return self._filepath

    @property
    def metadata(self) -> dict:
        """Get metadata dictionary."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """Set metadata dictionary."""
        self._metadata = metadata

    def load(self, filepath: PathLike, clean_load: bool = True) -> "Structure":
        """
        Load structure from PDB or mmCIF file via AtomWorks IO for cleaning.

        Note: If clean_load is False, chain_info, ligand_info, asym_unit, and metadata will not be populated.

        Args:
            filepath (PathLike): Path to structure file (PDB or mmCIF).
            clean_load (bool, optional): Whether to clean the structure using AtomWorks IO. Defaults to True.

        Returns:
            Structure: Self for method chaining.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self._filepath = filepath

        if clean_load:
            print(f"Loading and cleaning structure from {filepath} using AtomWorks IO...")
            try:
                parsed_structure = parser.parse(filepath, hydrogen_policy="remove")
                if (
                    parsed_structure is None
                    or "assembly" not in parsed_structure
                    or len(parsed_structure["assembly"]) == 0
                ):
                    raise ValueError("Parsed structure is empty or missing assembly information.")
                self._assembly = parsed_structure["assembly"]
                self._metadata = parsed_structure.get("metadata", {})
                self._chain_info = parsed_structure.get("chain_info", None)
                self._ligand_info = parsed_structure.get("ligand_info", None)
                self._asym_unit = parsed_structure.get("asym_unit", None)

                # Use the first assembly by default
                self._atom_array = self._assembly["1"]

            except Exception as e:
                raise RuntimeError(f"Failed to load and clean structure from {filepath}: {e}")

        else:
            print(f"Loading structure from {filepath} using Biotite...")
            try:
                self._atom_array = load_any(filepath)

            except Exception as e:
                raise RuntimeError(f"Failed to load structure from {filepath}: {e}")

        # If multiple models, take the first one only.
        if isinstance(self._atom_array, AtomArrayStack):
            if len(self._atom_array) > 1:
                print("Warning: Multiple models found. Using the first model.")
            self._atom_array = self._atom_array[0]

        # Remove atoms with NaN coordinates
        assert isinstance(self._atom_array, AtomArray)
        self._atom_array = remove_nan_coords(self._atom_array)

        if isinstance(self._atom_array, AtomArray) and len(self._atom_array) > 0:
            print(f"Loaded structure from {filepath} with {len(self._atom_array)} valid atoms.")
        else:
            warnings.warn(f"Loaded structure from {filepath} but atom array is empty or invalid.")
        return self

    def save(self, filepath: PathLike, fileformat: str = "auto") -> "Structure":
        """
        Save structure to PDB or mmCIF file.

        Args:
            filepath (PathLike): Output file path.
            fileformat (str, optional): File format ('pdb', 'cif', or 'auto'). Defaults to 'auto'.

        Returns:
            Structure: Self for method chaining.
        """
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to save.")

        filepath = Path(filepath)

        if fileformat == "auto":
            fileformat = "pdb" if filepath.suffix.lower() in [".pdb", ".ent"] else "cif"

        print(f"Saving structure to {filepath} in {fileformat} format...")

        try:
            if fileformat == "pdb":
                pdb_file = pdb.PDBFile()
                pdb.set_structure(pdb_file, self._atom_array)
                pdb_file.write(str(filepath))
            elif fileformat == "cif":
                to_cif_file(self._atom_array, filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            raise RuntimeError(f"Failed to save structure to {filepath}: {e}")

        return self

    @property
    def coordinates(self) -> np.ndarray:
        """Get coordinates as NumPy array [N, 3]."""
        if not isinstance(self._atom_array, AtomArray):
            return np.empty((0, 3))
        coord = self._atom_array.coord
        return coord or np.empty((0, 3))

    @coordinates.setter
    def coordinates(self, coords: np.ndarray | torch.Tensor):
        """Set coordinates from NumPy array or PyTorch tensor."""
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to set coordinates.")

        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        self._atom_array.coord = coords.astype(np.float32)

    def mask_atoms(self, **kwargs) -> np.ndarray:
        """
        Create a boolean mask for atoms based on AtomArray annotations.

        Returns:
            np.ndarray: Boolean mask array.
        """
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to mask atoms.")

        mask = np.ones(len(self._atom_array), dtype=bool)

        for key, value in kwargs.items():
            if hasattr(self._atom_array, key):
                attr = getattr(self._atom_array, key)
                if isinstance(value, (list, np.ndarray)):
                    attr_mask = np.isin(attr, value)
                else:
                    attr_mask = attr == value
                mask &= attr_mask

        return mask

    def filter_atoms(self, **kwargs) -> "Structure":
        """
        Filter atoms based on AtomArray annotations.

        Returns:
            Structure: Filtered structure.
        """

        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to filter atoms.")

        mask = self.mask_atoms(**kwargs)
        filtered_array = self._atom_array[mask]

        assert isinstance(filtered_array, AtomArray)
        new_structure = Structure(atom_array=filtered_array, device=self.device)
        new_structure.metadata = self.metadata.copy()
        return new_structure

    def get_atom_by_elements(self, elements: str | list[str]) -> "Structure":
        """Get a new Structure containing only atoms of specified elements."""
        if isinstance(elements, str):
            elements = [elements]
        return self.filter_atoms(element=elements)

    def get_backbone_atoms_mask(self) -> np.ndarray:
        """Get a boolean mask for backbone atoms (N, CA, C, O)."""
        return self.mask_atoms(atom_name=BACKBONE_ATOMS)

    def get_backbone_atoms(self) -> "Structure":
        """Get a new Structure containing only backbone atoms (N, CA, C, O)."""
        return self.filter_atoms(atom_name=BACKBONE_ATOMS)

    def get_atoms_in_radius_mask(self, center: np.ndarray | torch.Tensor, radius: float) -> np.ndarray:
        """Get a boolean mask for atoms within a specified radius from a center point."""
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to filter atoms.")

        if isinstance(center, torch.Tensor):
            center = center.detach().cpu().numpy()
        center = np.asarray(center).reshape(
            3,
        )

        distances = np.linalg.norm(self.coordinates - center, axis=1)
        mask = distances <= radius
        return mask

    def get_atoms_in_radius(self, center: np.ndarray | torch.Tensor, radius: float) -> "Structure":
        """Get a new Structure containing only atoms within a specified radius from a center point."""
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to filter atoms.")

        mask = self.get_atoms_in_radius_mask(center, radius)
        filtered_array = self._atom_array[mask]

        assert isinstance(filtered_array, AtomArray)
        new_structure = Structure(atom_array=filtered_array, device=self.device)
        new_structure.metadata = self.metadata.copy()
        return new_structure

    def calculate_rmsd(
        self,
        other: "Structure",
        superimpose: Literal["none", "backbone", "all"] = "none",
        backbone_only: bool = False,
    ) -> float:
        """
        Calculate RMSD to another Structure.

        Args:
            other (Structure): The other structure to compare.
            superimpose (Literal["none", "backbone", "all"], optional): Whether to superimpose before comparison. Defaults to "none".
            backbone_only (bool, optional): Whether to consider only backbone atoms. Defaults to False.

        Returns:
            float: The calculated RMSD value in Angstroms.
        """
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to calculate RMSD.")
        if not isinstance(other._atom_array, AtomArray):
            raise ValueError("The other structure has no valid structure loaded to calculate RMSD.")

        if superimpose != "none":
            superimposed_other = self.superimpose_on(other, mode=superimpose)
            assert isinstance(superimposed_other._atom_array, AtomArray)

            return struc.rmsd(
                self.get_backbone_atoms()._atom_array if backbone_only else self._atom_array,
                superimposed_other.get_backbone_atoms()._atom_array
                if backbone_only
                else superimposed_other._atom_array,
            )

        return struc.rmsd(
            self.get_backbone_atoms()._atom_array if backbone_only else self._atom_array,
            other.get_backbone_atoms()._atom_array if backbone_only else other._atom_array,
        )

    def superimpose_on(self, ref: "Structure", mode: Literal["backbone", "all"] = "all") -> "Structure":
        """
        Superimpose this structure onto a reference structure.

        Args:
            ref (Structure): The reference structure to superimpose onto.
            mode (Literal["backbone", "all"], optional): The mode of superimposition. Defaults to "all".

        Returns:
            Structure: A new Structure instance that is superimposed onto the reference.
        """
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to superimpose.")
        if not isinstance(ref._atom_array, AtomArray):
            raise ValueError("The reference structure has no valid structure loaded to superimpose.")

        superimposed_array = struc.superimpose(
            self._atom_array,
            ref._atom_array,
            atom_mask=self.get_backbone_atoms_mask() if mode == "backbone" else None,
        )
        assert isinstance(superimposed_array, AtomArray)

        new_structure = Structure(atom_array=superimposed_array, device=self.device)
        new_structure.metadata = self.metadata.copy()
        return new_structure

    def calculate_angles(self, triplets: list[tuple[int, int, int]] | None = None) -> np.ndarray:
        """
        Calculate angles (in radians) for specified triplets of atoms.

        Args:
            triplets (list[tuple[int, int, int]] | None, optional): A list of triplets, where each triplet is a tuple
            of three atom indices. Defaults to None, in which case all possible triplets are considered.

        Returns:
            np.ndarray: An array of angles (in radians) for the specified triplets.
        """
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to calculate angles.")

        if triplets is None:
            # Calculate backbone angles (N-CA-C, CA-C-N, C-N-CA) for proteins
            backbone = self.get_backbone_atoms()
            assert isinstance(backbone._atom_array, AtomArray)
            if len(backbone._atom_array) < 3:
                return np.array([])

            ca_indices = np.where(backbone._atom_array.atom_name == "CA")[0]
            if len(ca_indices) < 3:
                return np.array([])

            triplets = [(ca_indices[i], ca_indices[i + 1], ca_indices[i + 2]) for i in range(len(ca_indices) - 2)]

        angles = []
        coords = self.coordinates
        for i, j, k in triplets:
            if i < len(coords) and j < len(coords) and k < len(coords):
                angle = struc.angle(coords[i], coords[j], coords[k])
                angles.append(angle)

        return np.array(angles)

    def calculate_sasa(self, probe_radius: float = 1.4) -> np.ndarray:
        """
        Calculate the solvent-accessible surface area (SASA) for each atom.

        Args:
            probe_radius (float, optional): The radius of the probe sphere. Defaults to 1.4 Å.

        Returns:
            np.ndarray: An array of SASA values for each atom in square angstroms (Å²).
        """
        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to calculate SASA.")

        return struc.sasa(self._atom_array, probe_radius=probe_radius)  # type: ignore

    def to_density_map(
        self,
        voxel_size: float = 0.5,
        padding: float = 10.0,
        sigma: float = 1.5,
        device: str | torch.device | None = None,
    ) -> "DensityMap":
        """
        Simulate a 3D density map from the atomic structure.

        Args:
            voxel_size (float, optional): The size of each voxel in the density map. Defaults to 0.5.
            padding (float, optional): The amount of padding to add around the structure. Defaults to 10.0.
            sigma (float, optional): The standard deviation for the Gaussian kernel. Defaults to 1.5.
            device (str | torch.device | None, optional): The device to perform the computation on. Defaults to None.

        Returns:
            DensityMap: DensityMap object representing the simulated density map.
        """

        if not isinstance(self._atom_array, AtomArray):
            raise ValueError("No valid structure loaded to convert to density map.")

        device = device or self.device

        # Get bounding box with padding using vectorized operations
        coords = self.coordinates
        min_coords = coords.min(axis=0) - padding
        max_coords = coords.max(axis=0) + padding

        # Calculate grid dimensions
        grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

        print(f"Generating density map with dimensions: {grid_dims} on device: {device}")

        # Initialize empty density grid
        device_torch = torch.device(device)
        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.arange(grid_dims[2], device=device_torch) * voxel_size + min_coords[2],
            torch.arange(grid_dims[1], device=device_torch) * voxel_size + min_coords[1],
            torch.arange(grid_dims[0], device=device_torch) * voxel_size + min_coords[0],
            indexing="ij",
        )

        # Initialize density map
        density = torch.zeros_like(z_grid)

        # Convert coordinates to torch tensor
        coords_tensor = torch.from_numpy(coords).to(device_torch)

        # Add Gaussian for each atom
        for i, (x, y, z) in enumerate(coords_tensor):
            # Calculate squared distance from the atom to each voxel
            dist_squared = (x_grid - x) ** 2 + (y_grid - y) ** 2 + (z_grid - z) ** 2
            # Add Gaussian contribution
            density += torch.exp(-dist_squared / (2 * sigma**2))

        # Create DensityMap object
        density_map = DensityMap(device=device)
        density_map.set_data(density.cpu().numpy(), voxel_size=voxel_size, origin=min_coords)

        return density_map
