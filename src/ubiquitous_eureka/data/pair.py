"""
Module for streaming model training and validation datasets.
"""

import logging
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, IterableDataset
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ZipStore

from ubiquitous_eureka.common import get_git_hash
from ubiquitous_eureka.data.density import DensityMap
from ubiquitous_eureka.data.prep_utils import (
    fetch_density_map,
    fetch_structure,
    get_fitted_structure_id_from_emdb_entry,
)
from ubiquitous_eureka.data.structure import Structure

logger = logging.getLogger(__name__)


def world_xyz_to_voxel_zyx(
    coord_world_xyz: np.ndarray,  # (3, ) in (x, y, z), Å
    origin_world_xyz: np.ndarray,  # (3, ) in (x, y, z), Å
    voxel_size_zyx: np.ndarray,  # (3, ) in (z, y, x), Å
) -> np.ndarray:
    """Convert world coordinates to voxel coordinates."""
    vx, vy, vz = voxel_size_zyx[2], voxel_size_zyx[1], voxel_size_zyx[0]
    x0, y0, z0 = origin_world_xyz[0], origin_world_xyz[1], origin_world_xyz[2]
    x, y, z = coord_world_xyz[0], coord_world_xyz[1], coord_world_xyz[2]
    zi = (z - z0) / vz
    yi = (y - y0) / vy
    xi = (x - x0) / vx
    return np.array([zi, yi, xi])  # (3, ) in (z, y, x)


class DensityStructPair(Dataset):
    """Class representing pairs of density and structure."""

    def __init__(
        self,
        density_list: list[DensityMap] | None = None,
        structure_list: list[Structure] | None = None,
        simulated: bool = False,
    ):
        """Initialize DensityStructPair."""
        self.density_list = density_list or []
        self.structure_list = structure_list or []
        self.density_transforms = []
        self.simulated = simulated

        if self.density_list and self.structure_list:
            if len(self.density_list) != len(self.structure_list):
                raise ValueError("Density and structure must have the same length.")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        assert len(self.density_list) == len(self.structure_list)
        return len(self.density_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Extract features from the density and structure pairs."""
        density = self.density_list[index]
        structure = self.structure_list[index]

        density_feature = {
            "volume": np.asarray(density.data, dtype=np.float32),
            "voxel_size": np.asarray(density.voxel_size, dtype=np.float32),
            "origin_world": np.asarray(density.origin, dtype=np.float32),
            # "local_res": np.asarray(density.local_res, dtype=np.float32),
        }

        structure_feature = {
            "coord_world": np.asarray(structure.coordinates, dtype=np.float32),
            "atomic_number": np.asarray(structure.get_annotation("atomic_number"), dtype=np.uint8),
            "is_backbone": np.asarray(structure.get_annotation("is_backbone"), dtype=np.bool_),
            "chain_type": np.asarray(structure.get_annotation("chain_type"), dtype=np.uint8),
            "atom_name": np.asarray(structure.get_annotation("atom_name"), dtype=np.str_),
            "res_id": np.asarray(structure.get_annotation("res_id"), dtype=np.int32),
            "res_name": np.asarray(structure.get_annotation("res_name"), dtype=np.str_),
            "chain_id": np.asarray(structure.get_int_chain_ids(), dtype=np.uint8),
        }
        structure_bonds = structure.get_all_bonds()
        if structure_bonds is not None:
            structure_feature["bonds"] = np.asarray(structure_bonds[0], dtype=np.int32)
            structure_feature["bonds_type"] = np.asarray(structure_bonds[1], dtype=np.uint8)
        else:
            structure_feature["bonds"] = np.empty((0, 3), dtype=np.int32)
            structure_feature["bonds_type"] = np.empty((0, 3), dtype=np.uint8)

        emdb_id = density.emdb_id or ""
        wwpdb_id = structure.wwpdb_id or ""

        return {
            "sample_id": index,
            "density": density_feature,
            "structure": structure_feature,
            "emdb_id": emdb_id,
            "wwpdb_id": wwpdb_id,
        }

    def fetch_by_ids(
        self, cache_dir: PathLike, emdb_ids: list[str], wwpdb_ids: list[str] | None = None, overwrite: bool = False
    ) -> "DensityStructPair":
        """
        Fetch the pair data from EMDB IDs.
        Will auto-check for fitted structure IDs if not provided.

        Args:
            cache_dir (PathLike): The directory to cache the data.
            emdb_ids (list[str]): The list of EMDB IDs.
            wwpdb_ids (list[str] | None): The list of wwPDB IDs.
            overwrite (bool): Whether to overwrite the cached data.
        """

        if wwpdb_ids is not None and len(wwpdb_ids) != len(emdb_ids):
            raise ValueError("Number of wwPDB IDs must be the same as the number of EMDB IDs.")

        cache_dir = Path(cache_dir) / __name__
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Fetching {len(emdb_ids)} density maps and structures...")

        success_count = 0
        for i, emdb_id in enumerate(emdb_ids):
            wwpdb_id = wwpdb_ids[i] if wwpdb_ids is not None else get_fitted_structure_id_from_emdb_entry(emdb_id)
            if wwpdb_id is None:
                logging.warning(f"No fitted structure found for EMDB ID: {emdb_id}, skipping...")
                continue
            density_map_path = cache_dir / "density" / f"emd_{emdb_id}.map"
            structure_path = cache_dir / "structure" / f"{wwpdb_id}.cif"
            if not density_map_path.exists() or overwrite:
                density_map_path = fetch_density_map(emdb_id, cache_dir / "density")
                if density_map_path is None:
                    logging.warning(f"Failed to download density map for EMDB ID: {emdb_id}, skipping...")
                    continue
            if not structure_path.exists() or overwrite:
                structure_path = fetch_structure(wwpdb_id, cache_dir / "structure")
                if structure_path is None:
                    logging.warning(f"Failed to download structure for wwPDB ID: {wwpdb_id}, skipping...")
                    continue

            self.density_list.append(DensityMap(density_map_path, emdb_id=emdb_id))
            self.structure_list.append(Structure(structure_path, wwpdb_id=wwpdb_id))
            success_count += 1

        logger.info(
            f"Fetched {success_count}/{len(emdb_ids)} density maps and structures. Current total: {len(self.density_list)}"
        )

        return self

    def apply_density_transforms(
        self, density_transforms: Iterable[Callable[[DensityMap], DensityMap]]
    ) -> "DensityStructPair":
        """
        Apply a list of density transforms to the density maps.

        Args:
            density_transforms (Iterable[Callable[[DensityMap], DensityMap]]): The list of density transforms to apply.

        Returns:
            DensityStructPair: The DensityStructPair with the applied density transforms.
        """
        for density in self.density_list:
            for func in density_transforms:
                density = func(density)

        self.density_transforms = list(str(func) for func in density_transforms)

        return self

    def save_to_zarr(self, filepath: PathLike) -> PathLike:
        """Save the features to a zarr file."""
        features = [self.__getitem__(i) for i in range(len(self))]
        store_filepath = Path(filepath) / (str(int(datetime.now().timestamp())) + ".zip")
        store = ZipStore(store_filepath, mode="w")

        root = zarr.create_group(store=store)
        meta = root.create_group("meta")
        meta.attrs["schema_version"] = get_git_hash()
        meta.attrs["creation_time"] = str(int(datetime.now().timestamp()))

        logger.info(f"Saving {len(features)} features to {store_filepath}...")

        for i, feature in enumerate(features):
            group = root.create_group(f"items/{i}")

            density_group = group.create_group("density")
            # Explicitly store density volume with compression and pre-determined chunks
            density_group.create_array(
                "volume",
                data=feature["density"]["volume"],
                chunks=(64, 64, 64),
                compressor=BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.bitshuffle),
            )
            for key, value in feature["density"][1:].items():
                density_group.create_array(key, data=value)
            density_group.attrs["transforms"] = self.density_transforms
            density_group.attrs["simulated"] = self.simulated

            structure_group = group.create_group("structure")
            for key, value in feature["structure"].items():
                structure_group.create_array(key, data=value)

            group.attrs["emdb_id"] = feature["emdb_id"]
            group.attrs["wwpdb_id"] = feature["wwpdb_id"]

        logger.info(f"Zarr data saved to {store_filepath}.")

        return store_filepath


class DensityStructPairPatch(DensityStructPair, Dataset):
    """
    Class for patching DensityStructPair for model input.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (64, 64, 64),
        stride: tuple[int, int, int] | None = None,
        crop_atoms_to_patch: bool = True,
        atom_coords_relative: bool = True,
        shuffle_patches: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Initialize DensityStructPairPatch.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.crop_atoms_to_patch = crop_atoms_to_patch
        self.atom_coords_relative = atom_coords_relative
        self.shuffle_patches = shuffle_patches
        self.seed = seed

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the item at the given index."""
        return self._patch(index)

    def _patch(self, index: int):
        """Patch the density and structure volumes."""
        patches = {}
        patch_slice = self._build_patch_slice(index)
        density_vol = self.density_list[index].data
        density_origin = np.array(self.density_list[index].origin)
        density_voxel_size = np.array(self.density_list[index].voxel_size)
        structure_coord = self.structure_list[index].coordinates
        atom_voxel_zyx = world_xyz_to_voxel_zyx(structure_coord, density_origin, density_voxel_size)
        assert isinstance(density_vol, np.ndarray)
        pz, py, px = self.patch_size
        for z0, y0, x0 in patch_slice:
            slc = (slice(z0, z0 + pz), slice(y0, y0 + py), slice(x0, x0 + px))
            density_vol_np = density_vol[slc]  # (pz, py, px) float32
            # add channel dim -> (1, Z, Y, X) for 3D convs
            density_vol_t = torch.from_numpy(density_vol_np[None, ...]).to(torch.float32)
            patches["density"]["volume"] = density_vol_t
            patches["density"]["voxel_size"] = torch.from_numpy(density_voxel_size[None, ...]).to(torch.float32)
            patches["density"]["origin_world"] = torch.from_numpy(density_origin[None, ...]).to(torch.float32)

            if self.crop_atoms_to_patch:
                z, y, x = atom_voxel_zyx[0], atom_voxel_zyx[1], atom_voxel_zyx[2]
                keep = (z >= z0) & (z < z0 + pz) & (y >= y0) & (y < y0 + py) & (x >= x0) & (x < x0 + px)
                idx = np.nonzero(keep)[0]
            else:
                idx = np.arange(atom_voxel_zyx.shape[0])

            atom_voxel_zyx_sel = atom_voxel_zyx[idx]
            if self.atom_coords_relative:
                atom_voxel_zyx_sel = atom_voxel_zyx_sel - np.array([[z0, y0, x0]], dtype=atom_voxel_zyx_sel.dtype)
            patches["structure"]["coord_voxel"] = torch.from_numpy(atom_voxel_zyx_sel).to(torch.float32)

            for key, value in super().__getitem__(index)["structure"].items():
                patches["structure"][key] = torch.from_numpy(value[idx])

        return patches

    def _build_patch_slice(self, index: int):
        """Build a list of (sample_id, z0, y0, x0) patch starts across all items."""
        indices = []
        density = self.density_list[index]
        vol = density.data
        assert isinstance(vol, np.ndarray)
        z, y, x = vol.shape
        pz, py, px = self.patch_size
        sz, sy, sx = self.stride or self.patch_size
        for z0 in range(0, max(z - pz + 1, 1), sz):
            z0 = z - pz if z0 + pz > z else z0
            for y0 in range(0, max(y - py + 1, 1), sy):
                y0 = y - py if y0 + py > y else y0
                for x0 in range(0, max(x - px + 1, 1), sx):
                    x0 = x - px if x0 + px > x else x0
                    indices.append((z0, y0, x0))
        # de-duplicate tail adjustments
        indices = list(dict.fromkeys(indices))
        return indices


class DensityStructPairZarrStream(IterableDataset):
    """
    Stream random 3D crops from zarr files.
    """

    def __init__(
        self,
        zarr_store_paths: list[PathLike],
        patch_size: tuple[int, int, int] = (64, 64, 64),
        stride: tuple[int, int, int] | None = None,
        sample_ids: list[int] | None = None,
        crop_atoms_to_patch: bool = True,
        atom_coords_relative: bool = True,
        shuffle_patches: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize PairZarrStream.
        """
        self.zarr_store_paths = zarr_store_paths
        self.patch_size = patch_size
        self.stride = stride
        self.sample_ids = sample_ids
        self.crop_atoms_to_patch = crop_atoms_to_patch
        self.atom_coords_relative = atom_coords_relative
        self.shuffle_patches = shuffle_patches
        self.seed = seed

    def _open_store(self, store_path: PathLike):
        store = ZipStore(Path(store_path), mode="r")
        root = zarr.open_group(store=store)

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        raise NotImplementedError("__iter__ is not implemented for DensityStructPairZarrStream.")