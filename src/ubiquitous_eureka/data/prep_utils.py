"""
Module for preparing data from EMDB and wwPDB IDs.
"""

import logging
import re
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import requests
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ZipStore

from ubiquitous_eureka.common import download_file, get_git_hash, uncompress_gz_file
from ubiquitous_eureka.data.density import DensityMap
from ubiquitous_eureka.data.structure import Structure

logger = logging.getLogger(__name__)


def fetch_density_map(emdb_id: str, save_dir: PathLike) -> Path | None:
    """Fetch a density map from EMDB and save it to the specified directory."""

    if re.match(r"^EMD-\d{4,5}$", emdb_id):
        emdb_id = emdb_id.split("-")[1]
    if not re.match(r"^\d{4,5}$", emdb_id):
        raise ValueError(f"Invalid EMDB ID: {emdb_id}")

    save_dir = Path(save_dir)

    file_url = f"https://files.wwpdb.org/pub/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz"
    file_path = save_dir / f"emd_{emdb_id}.map.gz"

    # Download the density map from EMDB
    logger.info(f"Downloading density map from {file_url} to {file_path}...")
    file_path = download_file(file_url, save_dir)

    if file_path is None:
        return None

    # Uncompress the density map
    if file_path.suffix == ".gz":
        file_path = uncompress_gz_file(file_path)

    logger.info(f"Saved density map to {file_path}")
    return file_path


def fetch_structure(wwpdb_id: str, save_dir: PathLike) -> Path | None:
    """Fetch a structure from wwPDB and save it to the specified directory."""

    if not re.match(r"^[0-9A-Za-z]{4}$", wwpdb_id):
        raise ValueError(f"Invalid wwPDB ID: {wwpdb_id}")

    save_dir = Path(save_dir)

    file_url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{wwpdb_id[1:2]}/{wwpdb_id}.cif.gz"
    file_path = save_dir / f"{wwpdb_id}.cif.gz"

    # Download the structure from wwPDB
    logger.info(f"Downloading structure from {file_url} to {file_path}...")
    file_path = download_file(file_url, save_dir)

    if file_path is None:
        return None

    # Uncompress the structure
    if file_path.suffix == ".gz":
        file_path = uncompress_gz_file(file_path)

    logger.info(f"Saved structure to {file_path}")
    return file_path


def get_fitted_structure_id_from_emdb_entry(emdb_id: str) -> str | None:
    """Fetch a fitted structure wwPDB id from EMDB and save it to the specified directory."""

    if re.match(r"^EMD-\d{4,5}$", emdb_id):
        emdb_id = emdb_id.split("-")[1]
    if not re.match(r"^\d{4,5}$", emdb_id):
        raise ValueError(f"Invalid EMDB ID: {emdb_id}")

    request_url = f"https://www.ebi.ac.uk/emdb/api/entry/fitted/{emdb_id}"
    response = requests.get(request_url)
    response.raise_for_status()

    if "pdb_list" not in response.json()["crossreferences"]:
        logging.warning(f"No fitted structure found for EMDB ID: {emdb_id}")
        return None

    return response.json()["crossreferences"]["pdb_list"]["pdb_reference"][0]["pdb_id"]


class DensityStructPair:
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

        logger.info(f"Fetched {success_count}/{len(emdb_ids)} density maps and structures. Current total: {len(self.density_list)}")

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

    def _extract_features(self) -> list[dict[str, Any]]:
        """Extract features from the density and structure pairs."""
        features = []

        for density, structure in zip(self.density_list, self.structure_list):
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

            features.append(
                {
                    "density": density_feature,
                    "structure": structure_feature,
                    "emdb_id": emdb_id,
                    "wwpdb_id": wwpdb_id,
                }
            )

        return features

    def save_to_zarr(self, filepath: PathLike) -> PathLike:
        """Save the features to a zarr file."""
        features = self._extract_features()
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
