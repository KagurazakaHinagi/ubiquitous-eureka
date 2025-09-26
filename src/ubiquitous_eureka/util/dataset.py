"""
Module for creating datasets for given EMDB and wwPDB IDs.
"""

import logging
import os
import re
from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import requests

from ubiquitous_eureka.common import download_file, uncompress_gz_file
from ubiquitous_eureka.data.density import DensityMap
from ubiquitous_eureka.data.structure import Structure

logger = logging.getLogger(__name__)


def fetch_density_map(emdb_id: str, save_dir: PathLike) -> Path:
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

    # Uncompress the density map
    if file_path.suffix == ".gz":
        file_path = uncompress_gz_file(file_path)

    logger.info(f"Saved density map to {file_path}")
    return file_path


def fetch_structure(wwpdb_id: str, save_dir: PathLike) -> Path:
    """Fetch a structure from wwPDB and save it to the specified directory."""

    if not re.match(r"^[0-9A-Za-z]{4}$", wwpdb_id):
        raise ValueError(f"Invalid wwPDB ID: {wwpdb_id}")

    save_dir = Path(save_dir)

    file_url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{wwpdb_id[1:2]}/{wwpdb_id}.cif.gz"
    file_path = save_dir / f"{wwpdb_id}.cif.gz"

    # Download the structure from wwPDB
    logger.info(f"Downloading structure from {file_url} to {file_path}...")
    file_path = download_file(file_url, save_dir)

    # Uncompress the structure
    if file_path.suffix == ".gz":
        file_path = uncompress_gz_file(file_path)

    logger.info(f"Saved structure to {file_path}")
    return file_path


def get_fitted_structure_from_emdb_entry(emdb_id: str) -> str:
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
        return "N/A"

    return response.json()["crossreferences"]["pdb_list"]["pdb_reference"][0]["pdb_id"]


class DensityStructDataset:
    """Class for creating datasets from EMDB and wwPDB IDs."""

    def __init__(
        self,
        emdb_ids: list[str],
        save_dir: PathLike,
        wwpdb_ids: list[str] | None = None,
        overwrite: bool = False,
    ):
        """Initialize Dataset."""
        self.emdb_ids = emdb_ids
        self.save_dir = save_dir
        self.overwrite = overwrite

        self.density = None
        self.structure = None

        self.valid_indices = None

        self.wwpdb_ids = wwpdb_ids or [get_fitted_structure_from_emdb_entry(emdb_id) for emdb_id in self.emdb_ids]
        if len(self.wwpdb_ids) != len(self.emdb_ids):
            raise ValueError(
                f"Number of wwPDB IDs must be the same as the number of EMDB IDs. Got {len(self.wwpdb_ids)} wwPDB IDs and {len(self.emdb_ids)} EMDB IDs."
            )

    def fetch(self):
        """Download the data from the EMDB and wwPDB IDs."""
        for emdb_id in self.emdb_ids:
            fetch_density_map(emdb_id, self.save_dir)
        for wwpdb_id in self.wwpdb_ids:
            fetch_structure(wwpdb_id, self.save_dir)

    def process_files(self) -> "DensityStructDataset":
        """Process the data from the EMDB and wwPDB IDs."""
        self.density = []
        self.structure = []
        self.valid_indices = []

        for i, emdb_id in enumerate(self.emdb_ids):
            wwpdb_id = self.wwpdb_ids[self.emdb_ids.index(emdb_id)]
            density_map_path = Path(self.save_dir) / f"emd_{emdb_id}.map"
            structure_path = Path(self.save_dir) / f"{wwpdb_id}.cif"
            if not density_map_path.exists():
                logging.warning(f"Density map file not found: {density_map_path}, skipping...")
                continue
            if not structure_path.exists():
                logging.warning(f"Structure file not found: {structure_path}, skipping...")
                continue

            self.density.append(DensityMap(density_map_path))
            self.structure.append(Structure(structure_path))
            self.valid_indices.append(i)

        return self

    def data_to_hdf5(self, filepath: PathLike) -> PathLike:
        """Save the data to an HDF5 file."""
        raise NotImplementedError("Not implemented yet.")


class SimulatedDensityStructDataset:
    """Class for creating datasets from simulated density maps and structures."""
