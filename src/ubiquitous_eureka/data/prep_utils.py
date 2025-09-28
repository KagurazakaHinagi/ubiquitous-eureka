"""
Module for preparing data from EMDB and wwPDB IDs.
"""

import logging
import re
from os import PathLike
from pathlib import Path

import requests

from ubiquitous_eureka.common import download_file, uncompress_gz_file

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
