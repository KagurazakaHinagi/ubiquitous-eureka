import gzip
import logging
import os
import shutil
from pathlib import Path

import requests
import torch


def detect_device() -> torch.device:
    """Detect acceleration device for PyTorch."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Device Autodetected: Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Device Autodetected: Using MPS device.")
    else:
        device = torch.device("cpu")
        logging.info("Device Autodetected: Using CPU device.")

    return device


def download_file(url: str, save_dir: os.PathLike, overwrite: bool = False) -> Path:
    """Download a file from a URL and save it to the specified directory."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        logging.warning(f"Directory {save_dir} does not exist. Creating it...")
        save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / Path(url).name
    if file_path.exists() and not overwrite:
        logging.info(f"File {file_path} already exists. Skipping download.")
        return file_path

    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

    logging.info(f"Downloaded file to {file_path} from {url}")
    return file_path


def uncompress_gz_file(file_path: os.PathLike) -> Path:
    """Uncompress a file and save it to the specified directory."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if file_path.suffix != ".gz":
        raise ValueError(f"File {file_path} is not a gzipped file.")

    with gzip.open(file_path, "rb") as f_in:
        with open(file_path.with_suffix(""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    file_path = file_path.with_suffix("")

    logging.info(f"Uncompressed file to {file_path}")
    return file_path
