import gzip
import logging
import os
import shutil
import subprocess
from pathlib import Path

import requests
import torch


def get_git_hash() -> str:
    """Get the git hash of the current repository."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


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


def download_file(url: str, save_dir: os.PathLike, overwrite: bool = False) -> Path | None:
    """Download a file from a URL and save it to the specified directory."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        logging.warning(f"Directory {save_dir} does not exist. Creating it...")
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        file_path = save_dir / Path(url).name
        if file_path.exists() and not overwrite:
            logging.info(f"File {file_path} already exists. Skipping download.")
            return file_path

        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"Failed to download file {url}: {response.status_code}")
            return None

        with open(file_path, "wb") as f:
            f.write(response.content)

    except Exception as e:
        logging.error(f"Error downloading file {url}: {e}")
        return None

    logging.info(f"Downloaded file to {file_path} from {url}")
    return file_path


def uncompress_gz_file(file_path: os.PathLike, keep_original: bool = False) -> Path:
    """Uncompress a file and save it to the specified directory."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if file_path.suffix != ".gz":
        raise ValueError(f"File {file_path} is not a gzipped file.")

    try:
        with gzip.open(file_path, "rb") as f_in:
            with open(file_path.with_suffix(""), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        if not keep_original:
            os.remove(file_path)
        file_path = file_path.with_suffix("")

    except Exception as e:
        logging.error(f"Error uncompressing file {file_path}: {e}")
        raise e

    logging.info(f"Uncompressed file to {file_path}")
    return file_path
