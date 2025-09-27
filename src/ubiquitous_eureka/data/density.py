"""
Module for handling 3D density maps.
"""

import logging
import re
from os import PathLike
from pathlib import Path
from typing import Literal

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F

from ubiquitous_eureka.common import detect_device
from ubiquitous_eureka.util.ndimage import gaussian_blur3d_separable, otsu_threshold

logger = logging.getLogger(__name__)


class DensityMap:
    """
    A class to handle 3D density maps, allowing loading from MRC files or directly from
    numpy ndarrays and torch tensors.

    See https://www.ccpem.ac.uk/mrc-format/mrc2014/
    """

    def __init__(
        self,
        filepath: PathLike | None = None,
        emdb_id: str | None = None,
        data: np.ndarray | None = None,  # shape: (z, y, x) or (D, H, W)
        voxel_size: tuple[float, float, float] | None = None,
        origin: tuple[float, float, float] | None = None,  # (x, y, z) Angstroms
        cell_dimensions: tuple[float, float, float] | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """Initialize DensityMap.

        Args:
            filepath (PathLike | None, optional): Path to MRC file to load. Defaults to None.
            emdb_id (str | None, optional): EMDB ID. Defaults to None.
            data (np.ndarray | None, optional): Data for the density map. Defaults to None.
            voxel_size (float, optional): Voxel size in Angstroms. Defaults to 0.5.
            origin (tuple[float, float, float], optional): Origin coordinates. Defaults to (0, 0, 0).
            cell_dimensions (tuple[float, float, float], optional): Cell dimensions in Angstroms. Defaults to (0, 0, 0).
            device (str | torch.device | None, optional): Device to use. Defaults to None for auto-detection.
            dtype (torch.dtype, optional): Data type for torch tensors. Defaults to torch.float32.
        """

        self.device = device or detect_device()
        self.dtype = dtype

        # Data storage
        self._data = None  # Axis order: z, y, x
        self._voxel_size = voxel_size
        self._origin = origin
        self._cell_dimensions = cell_dimensions
        self._emdb_id = emdb_id
        self._metadata = kwargs

        # Calculate nx, ny, nz
        if self._data is not None:
            self._nx = self._data.shape[2]
            self._ny = self._data.shape[1]
            self._nz = self._data.shape[0]

        if filepath is not None:
            self.load(filepath)

        self._validate()

    def _validate(self) -> None:
        """Validate the density data is correctly loaded."""
        if not isinstance(self._data, np.ndarray):
            raise ValueError("No data loaded to validate.")
        if not isinstance(self._voxel_size, tuple) or len(self._voxel_size) != 3:
            raise ValueError("Voxel size is not set or is not a tuple of length 3.")
        if not isinstance(self._origin, tuple) or len(self._origin) != 3:
            raise ValueError("Origin is not set or is not a tuple of length 3.")
        if not isinstance(self._cell_dimensions, tuple) or len(self._cell_dimensions) != 3:
            raise ValueError("Cell dimensions are not set or is not a tuple of length 3.")

    @torch.no_grad()
    def _data_to_tensor(self) -> torch.Tensor:
        """Convert the density map data to a torch tensor with shape (D, H, W) -> (1, 1, D, H, W)."""
        return (
            torch.from_numpy(self._data)
            .to(dtype=self.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, non_blocking=True)
        )

    @torch.no_grad()
    def _data_from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a 5D tensor back to a numpy array with shape (1, 1, D, H, W) -> (D, H, W)."""
        return tensor.squeeze(0).squeeze(0).detach().cpu().numpy()

    @property
    def data(self) -> np.ndarray | None:
        """Get the density map data."""
        return self._data

    @property
    def voxel_size(self) -> tuple[float, float, float] | None:
        """Get the voxel size."""
        return self._voxel_size

    @property
    def origin(self) -> tuple[float, float, float] | None:
        """Get the origin coordinates."""
        return self._origin

    @property
    def emdb_id(self) -> str | None:
        """Get the EMDB ID."""
        return self._emdb_id

    @emdb_id.setter
    def emdb_id(self, emdb_id: str):
        """Set the EMDB ID."""
        if re.match(r"^EMD-\d{4,5}$", emdb_id):
            emdb_id = emdb_id.split("-")[1]
        if not re.match(r"^\d{4,5}$", emdb_id):
            raise ValueError(f"Invalid EMDB ID: {emdb_id}")
        self._emdb_id = emdb_id

    @property
    def metadata(self) -> dict:
        """Get the metadata dictionary."""
        return self._metadata

    def get_world_coordinates(self, voxel_coordinates: tuple[int, int, int]) -> tuple[float, float, float]:
        """Get the world coordinates of a voxel in (x, y, z) Angstroms."""
        self._validate()

        assert self._voxel_size is not None
        assert self._origin is not None
        return (
            voxel_coordinates[2] * self._voxel_size[2] + self._origin[2],
            voxel_coordinates[1] * self._voxel_size[1] + self._origin[1],
            voxel_coordinates[0] * self._voxel_size[0] + self._origin[0],
        )

    def load(self, filepath: PathLike) -> "DensityMap":
        """
        Load density map from an MRC file.

        Args:
            filepath (PathLike): Path to the MRC file.

        Returns:
            DensityMap: An instance of DensityMap containing the loaded data.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        with mrcfile.open(str(filepath), mode="r") as f:
            if f.data is None or f.header is None:
                raise ValueError(f"Density map corrupted or missing in MRC file {filepath}.")

            self._data = f.data.copy().astype(np.float32)

            self._cell_dimensions = (
                float(f.header.cella.x),
                float(f.header.cella.y),
                float(f.header.cella.z),
            )
            self._voxel_size = (float(f.voxel_size.x), float(f.voxel_size.y), float(f.voxel_size.z))
            self._origin = (float(f.header.origin.x), float(f.header.origin.y), float(f.header.origin.z))
            self._nx = int(f.header.nx)
            self._ny = int(f.header.ny)
            self._nz = int(f.header.nz)

            # Store additional metadata
            self._metadata = {
                "filepath": str(filepath),
                "original_shape": tuple(f.data.shape),
                "original_dtype": str(f.data.dtype),
                "space_group": int(f.header.ispg),
                "map_label": str(f.header.label[0]),
            }

            logger.info(f"Loaded density map from {filepath}.")

            return self

    def save(self, filepath: PathLike, overwrite: bool = True) -> "DensityMap":
        """
        Save the density map to an MRC file.

        Args:
            filepath (PathLike): Path to save the MRC file.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.

        Returns:
            DensityMap: The current instance for chaining.
        """
        self._validate()

        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filepath} already exists and overwrite is set to False.")

        with mrcfile.new(str(filepath), overwrite=overwrite) as f:
            assert f.header is not None, "MRC file header is missing."
            f.set_data(self._data)
            f.voxel_size = self._voxel_size

            # Set origins in header
            assert self._origin is not None
            f.header.origin.x = np.float32(self._origin[0])
            f.header.origin.y = np.float32(self._origin[1])
            f.header.origin.z = np.float32(self._origin[2])

            # Set cell dimensions
            assert self._cell_dimensions is not None
            f.header.cella.x = np.float32(self._cell_dimensions[0])
            f.header.cella.y = np.float32(self._cell_dimensions[1])
            f.header.cella.z = np.float32(self._cell_dimensions[2])

            # Update header statistics
            f.update_header_from_data()
            f.update_header_stats()

        logger.info(f"Saved density map to {filepath}.")

        return self

    @torch.no_grad()
    def resample(self, target_voxel_size: float | tuple[float, float, float], mode: str = "trilinear") -> "DensityMap":
        """
        Resample the density map to the target voxel size.

        Args:
            target_voxel_size (float): Target voxel size in Angstroms.
            mode (str, optional): Interpolation mode ('nearest', 'trilinear', 'area'). Defaults to 'trilinear'.

        Returns:
            DensityMap: The resampled density map.
        """
        self._validate()

        if isinstance(target_voxel_size, (int, float)):
            target_voxel_size = (target_voxel_size, target_voxel_size, target_voxel_size)

        # Compute output grid size
        assert self._cell_dimensions is not None
        new_nx = max(1, int(round(self._cell_dimensions[0] / target_voxel_size[0])))
        new_ny = max(1, int(round(self._cell_dimensions[1] / target_voxel_size[1])))
        new_nz = max(1, int(round(self._cell_dimensions[2] / target_voxel_size[2])))

        logger.info(
            f"Resampling from voxel size {self._voxel_size:.3f}Å to {target_voxel_size:.3f}Å "
            f"with new shape: {(new_nx, new_ny, new_nz)} "
            f"using {mode} interpolation on {self.device}."
        )

        # Prepare torch tensor for GPU-accelerated interpolation
        data_5d = self._data_to_tensor()

        resampled_5d = F.interpolate(
            data_5d,
            size=(new_nz, new_ny, new_nx),
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )

        self._data = self._data_from_tensor(resampled_5d)

        # Update data attributes
        self._voxel_size = target_voxel_size
        self._nx = new_nx
        self._ny = new_ny
        self._nz = new_nz
        self._cell_dimensions = (
            target_voxel_size[0] * new_nx,
            target_voxel_size[1] * new_ny,
            target_voxel_size[2] * new_nz,
        )  # Recompute cell dimensions to match new shape

        self._validate()
        logger.info(f"Resampling complete. New voxel size: {self._voxel_size:.3f}Å, new shape: {self._data.shape}")

        return self

    @torch.no_grad()
    def foreground_mask(
        self,
        t: torch.Tensor,
        method: Literal["otsu", "quantile"] = "otsu",
        sigma: tuple[float, float, float] = (1.0, 1.0, 1.0),
        quantile: float = 0.75,
        min_positive_voxels: int = 1024,
    ) -> torch.Tensor:
        """
        Create a foreground mask based on the density map data.

        Args:
            method (Literal["otsu", "quantile"], optional): Method to use for thresholding. Defaults to "otsu".
            sigma (tuple[float, float, float], optional): Sigma for Gaussian blur. Defaults to (1.0, 1.0, 1.0).
            quantile (float, optional): Quantile for quantile thresholding. Defaults to 0.75.
            min_positive_voxels (int, optional): Minimum number of positive voxels to consider. Defaults to 1024.

        Returns:
            Foreground mask.
        """
        device, dtype = t.device, t.dtype

        # Smooth (help separate background solvent from foreground signal)
        smoothed_t = gaussian_blur3d_separable(t, sigmas=sigma)

        # Base positive region
        base_positive_t = smoothed_t > 0

        # Fallback if almost nothing is positive
        if int(base_positive_t.sum().item()) < min_positive_voxels:
            # Fallback to global stats: mean + std
            mean, std = smoothed_t.mean(), smoothed_t.std()
            threshold = mean + std
            return smoothed_t > threshold

        positive_values = smoothed_t[base_positive_t]

        # Method-specific thresholding
        if method == "otsu":
            threshold = otsu_threshold(positive_values)
            threshold = torch.maximum(threshold, torch.tensor(0.0, dtype=dtype, device=device))
        elif method == "quantile":
            threshold = torch.quantile(positive_values, quantile)
            threshold = torch.maximum(threshold, torch.tensor(0.0, dtype=dtype, device=device))
        else:
            raise ValueError(f"Invalid method: {method}")

        logger.debug(f"Foreground mask created with threshold: {threshold:.3f}.")

        return smoothed_t > threshold

    @torch.no_grad()
    def _normalize_tensor_robust_zscore(
        self,
        t: torch.Tensor,
        clip: float = 3.0,
        eps: float = 1e-8,
        foreground_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Normalize a tensor using robust Z-score masking.

        Args:
            t (torch.Tensor): Tensor to normalize.
            clip (float, optional): Clip the normalized values to this value. Defaults to 3.0.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            foreground_mask (torch.Tensor | None, optional): Foreground mask. Defaults to None.

        Returns:
            Normalized tensor.
        """
        vals = t[foreground_mask] if foreground_mask is not None else t
        med = vals.median()
        mad = (vals - med).abs().median()
        rstd = 1.4826 * mad
        tz = (t - med) / (rstd + eps)
        if clip is not None:
            tz = torch.clamp(tz, -clip, clip)
            tz = tz / clip
        logger.debug(f"Normalized tensor using Z-score masking with clip: {clip:.3f}.")
        return tz

    @torch.no_grad()
    def _normalize_tensor_zscore(
        self,
        t: torch.Tensor,
        clip: float = 3.0,
        eps: float = 1e-8,
        foreground_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Normalize a tensor using Z-score masking.

        Args:
            t (torch.Tensor): Tensor to normalize.
            clip (float, optional): Clip the normalized values to this value. Defaults to 3.0.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            foreground_mask (torch.Tensor | None, optional): Foreground mask. Defaults to None.

        Returns:
            Normalized tensor.
        """
        vals = t[foreground_mask] if foreground_mask is not None else t
        mean = vals.mean()
        std = vals.std()
        tz = (t - mean) / (std + eps)
        if clip is not None:
            tz = torch.clamp(tz, -clip, clip)
            tz = tz / clip
        logger.debug(f"Normalized tensor using Z-score masking with clip: {clip:.3f}.")
        return tz

    @torch.no_grad()
    def _normalize_tensor_percentile(
        self,
        t: torch.Tensor,
        percentile_range: tuple[float, float] = (1, 99),
        clip: float = 1.0,
        eps: float = 1e-8,
        foreground_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Normalize a tensor using percentile masking.

        Args:
            t (torch.Tensor): Tensor to normalize.
            percentile_range (tuple[float, float], optional): Percentiles for normalization. Defaults to (1, 99).
            clip (float, optional): Clip the normalized values to this value. Defaults to 1.0.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            foreground_mask (torch.Tensor | None, optional): Foreground mask. Defaults to None.

        Returns:
            Normalized tensor.
        """
        vals = t[foreground_mask] if foreground_mask is not None else t
        lower_bound = torch.quantile(vals, percentile_range[0] / 100.0)
        upper_bound = torch.quantile(vals, percentile_range[1] / 100.0)
        if clip is not None:
            t = torch.clamp(t, -clip, clip)
        t_norm = 2.0 * (t - lower_bound) / (upper_bound - lower_bound + eps) - 1.0
        logger.debug(f"Normalized tensor using percentile masking with clip: {clip:.3f}.")
        return t_norm

    @torch.no_grad()
    def normalize(
        self,
        mode: Literal["robust_zscore", "zscore", "percentile"] = "robust_zscore",
        foreground_mask: Literal["otsu", "quantile", "none"] = "otsu",
        percentile_range: tuple[float, float] = (1, 99),
    ) -> "DensityMap":
        """
        Normalize the density map data to the range [-1, 1] based on specified mode.

        Args:
            mode (Literal["robust_zscore", "zscore", "percentile"], optional): Method to use for normalization. Defaults to "robust_zscore".
            foreground_mask (Literal["otsu", "quantile", "none"], optional): Whether to generate and use a foreground mask. Defaults to "otsu".
            percentile_range (tuple[float, float], optional): Percentiles for normalization if mode is "percentile". Defaults to (1, 99).

        Returns:
            DensityMap: The normalized density map.
        """
        self._validate()

        data_3d = torch.from_numpy(self._data).to(device=self.device, dtype=self.dtype)
        mask = self.foreground_mask(data_3d, method=foreground_mask) if foreground_mask != "none" else None

        if mode == "robust_zscore":
            normalized_3d = self._normalize_tensor_robust_zscore(data_3d, foreground_mask=mask)
        elif mode == "zscore":
            normalized_3d = self._normalize_tensor_zscore(data_3d, foreground_mask=mask)
        elif mode == "percentile":
            normalized_3d = self._normalize_tensor_percentile(
                data_3d, foreground_mask=mask, percentile_range=percentile_range
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self._data = normalized_3d.detach().cpu().numpy()

        logger.info(
            f"Normalized density map using {mode} mode with foreground mask: {foreground_mask} and percentile range: {percentile_range}."
        )

        return self

    def crop(self, start: tuple[int, int, int], end: tuple[int, int, int]) -> "DensityMap":
        """
        Crop the density map to the specified start and end indices.

        Args:
            start (tuple[int, int, int]): Starting indices (z, y, x).
            end (tuple[int, int, int]): Ending indices (z, y, x).

        Returns:
            DensityMap: The cropped density map.
        """
        self._validate()

        # Validate indices
        start = (max(0, start[0]), max(0, start[1]), max(0, start[2]))
        end = (
            min(self._nz, end[0]),
            min(self._ny, end[1]),
            min(self._nx, end[2]),
        )

        assert self._data is not None
        self._data = self._data[start[0] : end[0], start[1] : end[1], start[2] : end[2]]

        assert self._voxel_size is not None
        assert self._origin is not None
        origin_shift = tuple(start[i] * self._voxel_size[i] for i in range(3))[::-1]
        self._origin = (
            self._origin[0] + origin_shift[0],
            self._origin[1] + origin_shift[1],
            self._origin[2] + origin_shift[2],
        )

        logger.info(f"Cropped density map to region: {start} - {end}.")
        return self
