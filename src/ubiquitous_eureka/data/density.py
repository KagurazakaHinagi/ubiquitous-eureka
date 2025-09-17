"""
Module for handling 3D density maps.
"""

import copy
from pathlib import Path

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F


class DensityMap:
    """
    A class to handle 3D density maps, allowing loading from MRC files or directly from
    numpy ndarrays and torch tensors.
    """

    def __init__(
        self,
        filepath: str | None = None,
        data: np.ndarray | torch.Tensor | None = None,
        voxel_size: float = 0.5,
        origin: tuple[float, float, float] = (0, 0, 0),
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize DensityMap.

        Args:
            filepath (str | None, optional): Path to MRC file to load. Defaults to None.
            data (np.ndarray | torch.Tensor | None, optional): Data for the density map. Defaults to None.
            voxel_size (float, optional): Voxel size in Angstroms. Defaults to 0.5.
            origin (tuple[float, float, float], optional): Origin coordinates. Defaults to (0, 0, 0).
            device (str | torch.device | None, optional): Device to use. Defaults to None for auto-detection.
            dtype (torch.dtype, optional): Data type for torch tensors. Defaults to torch.float32.
        """

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = dtype

        # Data storage
        self._data = None
        self._original_data = None
        self._voxel_size = voxel_size
        self._origin = origin
        self._metadata = {}

        if filepath is not None:
            self.load(filepath)
        elif data is not None:
            self.set_data(data, voxel_size, origin)

    def to(self, device: str | torch.device) -> "DensityMap":
        """Move data to specified device."""
        if isinstance(device, str):
            device = torch.device(device)

        if device != self.device:
            print(f"Moving data from {self.device} to {device}")

            if self._data is not None:
                self._data = self._data.to(device)
            if self._original_data is not None:
                self._original_data = self._original_data.to(device)

            self.device = device

        return self

    @property
    def data(self) -> torch.Tensor | None:
        """Get the density map data."""
        return self._data

    @property
    def numpy(self) -> np.ndarray | None:
        """Get the density map data as a numpy array."""
        return self._data.detach().cpu().numpy() if self._data is not None else None

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Get the shape of the density map data."""
        return self._data.shape if self._data is not None else None

    @property
    def voxel_size(self) -> float:
        """Get the voxel size."""
        return self._voxel_size

    @property
    def origin(self) -> tuple[float, float, float]:
        """Get the origin coordinates."""
        return self._origin

    @property
    def metadata(self) -> dict:
        """Get the metadata dictionary."""
        return self._metadata

    def __repr__(self) -> str:
        if not isinstance(self._data, torch.Tensor):
            return f"DensityMap(empty, device={self.device})"
        return (
            f"DensityMap(shape={self.shape}, voxel_size={self.voxel_size:.3f}Å, "
            f"origin={self.origin}, range=[{self._data.min():.3f}, {self._data.max():.3f}], "
            f"device={self.device})"
        )

    def load(self, filepath: str | Path) -> "DensityMap":
        """
        Load density map from an MRC file.

        Args:
            filepath (str | Path): Path to the MRC file.

        Returns:
            DensityMap: An instance of DensityMap containing the loaded data.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        with mrcfile.open(str(filepath), mode="r") as f:
            if f.data is None:
                raise ValueError(f"No data found in MRC file {filepath}.")

            # Load to cpu first to avoid GPU memory issues
            data_np = f.data.copy().astype(np.float32)
            self._data = torch.from_numpy(data_np).to(device=self.device, dtype=self.dtype)
            self._original_data = self._data.clone()

            # Get voxel size
            if f.voxel_size is None:
                raise ValueError(f"Voxel size not found in MRC file {filepath}.")
            if f.voxel_size.x != f.voxel_size.y or f.voxel_size.x != f.voxel_size.z:
                raise ValueError("Anisotropic voxel sizes are not supported.")
            self._voxel_size = float(f.voxel_size.x)

            # Get origin coordinates
            if f.header and hasattr(f.header, "origin"):
                self._origin = (
                    float(f.header.origin.x),
                    float(f.header.origin.y),
                    float(f.header.origin.z),
                )
            else:
                self._origin = (0.0, 0.0, 0.0)

            # Store additional metadata
            self._metadata = {
                "filepath": str(filepath),
                "original_shape": tuple(data_np.shape),
                "original_dtype": str(f.data.dtype),
                "cell_dimensions": (
                    float(f.header.cella.x),
                    float(f.header.cella.y),
                    float(f.header.cella.z),
                )
                if f.header and hasattr(f.header, "cella")
                else None,
                "space_group": int(f.header.ispg) if f.header and hasattr(f.header, "ispg") else 1,
                "map_label": str(f.header.label[0]) if f.header and hasattr(f.header, "label") else "MAP",
            }

            assert isinstance(self._data, torch.Tensor), "Data is not a torch tensor."
            print(f"Loaded: {self}")

            return self

    def save(self, filepath: str | Path, overwrite: bool = True) -> "DensityMap":
        """
        Save the density map to an MRC file.

        Args:
            filepath (str | Path): Path to save the MRC file.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.

        Returns:
            DensityMap: The current instance for chaining.
        """
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("No data to save.")

        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filepath} already exists and overwrite is set to False.")

        # Convert data to numpy for saving
        data_np = self.numpy

        with mrcfile.new(str(filepath), overwrite=overwrite) as f:
            assert f.header is not None, "MRC file header is missing."
            f.set_data(data_np)
            f.voxel_size = self._voxel_size

            # Set origins in header
            f.header.origin.x = self._origin[0]
            f.header.origin.y = self._origin[1]
            f.header.origin.z = self._origin[2]

            # Update header statistics
            f.update_header_from_data()
            f.update_header_stats()

        print(f"Saved density map to {filepath}")

        return self

    def set_data(
        self,
        data: np.ndarray | torch.Tensor,
        voxel_size: float | None = None,
        origin: tuple[float, float, float] | None = None,
    ) -> "DensityMap":
        """
        Set the density map data directly from a numpy array or torch tensor.

        Args:
            data (np.ndarray | torch.Tensor): The density map data.
            voxel_size (float | None, optional): Voxel size in Angstroms. If None, keeps existing. Defaults to None.
            origin (tuple[float, float, float] | None, optional): Origin coordinates. If None, keeps existing. Defaults to None.
        Returns:
            DensityMap: The current instance for chaining.
        """
        if isinstance(data, np.ndarray):
            self._data = torch.from_numpy(data.astype(np.float32)).to(device=self.device, dtype=self.dtype)
        else:
            self._data = data.to(device=self.device, dtype=self.dtype)
        self._original_data = self._data.clone()

        if voxel_size is not None:
            self._voxel_size = voxel_size
        if origin is not None:
            self._origin = origin

        return self

    def reset(self) -> "DensityMap":
        """Reset the density map to its original state."""
        if not isinstance(self._original_data, torch.Tensor):
            print("No original data to reset to.")
            return self

        self._data = self._original_data.clone()
        print("Density map has been reset to its original state.")
        return self

    def copy(self) -> "DensityMap":
        """Create a deep copy of the density map."""
        new_map = DensityMap(device=self.device, dtype=self.dtype)
        new_map._data = self._data.clone() if self._data is not None else None
        new_map._original_data = self._original_data.clone() if self._original_data is not None else None
        new_map._voxel_size = self._voxel_size
        new_map._origin = self._origin
        new_map._metadata = copy.deepcopy(self._metadata)

        return new_map

    def resample(self, target_voxel_size: float, mode: str = "trilinear") -> "DensityMap":
        """
        Resample the density map to the target voxel size.

        Args:
            target_voxel_size (float): Target voxel size in Angstroms.
            mode (str, optional): Interpolation mode ('nearest', 'trilinear', 'area'). Defaults to 'trilinear'.

        Returns:
            DensityMap: The resampled density map.
        """
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("No data loaded to resample.")
        if target_voxel_size <= 0:
            raise ValueError("Target voxel size must be positive.")

        zoom_factor = self._voxel_size / target_voxel_size
        new_size = [int(dim * zoom_factor) for dim in self._data.shape]

        print(
            f"Resampling from voxel size {self._voxel_size:.3f}Å to {target_voxel_size:.3f}Å "
            f"with zoom factor {zoom_factor:.3f}, new shape: {new_size} "
            f"using {mode} interpolation on {self.device}."
        )
        print(f"Shape: {self._data.shape} -> {new_size}")

        # Add batch and channel dimensions
        data_5d = self._data.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            resampled_5d = F.interpolate(
                data_5d,
                size=new_size,
                mode=mode,
                align_corners=False if mode == "trilinear" else None,
            )

        # Remove batch and channel dimensions
        self._data = resampled_5d.squeeze(0).squeeze(0)
        self._voxel_size = target_voxel_size

        print(f"Resampling complete. New voxel size: {self._voxel_size:.3f}Å, new shape: {self._data.shape}")
        return self

    def normalize(self, percentile_range: tuple[float, float] = (5, 95)) -> "DensityMap":
        """
        Normalize the density map data to the range [0, 1] based on specified percentiles.

        Args:
            percentile_range (tuple[float, float], optional): Percentiles for normalization. Defaults to (5, 95).

        Returns:
            DensityMap: The normalized density map.
        """
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("No data loaded to normalize.")
        if not (0 <= percentile_range[0] < percentile_range[1] <= 100):
            raise ValueError("Percentile range must be between 0 and 100 and min < max.")

        with torch.no_grad():
            lower_bound = torch.quantile(self._data, percentile_range[0] / 100.0)
            upper_bound = torch.quantile(self._data, percentile_range[1] / 100.0)

            clipped_data = torch.clamp(self._data, min=lower_bound, max=upper_bound)

            if upper_bound == lower_bound:
                print("Warning: Upper and lower bounds are equal. Normalization will result in zero data.")
                self._data.fill_(0.5)  # Set to mid-range if no variation
            else:
                self._data = (clipped_data - lower_bound) / (upper_bound - lower_bound)

        print(
            f"Normalization complete using percentiles {percentile_range}. "
            f"Data range is now [{self._data.min():.3f}, {self._data.max():.3f}]."
        )
        return self

    def gaussian_filter(self, sigma: float) -> "DensityMap":
        """
        Apply a Gaussian filter to the density map.

        Args:
            sigma (float): Standard deviation for Gaussian kernel in voxel units.

        Returns:
            DensityMap: The filtered density map.
        """
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("No data loaded to apply Gaussian filter.")
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

        with torch.no_grad():
            # Create 3D Gaussian kernel
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure kernel size is odd

            # Create 1D Gaussian kernel
            x = torch.arange(kernel_size, dtype=self.dtype, device=self.device) - kernel_size // 2
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()

            # Create 3D kernel by outer product
            kernel_3d = kernel_1d.view(1, 1, 1, 1, -1) * kernel_1d.view(1, 1, 1, -1, 1) * kernel_1d.view(1, 1, -1, 1, 1)

            # Add batch and channel dimensions
            data_5d = self._data.unsqueeze(0).unsqueeze(0)

            # Apply convolution with padding
            padding = kernel_size // 2
            filtered_5d = F.conv3d(data_5d, kernel_3d, padding=padding)

            # Remove batch and channel dimensions
            self._data = filtered_5d.squeeze(0).squeeze(0)

        print(f"Applied Gaussian filter with sigma={sigma} voxels.")
        return self

    def threshold(self, threshold_value: float, below_value: float = 0.0) -> "DensityMap":
        """
        Apply a threshold to the density map, setting values below the threshold to a specified value.

        Args:
            threshold_value (float): The threshold value.
            below_value (float, optional): Value to set for voxels below the threshold. Defaults to 0.0.

        Returns:
            DensityMap: The thresholded density map.
        """
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("No data loaded to apply threshold.")

        with torch.no_grad():
            mask = self._data < threshold_value
            original_below_count = mask.sum().item()

            self._data[mask] = below_value

        print(
            f"Applied threshold at {threshold_value}. "
            f"Set {original_below_count} voxels below threshold to {below_value}."
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
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("No data loaded to crop.")

        # Validate indices
        start = (max(0, start[0]), max(0, start[1]), max(0, start[2]))
        end = (
            min(self._data.shape[0], end[0]),
            min(self._data.shape[1], end[1]),
            min(self._data.shape[2], end[2]),
        )

        with torch.no_grad():
            self._data = self._data[start[0] : end[0], start[1] : end[1], start[2] : end[2]]

        # Update origin based on cropping
        origin_shift = tuple(start[i] * self._voxel_size for i in range(3))
        self._origin = (
            self._origin[0] + origin_shift[0],
            self._origin[1] + origin_shift[1],
            self._origin[2] + origin_shift[2],
        )

        print(f"Cropped density map to region: {start} - {end}.")
        return self

    def pad(
        self,
        padding: int | tuple[int, int, int],
        mode: str = "constant",
        value: float = 0.0,
    ) -> "DensityMap":
        """
        Pad the density map with specified padding and mode.

        Args:
            padding (int | tuple[int, int, int]): Amount of padding. If int, pads all sides equally.
            mode (str, optional): Padding mode ('constant', 'reflect', 'replicate', 'circular'). Defaults to "constant".
            value (float, optional): Value to use for 'constant' padding. Defaults to 0.0.

        Returns:
            DensityMap: The padded density map.
        """
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("No data loaded to pad.")

        if isinstance(padding, int):
            # (left, right, top, bottom, front, back)
            pad_tuple = (padding, padding, padding, padding, padding, padding)
        else:
            # (pad_z, pad_y, pad_x)
            pad_tuple = (
                padding[2],
                padding[2],
                padding[1],
                padding[1],
                padding[0],
                padding[0],
            )

        with torch.no_grad():
            self._data = (
                F.pad(
                    self._data.unsqueeze(0).unsqueeze(0),
                    pad_tuple,
                    mode=mode,
                    value=value,
                )
                .squeeze(0)
                .squeeze(0)
            )

        # Update origin based on padding
        if isinstance(padding, int):
            origin_shift = (
                padding * self._voxel_size,
                padding * self._voxel_size,
                padding * self._voxel_size,
            )
        else:
            origin_shift = (
                padding[0] * self._voxel_size,
                padding[1] * self._voxel_size,
                padding[2] * self._voxel_size,
            )
        self._origin = (
            self._origin[0] - origin_shift[0],
            self._origin[1] - origin_shift[1],
            self._origin[2] - origin_shift[2],
        )

        print(f"Padded density map with {pad_tuple} using mode '{mode}'.")
        return self
