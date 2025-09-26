"""
Module for multi-dimensional image processing. Optimized for 3D volumes.
"""

import torch
from torch.nn import functional as F
from math import ceil

@torch.no_grad()
def _gaussian_kernel1d(
    sigma: float,
    truncate: float = 3.0,
    radius: int | None = None,
    dtype = torch.float32,
    device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
    """
    1D Gaussian kernel.

    Args:
        sigma: Standard deviation for Gaussian kernel in voxel units.
        truncate: Truncate the kernel at this many standard deviations.
        radius: Radius of the kernel. If None, calculated from truncate and sigma.
        dtype: Data type for the kernel.
        device: Device for the kernel.

    Returns:
        1D Gaussian kernel.
    """
    if sigma <= 0:
        # delta kernel
        k = torch.tensor([1.0], dtype=dtype, device=device)
        return k / k.sum()
    radius = int(ceil(truncate * sigma)) if radius is None else radius
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-(x**2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k

@torch.no_grad()
def gaussian_blur3d_separable(vol: torch.Tensor, sigmas: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Apply a Gaussian blur to a 3D volume using separable 1D kernels.

    Args:
        vol: (D, H, W) or (N, D, H, W) float tensor on any device
        sigmas: (sd, sh, sw)

    Returns:
        (D, H, W) or (N, D, H, W) float tensor on any device
    """
    added_batch = False
    if vol.dim() == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)   # (1,1,D,H,W)
        added_batch = True
    elif vol.dim() == 4:
        vol = vol.unsqueeze(1)                # (N,1,D,H,W)

    N, C, D, H, W = vol.shape
    dev, dt = vol.device, vol.dtype
    sd, sh, sw = sigmas

    # D pass
    kz = _gaussian_kernel1d(sigma=sd, dtype=dt, device=dev).view(1, 1, -1, 1, 1)
    padz = (kz.shape[2] // 2, kz.shape[2] // 2)
    v = F.conv3d(F.pad(vol, (0,0,0,0,*padz), mode="replicate"), kz, groups=C)

    # H pass
    ky = _gaussian_kernel1d(sigma=sh, dtype=dt, device=dev).view(1, 1, 1, -1, 1)
    pady = (ky.shape[3] // 2, ky.shape[3] // 2)
    v = F.conv3d(F.pad(v, (0,0,*pady,0,0), mode="replicate"), ky, groups=C)

    # W pass
    kx = _gaussian_kernel1d(sigma=sw, dtype=dt, device=dev).view(1, 1, 1, 1, -1)
    padx = (kx.shape[4] // 2, kx.shape[4] // 2)
    v = F.conv3d(F.pad(v, (*padx,0,0,0,0), mode="replicate"), kx, groups=C)

    if added_batch:
        v = v.squeeze(0).squeeze(0)  # (D,H,W)
    else:
        v = v.squeeze(1)             # (N,D,H,W)
    return v

@torch.no_grad()
def otsu_threshold(vals: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    Otsu thresholding.

    Args:
        vals: 1D tensor of values (on device)
        bins: Number of bins for the histogram.

    Returns:
        Scalar threshold (same dtype/device)
    """
    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        return vmin

    # histogram on device
    hist = torch.histc(vals, bins=bins, min=float(vmin), max=float(vmax))
    bin_edges = torch.linspace(vmin, vmax, steps=bins+1, device=vals.device, dtype=vals.dtype)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    w0 = torch.cumsum(hist, dim=0)
    w1 = hist.sum() - w0
    mu = torch.cumsum(hist * bin_centers, dim=0)
    mu_t = mu[-1]
    # between-class variance
    denom = (w0 * w1).clamp_min(1e-12)
    bc_var = ((mu_t * w0 - mu)**2) / denom
    idx = torch.argmax(bc_var)
    return bin_centers[idx]