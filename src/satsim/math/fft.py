"""FFT-based convolution using PyTorch.

Performance-critical path: PSF convolution via FFT.
Replaces TensorFlow signal ops with torch.fft.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fftshift(x: torch.Tensor, dims: int = 2) -> torch.Tensor:
    """Shift the zero-frequency component to the center of the spectrum.

    Args:
        x: Input tensor.
        dims: Number of dimensions (1 or 2).

    Returns:
        The shifted tensor.
    """
    if dims == 2:
        return torch.fft.fftshift(x, dim=(-2, -1))
    elif dims == 1:
        return torch.fft.fftshift(x, dim=-1)
    else:
        raise ValueError("1 or 2 dimensional tensors supported.")


def fftconv2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Convolve two 2D tensors using FFT.

    Full-precision convolution with proper N+L-1 padding.
    x and y must have the same shape.

    Args:
        x: 2D input tensor.
        y: 2D input tensor of the same shape as x.

    Returns:
        The 2D convolution result (same size as inputs).
    """
    sx, sy = x.shape[0], x.shape[1]
    sxp = sx - 1
    syp = sy - 1

    # Pad for linear convolution via circular convolution
    x_padded = F.pad(x, (0, syp, 0, sxp))
    y_padded = F.pad(y, (0, syp, 0, sxp))

    # FFT, pointwise multiply, IFFT
    x_freq = torch.fft.rfft2(x_padded)
    y_freq = torch.fft.rfft2(y_padded)
    fftfull = torch.fft.irfft2(x_freq * y_freq, s=x_padded.shape)

    # Crop to return 'same' sized array
    start_r = sxp // 2
    start_c = syp // 2
    return fftfull[start_r:start_r + sx, start_c:start_c + sy]


# Module-level PSF FFT cache: stores pre-computed rfft2(psf) keyed by id
_psf_cache: dict[int, tuple[int, torch.Tensor]] = {}


def _get_cached_psf_fft(y: torch.Tensor, pad: int) -> torch.Tensor:
    """Get cached rfft2 of PSF tensor, computing if not cached."""
    key = id(y)
    if key in _psf_cache:
        cached_pad, cached_fft = _psf_cache[key]
        if cached_pad == pad:
            return cached_fft

    y_padded = F.pad(y, (pad, pad, pad, pad))
    y_freq = torch.fft.rfft2(y_padded)
    _psf_cache[key] = (pad, y_freq)
    return y_freq


def clear_psf_cache() -> None:
    """Clear the PSF FFT cache."""
    _psf_cache.clear()


def fftconv2p(x: torch.Tensor, y: torch.Tensor, pad: int = 32, cache_last_y: bool = True) -> torch.Tensor:
    """Convolve two 2D tensors using padded FFT (approximate, faster).

    x and y must have the same shape. Dimensions should be multiples of 2.

    Args:
        x: 2D input tensor.
        y: 2D input tensor of the same shape as x.
        pad: Number of pixels to pad all sides before FFT.
        cache_last_y: If True, cache the FFT of y for repeated calls.

    Returns:
        The approximate 2D convolution result (same size as inputs).
    """
    sx, sy = x.shape[0], x.shape[1]

    x_padded = F.pad(x, (pad, pad, pad, pad))
    x_freq = torch.fft.rfft2(x_padded)

    if cache_last_y:
        y_freq = _get_cached_psf_fft(y, pad)
    else:
        y_padded = F.pad(y, (pad, pad, pad, pad))
        y_freq = torch.fft.rfft2(y_padded)

    fftfull = torch.fft.irfft2(x_freq * y_freq, s=x_padded.shape)

    # Shift and crop
    fftfull = fftshift(fftfull)
    return fftfull[pad - 1:pad - 1 + sx, pad - 1:pad - 1 + sy]
