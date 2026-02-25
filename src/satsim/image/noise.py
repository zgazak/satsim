"""Noise models using PyTorch.

Replaces TF random ops with torch.poisson and torch.randn.
"""

from __future__ import annotations

import torch


def add_photon_noise(fpa: torch.Tensor, samples: int | None = None) -> torch.Tensor:
    """Add photon noise (Poisson distributed).

    Photon noise results from the inherent statistical variation in the
    arrival rate of photons incident on the CCD.

    Args:
        fpa: Input image as a 2D tensor in total photoelectrons per pixel.
        samples: Number of samples to average. Used to estimate averaging
            of multiple images. Default=None (single sample).

    Returns:
        The 2D tensor with photon noise applied.
    """
    if samples is not None:
        # Use float64 for multi-sample averaging to match TF precision
        fpa64 = fpa.to(torch.float64)
        accum = torch.zeros_like(fpa64)
        for _ in range(samples):
            accum = accum + torch.poisson(fpa64.clamp(min=0))
        return (accum / samples).to(fpa.dtype)
    else:
        return torch.poisson(fpa.clamp(min=0).float()).to(fpa.dtype)


def add_read_noise(fpa: torch.Tensor, rn: float, en: float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Add read noise (Gaussian distributed).

    Read noise is a combination of noise from the pixel and from the
    analog to digital converter (ADC).

    Args:
        fpa: Input image as a 2D tensor in real pixels.
        rn: Electrons RMS value of the read noise.
        en: Electrons RMS value of the electronic noise.

    Returns:
        Tuple of (image_with_noise, noise_tensor).
    """
    rn = torch.tensor(rn, dtype=torch.float32)
    en = torch.tensor(en, dtype=torch.float32)
    noise = torch.randn_like(fpa) * torch.sqrt(rn * rn + en * en)
    return fpa + noise, noise
