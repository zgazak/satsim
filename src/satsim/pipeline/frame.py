"""FrameResult dataclass for pipeline output.

Replaces the 14-tuple yielded by the old image_generator().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


@dataclass
class FrameResult:
    """Result of rendering a single frame.

    Attributes:
        frame_num: Frame number.
        fpa_digital: Final digitized image.
        fpa_star: Star signal (for SNR calculation).
        fpa_target: Target signal.
        background: Background signal.
        dark_current: Dark current.
        read_noise_sigma: Read noise sigma.
        astrometrics: WCS/pointing metadata.
        obs_pixels: Detected object pixel data.
        star_pixels: Star pixel data.
        ground_truth: Ground truth data.
        segmentation: Segmentation data.
    """
    frame_num: int = 0
    fpa_digital: torch.Tensor | np.ndarray | None = None
    fpa_star: torch.Tensor | np.ndarray | None = None
    fpa_target: torch.Tensor | np.ndarray | None = None
    background: torch.Tensor | np.ndarray | None = None
    dark_current: torch.Tensor | np.ndarray | None = None
    read_noise_sigma: torch.Tensor | np.ndarray | None = None
    astrometrics: dict = field(default_factory=dict)
    obs_pixels: list[dict] = field(default_factory=list)
    star_pixels: dict | None = None
    ground_truth: dict | None = None
    segmentation: dict | None = None
    obs_cache: Any = None

    def to_numpy(self) -> FrameResult:
        """Convert all tensor fields to numpy arrays."""
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        return FrameResult(
            frame_num=self.frame_num,
            fpa_digital=_to_np(self.fpa_digital),
            fpa_star=_to_np(self.fpa_star),
            fpa_target=_to_np(self.fpa_target),
            background=_to_np(self.background),
            dark_current=_to_np(self.dark_current),
            read_noise_sigma=_to_np(self.read_noise_sigma),
            astrometrics=self.astrometrics,
            obs_pixels=self.obs_pixels,
            star_pixels=self.star_pixels,
            ground_truth=self.ground_truth,
            segmentation=self.segmentation,
            obs_cache=self.obs_cache,
        )
