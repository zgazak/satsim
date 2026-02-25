"""Traditional 2D convolution using PyTorch.

Replaces TF nn.conv2d with torch F.conv2d.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from satsim.image.fpa import crop


def conv2(x: torch.Tensor, y: torch.Tensor, pad: int = 32) -> torch.Tensor:
    """Convolve two 2D tensors with traditional (spatial) convolution.

    Args:
        x: Input image as a 2D tensor.
        y: Input kernel as a 2D tensor.
        pad: Padding amount.

    Returns:
        The 2D convolution result (same size as x).
    """
    x = x.float()
    y = y.float()
    h, w = x.shape[0], x.shape[1]

    x = F.pad(x, (pad, pad, pad, pad))
    # Reshape to [batch, channels, H, W] for conv2d
    x_4d = x.unsqueeze(0).unsqueeze(0)
    # Flip kernel for correlation->convolution and reshape to [out_ch, in_ch, kH, kW]
    y_4d = y.flip(0, 1).unsqueeze(0).unsqueeze(0)

    out = F.conv2d(x_4d, y_4d, padding="same").squeeze(0).squeeze(0)
    out = crop(out, pad - 1, pad, h, w)
    return out
