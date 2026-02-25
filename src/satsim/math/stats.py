"""Signal-to-noise ratio statistics.

Replaces tf.sqrt with torch.sqrt for tensor inputs.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def signal_to_noise_ratio(signal, background, noise):
    """Calculate signal to noise ratio.

    Args:
        signal: Signal in total photoelectrons (tensor or scalar).
        background: Background in total photoelectrons (tensor or scalar).
        noise: RMS noise (tensor or scalar).

    Returns:
        Signal to noise ratio.
    """
    if isinstance(signal, torch.Tensor):
        return signal / torch.sqrt(signal + background + noise * noise)
    return signal / np.sqrt(signal + background + noise * noise)


def aperture_signal_to_noise_ratio(signal, background, noise, mask):
    """Calculate aperture signal to noise ratio for a mask.

    Args:
        signal: Signal in total photoelectrons (array or scalar).
        background: Background in total photoelectrons (array or scalar).
        noise: RMS noise (array or scalar).
        mask: Boolean mask for the aperture.

    Returns:
        SNR float, or None if the mask is empty.
    """
    mask = np.asarray(mask, dtype=bool)
    n_pix = int(mask.sum())
    if n_pix == 0:
        return None

    def _masked_sum(value):
        value = np.asarray(value)
        if value.ndim == 0:
            return float(value) * n_pix
        return float(value[mask].sum())

    def _masked_rn_sum(value):
        value = np.asarray(value)
        if value.ndim == 0:
            return float(value) * float(value) * n_pix
        return float(np.square(value[mask]).sum())

    signal_sum = _masked_sum(signal)
    background_sum = _masked_sum(background)
    rn_sum = _masked_rn_sum(noise)

    denom = math.sqrt(signal_sum + background_sum + rn_sum)
    if denom == 0:
        return 0.0
    return signal_sum / denom
