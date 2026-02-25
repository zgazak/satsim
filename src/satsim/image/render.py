"""Image rendering pipeline using PyTorch.

Replaces TF tensor creation and operations with PyTorch equivalents.
"""

from __future__ import annotations

import math
import logging

import numpy as np
import torch

from satsim.math.fft import fftconv2p
from satsim.image.fpa import downsample, crop, add_counts, transform_and_add_counts, transform_and_fft

logger = logging.getLogger(__name__)


def render_piecewise(h, w, h_sub, w_sub, h_pad_os, w_pad_os, s_osf, psf_os,
                     r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os,
                     pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate,
                     star_tran_os, render_separate=True, star_render_mode='transform'):
    """Render image in sub-sections (tiled rendering).

    Useful when the full oversampled image doesn't fit in GPU memory.

    Args:
        h: Image height in real pixels.
        w: Image width in real pixels.
        h_sub: Sub-section height in real pixels.
        w_sub: Sub-section width in real pixels.
        h_pad_os: Total vertical pad in oversampled space.
        w_pad_os: Total horizontal pad in oversampled space.
        s_osf: Spatial oversample factor.
        psf_os: PSF in oversampled space.
        r_obs_os: Target row coordinates in oversampled space.
        c_obs_os: Target column coordinates in oversampled space.
        pe_obs_os: Target brightnesses in photoelectrons per pixel.
        r_stars_os: Star row coordinates in oversampled space at epoch.
        c_stars_os: Star column coordinates in oversampled space at epoch.
        pe_stars_os: Star brightnesses in photoelectrons per second.
        t_start_star: Start time in seconds from epoch.
        t_end_star: End time in seconds from epoch.
        t_osf: Temporal oversample factor.
        star_rot_rate: Star rotation rate in degrees per second.
        star_tran_os: Star translation rate in oversampled pixels/sec [row, col].
        render_separate: If True, render targets and stars separately.
        star_render_mode: 'fft' or 'transform'.

    Returns:
        Tuple: (fpa_conv_star, fpa_conv_targ, None, None, None)
    """
    h_fpa_os = int(h * s_osf)
    w_fpa_os = int(w * s_osf)
    h_sub_os = int(h_sub * s_osf)
    w_sub_os = int(w_sub * s_osf)
    h_sub_os_f = float(h_sub_os)
    w_sub_os_f = float(w_sub_os)
    h_sub_pad_os = int(h_sub_os + h_pad_os)
    w_sub_pad_os = int(w_sub_os + w_pad_os)
    h_pad_os_div2 = int(h_pad_os / 2)
    w_pad_os_div2 = int(w_pad_os / 2)

    n_h_div = math.ceil(h_fpa_os / h_sub_os)
    n_w_div = math.ceil(w_fpa_os / w_sub_os)

    fpa_conv_star = np.zeros((n_h_div, n_w_div, h_sub, w_sub))
    fpa_conv_targ = np.zeros((n_h_div, n_w_div, h_sub, w_sub))

    logger.debug('Rendering %dx%d divisions with %dx%d pixels.', n_h_div, n_w_div, h_sub, w_sub)

    for ir in range(n_h_div):
        logger.debug('Rendering row %d of %d.', ir + 1, n_h_div)
        for ic in range(n_w_div):
            r_sub_os_start = float(ir) * h_sub_os_f
            c_sub_os_start = float(ic) * w_sub_os_f

            r_stars_sub = r_stars_os - r_sub_os_start
            c_stars_sub = c_stars_os - c_sub_os_start
            pe_stars_sub = pe_stars_os

            r_obs_sub = r_obs_os - r_sub_os_start
            c_obs_sub = c_obs_os - c_sub_os_start
            pe_obs_sub = pe_obs_os

            star, targ, _, _, _ = render_full(
                h_sub_os, w_sub_os, h_sub_pad_os, w_sub_pad_os,
                h_pad_os_div2, w_pad_os_div2, s_osf, psf_os,
                r_obs_sub, c_obs_sub, pe_obs_sub,
                r_stars_sub, c_stars_sub, pe_stars_sub,
                t_start_star, t_end_star, t_osf,
                star_rot_rate, star_tran_os,
                render_separate=render_separate,
                star_render_mode=star_render_mode
            )
            fpa_conv_star[ir][ic] = star.cpu().numpy() if isinstance(star, torch.Tensor) else star
            fpa_conv_targ[ir][ic] = targ.cpu().numpy() if isinstance(targ, torch.Tensor) else targ

    # Stitch sub-sections
    fpa_conv_star_t = torch.tensor(fpa_conv_star, dtype=torch.float32)
    fpa_conv_targ_t = torch.tensor(fpa_conv_targ, dtype=torch.float32)
    fpa_conv_star_t = fpa_conv_star_t.permute(0, 2, 1, 3).reshape(h_sub * n_h_div, w_sub * n_w_div)
    fpa_conv_targ_t = fpa_conv_targ_t.permute(0, 2, 1, 3).reshape(h_sub * n_h_div, w_sub * n_w_div)

    fpa_conv_star_out = fpa_conv_star_t[0:h, 0:w].float()
    fpa_conv_targ_out = fpa_conv_targ_t[0:h, 0:w].float()

    return fpa_conv_star_out, fpa_conv_targ_out, None, None, None


def render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os,
                h_pad_os_div2, w_pad_os_div2, s_osf, psf_os,
                r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os,
                pe_stars_os, t_start_star, t_end_star, t_osf,
                star_rot_rate, star_tran_os, render_separate=True,
                obs_model=None, star_render_mode='transform'):
    """Render a full-frame image.

    Args:
        h_fpa_os: Image height in oversampled space.
        w_fpa_os: Image width in oversampled space.
        h_fpa_pad_os: Image height with pad in oversampled space.
        w_fpa_pad_os: Image width with pad in oversampled space.
        h_pad_os_div2: Half pad height in oversampled space.
        w_pad_os_div2: Half pad width in oversampled space.
        s_osf: Spatial oversample factor.
        psf_os: PSF in oversampled space.
        r_obs_os: Target row coordinates in oversampled space.
        c_obs_os: Target column coordinates in oversampled space.
        pe_obs_os: Target brightnesses.
        r_stars_os: Star row coordinates in oversampled space.
        c_stars_os: Star column coordinates in oversampled space.
        pe_stars_os: Star brightnesses.
        t_start_star: Start time from epoch.
        t_end_star: End time from epoch.
        t_osf: Temporal oversample factor.
        star_rot_rate: Star rotation rate.
        star_tran_os: Star translation rate [row, col].
        render_separate: If True, render targets and stars separately.
        obs_model: List of model arrays to add to target image.
        star_render_mode: 'fft' or 'transform'.

    Returns:
        Tuple: (fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, fpa_conv_os, fpa_conv_crop)
    """
    # Render stars
    fpa_os_w_stars = torch.zeros([h_fpa_pad_os, w_fpa_pad_os], dtype=torch.float32)
    if star_render_mode == 'fft':
        fpa_os_w_stars = transform_and_fft(
            fpa_os_w_stars, r_stars_os, c_stars_os, pe_stars_os,
            t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os
        )
    else:
        fpa_os_w_stars = transform_and_add_counts(
            fpa_os_w_stars, r_stars_os, c_stars_os, pe_stars_os,
            t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os
        )

    # Render modeled targets
    fpa_os_w_targets = torch.zeros([h_fpa_pad_os, w_fpa_pad_os], dtype=torch.float32)
    if obs_model is not None and len(obs_model) > 0:
        for om in obs_model:
            if isinstance(om, torch.Tensor):
                fpa_os_w_targets = fpa_os_w_targets + om.float()
            else:
                fpa_os_w_targets = fpa_os_w_targets + torch.tensor(om, dtype=torch.float32)

        # Mask stars (occultation)
        condition = fpa_os_w_targets > 0
        fpa_os_w_stars = torch.where(condition, torch.zeros_like(fpa_os_w_stars), fpa_os_w_stars)

    # Render point source targets
    fpa_os_w_targets = add_counts(fpa_os_w_targets, r_obs_os, c_obs_os, pe_obs_os, h_pad_os_div2, w_pad_os_div2)

    if psf_os is None:
        if render_separate:
            fpa_conv_crop = crop(fpa_os_w_stars, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
            fpa_conv_star = downsample(fpa_conv_crop, s_osf, 'pool')

            fpa_conv_crop_targ = crop(fpa_os_w_targets, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
            fpa_conv_targ = downsample(fpa_conv_crop_targ, s_osf, 'pool')

            return fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, None, fpa_conv_crop

        fpa_os_sum = fpa_os_w_stars + fpa_os_w_targets
        fpa_conv_crop = crop(fpa_os_sum, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_ds = downsample(fpa_conv_crop, s_osf, 'pool')

        return fpa_conv_ds, torch.zeros_like(fpa_conv_ds), fpa_os_w_targets, None, fpa_conv_crop

    if render_separate:
        fpa_conv_os = fftconv2p(fpa_os_w_stars, psf_os, pad=1)
        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_star = downsample(fpa_conv_crop, s_osf, 'pool')

        fpa_conv_os = fftconv2p(fpa_os_w_targets, psf_os, pad=1)
        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_targ = downsample(fpa_conv_crop, s_osf, 'pool')

        return fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, fpa_conv_os, fpa_conv_crop

    else:
        fpa_os_sum = fpa_os_w_stars + fpa_os_w_targets
        if torch.all(fpa_os_sum == 0):
            fpa_conv_os = fpa_os_sum
        else:
            fpa_conv_os = fftconv2p(fpa_os_sum, psf_os, pad=1)

        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_ds = downsample(fpa_conv_crop, s_osf, 'pool')

        return fpa_conv_ds, torch.zeros_like(fpa_conv_ds), fpa_os_w_targets, fpa_conv_os, fpa_conv_crop
