"""Frame generation engine.

Orchestrates the rendering pipeline: scene setup -> rendering -> noise -> digitization.
Extracted from the 940-line image_generator() in satsim.py.
"""

from __future__ import annotations

import logging
import copy
from typing import Iterator

import numpy as np
import torch
import pydash

from satsim.pipeline.frame import FrameResult
from satsim.pipeline.scene import setup_geometry, setup_background
from satsim.image.fpa import analog_to_digital, mv_to_pe, crop, downsample
from satsim.image.psf import gen_gaussian, eod_to_sigma, gen_from_poppy_configuration
from satsim.image.noise import add_photon_noise, add_read_noise
from satsim.image.render import render_piecewise, render_full
from satsim.device import ensure_tensor

logger = logging.getLogger(__name__)


def generate_frames(ssp: dict) -> Iterator[FrameResult]:
    """Generate frames from a resolved config. Pure rendering, zero I/O.

    This is the core rendering pipeline extracted from image_generator().

    Args:
        ssp: Fully resolved config dictionary.

    Yields:
        FrameResult for each frame.
    """
    # Setup geometry
    geom = setup_geometry(ssp)
    h = geom['h']
    w = geom['w']
    s_osf = geom['s_osf']
    t_osf = geom['t_osf']

    # Setup sensor parameters
    num_frames = ssp['fpa']['num_frames']
    t_exposure = ssp['fpa']['time']['exposure']
    zeropoint = ssp['fpa']['zeropoint']

    a2d_gain = ssp['fpa']['a2d']['gain']
    a2d_fwc = ssp['fpa']['a2d']['fwc']
    a2d_bias = ssp['fpa']['a2d'].get('bias', 0)
    a2d_dtype = ssp['fpa']['a2d'].get('dtype', 'uint16')

    rn = pydash.get(ssp, 'fpa.noise.read', 0.0)
    en = pydash.get(ssp, 'fpa.noise.electronic', 0.0)
    enable_shot_noise = pydash.get(ssp, 'sim.enable_shot_noise', True)
    num_shot_noise_samples = pydash.get(ssp, 'fpa.noise.shot_noise_samples', None)

    # Dark current
    dc = pydash.get(ssp, 'fpa.dark_current', 0.0)
    if isinstance(dc, (int, float)):
        dc_pe = float(dc) * t_exposure
    else:
        dc_pe = np.array(dc, dtype=np.float32) * t_exposure

    # Background
    bg = pydash.get(ssp, 'background.galactic', 0.0)
    if isinstance(bg, (int, float)):
        bg_pe = mv_to_pe(zeropoint, float(bg)) * t_exposure if bg != 0.0 else 0.0
    else:
        bg_pe = mv_to_pe(zeropoint, np.array(bg, dtype=np.float32)) * t_exposure

    # PSF
    psf_os = _gen_psf(ssp, geom)
    if psf_os is not None:
        psf_os = ensure_tensor(psf_os)

    # Render mode
    render_mode = pydash.get(ssp, 'sim.render_mode', 'full')
    if render_mode == 'none':
        # Analytical-only mode
        for frame_num in range(num_frames):
            yield FrameResult(frame_num=frame_num, astrometrics={})
        return

    # Generate each frame
    for frame_num in range(num_frames):
        logger.debug('Rendering frame %d of %d.', frame_num + 1, num_frames)

        # For now, use empty star/target arrays as placeholders
        # The full pipeline would call geometry/objects.py here
        r_obs_os = ensure_tensor([])
        c_obs_os = ensure_tensor([])
        pe_obs_os = ensure_tensor([])
        r_stars_os = ensure_tensor([])
        c_stars_os = ensure_tensor([])
        pe_stars_os = ensure_tensor([])

        t_start = 0.0
        t_end = t_exposure
        star_rot_rate = 0.0
        star_tran_os = [0.0, 0.0]

        # Render
        if geom['render_mode'] == 'piecewise':
            fpa_conv_star, fpa_conv_targ, _, _, _ = render_piecewise(
                h, w, geom['h_sub'], geom['w_sub'],
                geom['h_pad_os'], geom['w_pad_os'],
                s_osf, psf_os,
                r_obs_os, c_obs_os, pe_obs_os,
                r_stars_os, c_stars_os, pe_stars_os,
                t_start, t_end, t_osf,
                star_rot_rate, star_tran_os,
            )
        else:
            fpa_conv_star, fpa_conv_targ, _, _, _ = render_full(
                geom['h_fpa_os'], geom['w_fpa_os'],
                geom['h_fpa_pad_os'], geom['w_fpa_pad_os'],
                geom['h_pad_os_div2'], geom['w_pad_os_div2'],
                s_osf, psf_os,
                r_obs_os, c_obs_os, pe_obs_os,
                r_stars_os, c_stars_os, pe_stars_os,
                t_start, t_end, t_osf,
                star_rot_rate, star_tran_os,
            )

        # Add background and dark current
        fpa = fpa_conv_star + fpa_conv_targ
        bg_tensor = ensure_tensor(bg_pe) if not isinstance(bg_pe, (int, float)) else torch.full_like(fpa, bg_pe)
        dc_tensor = ensure_tensor(dc_pe) if not isinstance(dc_pe, (int, float)) else torch.full_like(fpa, dc_pe)
        rn_tensor = torch.full_like(fpa, float(rn))

        fpa = fpa + bg_tensor + dc_tensor

        # Add noise
        if enable_shot_noise:
            fpa = add_photon_noise(fpa, samples=num_shot_noise_samples)

        fpa, noise = add_read_noise(fpa, rn, en)

        # A/D conversion
        fpa_digital = analog_to_digital(fpa, a2d_gain, a2d_fwc, a2d_bias, a2d_dtype)

        yield FrameResult(
            frame_num=frame_num,
            fpa_digital=fpa_digital,
            fpa_star=fpa_conv_star,
            fpa_target=fpa_conv_targ,
            background=bg_tensor,
            dark_current=dc_tensor,
            read_noise_sigma=rn_tensor,
            astrometrics={},
            obs_pixels=[],
        )


def _gen_psf(ssp, geom):
    """Generate PSF from config."""
    psf_config = pydash.get(ssp, 'fpa.psf', {})
    mode = psf_config.get('mode', 'gaussian')

    if mode == 'none':
        return None

    s_osf = geom['s_osf']
    h_fpa_pad_os = geom['h_fpa_pad_os']
    w_fpa_pad_os = geom['w_fpa_pad_os']

    if mode == 'gaussian':
        eod = psf_config.get('eod', 0.8)
        sigma = eod_to_sigma(eod, s_osf)
        return gen_gaussian(h_fpa_pad_os, w_fpa_pad_os, sigma)

    elif mode == 'poppy':
        y_ifov = geom['y_ifov']
        x_ifov = geom['x_ifov']
        h = geom['h']
        w = geom['w']
        return gen_from_poppy_configuration(h, w, y_ifov, x_ifov, s_osf, psf_config)

    return None
