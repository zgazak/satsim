"""Frame generation engine.

Orchestrates the rendering pipeline: scene setup -> rendering -> noise -> digitization.
Extracted from the 940-line image_generator() in satsim.py.
"""

from __future__ import annotations

import math
import logging
import copy
from typing import Iterator

import numpy as np
import torch
import pydash

from satsim.pipeline.frame import FrameResult
from satsim.pipeline.scene import setup_geometry, setup_background
from satsim.image.fpa import analog_to_digital, mv_to_pe, pe_to_mv, crop, downsample
from satsim.image.psf import gen_gaussian, eod_to_sigma, gen_from_poppy_configuration
from satsim.image.noise import add_photon_noise, add_read_noise
from satsim.image.render import render_piecewise, render_full
from satsim.geometry.random import gen_random_points
from satsim.geometry.draw import gen_line
from satsim.geometry.transform import apply_wrap_around
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
    t_gap = ssp['fpa']['time'].get('gap', 0.0)
    zeropoint = ssp['fpa']['zeropoint']

    y_ifov = geom['y_ifov']
    x_ifov = geom['x_ifov']

    a2d_gain = ssp['fpa']['a2d']['gain']
    a2d_fwc = ssp['fpa']['a2d']['fwc']
    a2d_bias = ssp['fpa']['a2d'].get('bias', 0)
    a2d_dtype = ssp['fpa']['a2d'].get('dtype', 'uint16')

    rn = pydash.get(ssp, 'fpa.noise.read', 0.0)
    en = pydash.get(ssp, 'fpa.noise.electronic', 0.0)
    enable_shot_noise = pydash.get(ssp, 'sim.enable_shot_noise', True)
    num_shot_noise_samples = pydash.get(ssp, 'fpa.noise.shot_noise_samples', None)

    # Velocity unit conversion factors
    velocity_units = pydash.get(ssp, 'sim.velocity_units', 'pixels')
    if velocity_units == 'arcsec':
        y_to_pix = y_ifov * 3600
        x_to_pix = x_ifov * 3600
    else:
        y_to_pix = 1.0
        x_to_pix = 1.0

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

    # Bias
    bias_pe = pydash.get(ssp, 'fpa.bias', 0.0)

    # PSF
    psf_os = _gen_psf(ssp, geom)
    if psf_os is not None:
        psf_os = ensure_tensor(psf_os)

    # Render mode
    render_mode = pydash.get(ssp, 'sim.render_mode', 'full')
    if render_mode == 'none':
        for frame_num in range(num_frames):
            yield FrameResult(frame_num=frame_num, astrometrics={})
        return

    # --- Generate stars ---
    r_stars_os, c_stars_os, pe_stars_os, star_rot_rate, star_tran_os = _gen_stars(
        ssp, geom, zeropoint, t_exposure, y_to_pix, x_to_pix
    )

    # --- Generate observation objects (targets) for each frame ---
    obs_list = _parse_obs_list(ssp)

    # Track star bounds for wrap-around
    star_bounds = None

    # Generate each frame
    for frame_num in range(num_frames):
        logger.debug('Rendering frame %d of %d.', frame_num + 1, num_frames)

        t_start = frame_num * (t_exposure + t_gap)
        t_end = t_start + t_exposure

        # Apply star motion with wrap-around
        if len(r_stars_os) > 0:
            r_stars_frame, c_stars_frame, star_bounds = apply_wrap_around(
                geom['h_fpa_pad_os'], geom['w_fpa_pad_os'],
                r_stars_os, c_stars_os,
                t_start, t_end, star_rot_rate, star_tran_os, star_bounds
            )
        else:
            r_stars_frame = r_stars_os
            c_stars_frame = c_stars_os

        # Generate target pixel arrays for this frame
        r_obs_os, c_obs_os, pe_obs_os = _gen_obs_pixels(
            obs_list, geom, zeropoint, t_exposure, t_start, t_end,
            y_to_pix, x_to_pix
        )

        r_stars_t = ensure_tensor(r_stars_frame)
        c_stars_t = ensure_tensor(c_stars_frame)
        pe_stars_t = ensure_tensor(pe_stars_os)
        r_obs_t = ensure_tensor(r_obs_os)
        c_obs_t = ensure_tensor(c_obs_os)
        pe_obs_t = ensure_tensor(pe_obs_os)

        # Render
        if geom['render_mode'] == 'piecewise':
            fpa_conv_star, fpa_conv_targ, _, _, _ = render_piecewise(
                h, w, geom['h_sub'], geom['w_sub'],
                geom['h_pad_os'], geom['w_pad_os'],
                s_osf, psf_os,
                r_obs_t, c_obs_t, pe_obs_t,
                r_stars_t, c_stars_t, pe_stars_t,
                t_start, t_end, t_osf,
                star_rot_rate, star_tran_os,
            )
        else:
            fpa_conv_star, fpa_conv_targ, _, _, _ = render_full(
                geom['h_fpa_os'], geom['w_fpa_os'],
                geom['h_fpa_pad_os'], geom['w_fpa_pad_os'],
                geom['h_pad_os_div2'], geom['w_pad_os_div2'],
                s_osf, psf_os,
                r_obs_t, c_obs_t, pe_obs_t,
                r_stars_t, c_stars_t, pe_stars_t,
                t_start, t_end, t_osf,
                star_rot_rate, star_tran_os,
            )

        # Add background, dark current, and bias
        fpa = fpa_conv_star + fpa_conv_targ
        bg_tensor = ensure_tensor(bg_pe) if not isinstance(bg_pe, (int, float)) else torch.full_like(fpa, bg_pe)
        dc_tensor = ensure_tensor(dc_pe) if not isinstance(dc_pe, (int, float)) else torch.full_like(fpa, dc_pe)
        rn_tensor = torch.full_like(fpa, float(rn))

        fpa = fpa + bg_tensor + dc_tensor

        if isinstance(bias_pe, (int, float)) and bias_pe != 0:
            fpa = fpa + bias_pe
        elif not isinstance(bias_pe, (int, float)):
            fpa = fpa + ensure_tensor(np.array(bias_pe, dtype=np.float32))

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


def _gen_stars(ssp, geom, zeropoint, t_exposure, y_to_pix, x_to_pix):
    """Generate star positions and brightnesses from config.

    Returns:
        (r_stars_os, c_stars_os, pe_stars_os, star_rot_rate, star_tran_os)
    """
    s_osf = geom['s_osf']
    stars_config = pydash.get(ssp, 'geometry.stars', {})
    mode = stars_config.get('mode', 'none')

    # Parse star motion
    motion = stars_config.get('motion', {})
    motion_mode = motion.get('mode', 'none')

    if motion_mode == 'affine':
        star_rot_rate = float(motion.get('rotation', 0.0))
        tran = motion.get('translation', [0.0, 0.0])
        star_tran_os = [
            float(tran[0]) / y_to_pix * s_osf,
            float(tran[1]) / x_to_pix * s_osf,
        ]
    elif motion_mode == 'affine-polar':
        star_rot_rate = float(motion.get('rotation', 0.0))
        vel_angle = float(motion.get('translation', [0.0, 0.0])[0]) * math.pi / 180
        vel_mag = float(motion.get('translation', [0.0, 0.0])[1])
        star_tran_os = [
            vel_mag * math.sin(vel_angle) / y_to_pix * s_osf,
            vel_mag * math.cos(vel_angle) / x_to_pix * s_osf,
        ]
    else:
        star_rot_rate = 0.0
        star_tran_os = [0.0, 0.0]

    if mode == 'none' or mode is None:
        return np.array([]), np.array([]), np.array([]), star_rot_rate, star_tran_os

    if mode == 'bins':
        mv_config = stars_config.get('mv', {})
        mv_bins = mv_config.get('bins', [])
        density = mv_config.get('density', [])

        if not mv_bins or not density:
            return np.array([]), np.array([]), np.array([]), star_rot_rate, star_tran_os

        # Convert magnitude bins to photoelectron bins
        pe_bins = [mv_to_pe(zeropoint, float(m)) * t_exposure for m in mv_bins]

        h_fpa_pad_os = geom['h_fpa_pad_os']
        w_fpa_pad_os = geom['w_fpa_pad_os']

        y_fov = ssp['fpa']['y_fov']
        x_fov = ssp['fpa']['x_fov']
        padding = geom['padding']

        # Padded FOV
        y_fov_pad = y_fov + 2 * padding * geom['y_ifov']
        x_fov_pad = x_fov + 2 * padding * geom['x_ifov']

        star_pad = stars_config.get('pad', 1)

        r_stars_os, c_stars_os, pe_stars_os = gen_random_points(
            h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad,
            pe_bins, density, pad_mult=star_pad
        )

        # Offset stars to pad origin (stars are generated relative to padded image)
        r_stars_os = r_stars_os.astype(np.float64)
        c_stars_os = c_stars_os.astype(np.float64)
        pe_stars_os = pe_stars_os.astype(np.float64)

        logger.info('Generated %d stars from bins.', len(r_stars_os))

        return r_stars_os, c_stars_os, pe_stars_os, star_rot_rate, star_tran_os

    # For catalog modes (sstr7, etc.), fall through to empty
    logger.warning('Star mode "%s" not yet implemented in v2 engine, skipping stars.', mode)
    return np.array([]), np.array([]), np.array([]), star_rot_rate, star_tran_os


def _parse_obs_list(ssp):
    """Parse the observation object list from config.

    Returns:
        list of obs dicts, or empty list.
    """
    obs_config = pydash.get(ssp, 'geometry.obs', {})
    mode = obs_config.get('mode', 'none')

    if mode == 'none' or mode is None:
        return []

    if mode == 'list':
        obs_list = obs_config.get('list', [])
        if isinstance(obs_list, dict):
            # Single object, wrap in list
            obs_list = [obs_list]
        return obs_list

    logger.warning('Obs mode "%s" not yet implemented in v2 engine.', mode)
    return []


def _gen_obs_pixels(obs_list, geom, zeropoint, t_exposure, t_start, t_end,
                    y_to_pix, x_to_pix):
    """Generate target pixel positions and brightnesses for one frame.

    Returns:
        (r_obs_os, c_obs_os, pe_obs_os) as numpy arrays.
    """
    s_osf = geom['s_osf']
    h_fpa_os = geom['h_fpa_os']
    w_fpa_os = geom['w_fpa_os']
    h_pad_os_div2 = geom['h_pad_os_div2']
    w_pad_os_div2 = geom['w_pad_os_div2']

    all_r = []
    all_c = []
    all_pe = []

    for o in obs_list:
        obj_mode = o.get('mode', 'none')

        if obj_mode == 'none':
            continue

        # Get brightness in pe/sec
        if 'mv' in o:
            mv_val = o['mv']
            if callable(mv_val):
                ope = 0.0  # pipeline function, handled per-pixel
            else:
                ope = mv_to_pe(zeropoint, float(mv_val))
        elif 'pe' in o:
            pe_val = o['pe']
            if callable(pe_val):
                ope = 0.0
            else:
                ope = float(pe_val)
        else:
            ope = 0.0

        if obj_mode == 'line':
            origin = o.get('origin', [0.5, 0.5])
            velocity = o.get('velocity', [0.0, 0.0])

            # Convert velocity to oversampled pixels/sec
            ovrc = [
                float(velocity[0]) / y_to_pix * s_osf,
                float(velocity[1]) / x_to_pix * s_osf,
            ]

            epoch = float(o.get('epoch', 0.0))

            rr, cc, pp, tt = gen_line(
                h_fpa_os, w_fpa_os, origin, ovrc, ope,
                t_start + epoch, t_end + epoch
            )

            all_r.append(rr)
            all_c.append(cc)
            all_pe.append(pp)

    if not all_r:
        return np.array([]), np.array([]), np.array([])

    r_obs_os = np.concatenate(all_r).astype(np.float64)
    c_obs_os = np.concatenate(all_c).astype(np.float64)
    pe_obs_os = np.concatenate(all_pe).astype(np.float64)

    return r_obs_os, c_obs_os, pe_obs_os


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
