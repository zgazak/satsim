"""Scene setup extracted from satsim.py.

Handles star catalog loading, geometry initialization, and background setup.
"""

from __future__ import annotations

import logging
import copy

import numpy as np
import pydash

logger = logging.getLogger(__name__)


def setup_geometry(ssp):
    """Extract and validate geometry parameters from config.

    Args:
        ssp: Resolved config dict.

    Returns:
        Dict of geometry parameters.
    """
    h = ssp['fpa']['height']
    w = ssp['fpa']['width']
    s_osf = ssp['sim']['spacial_osf']
    t_osf = ssp['sim']['temporal_osf']
    padding = ssp['sim']['padding']

    h_pad_os = 2 * padding * s_osf
    w_pad_os = 2 * padding * s_osf
    h_pad_os_div2 = padding * s_osf
    w_pad_os_div2 = padding * s_osf

    y_ifov = ssp['fpa']['y_fov'] / h
    x_ifov = ssp['fpa']['x_fov'] / w
    y_ifov_os = y_ifov / s_osf
    x_ifov_os = x_ifov / s_osf

    h_fpa_os = h * s_osf
    w_fpa_os = w * s_osf
    h_fpa_pad_os = h_fpa_os + h_pad_os
    w_fpa_pad_os = w_fpa_os + w_pad_os

    render_mode = 'full'
    h_sub = h
    w_sub = w
    if 'render_size' in ssp['sim']:
        h_sub = ssp['sim']['render_size'][0]
        w_sub = ssp['sim']['render_size'][1]
        render_mode = 'piecewise'

    return {
        'h': h, 'w': w,
        's_osf': s_osf, 't_osf': t_osf,
        'padding': padding,
        'h_pad_os': h_pad_os, 'w_pad_os': w_pad_os,
        'h_pad_os_div2': h_pad_os_div2, 'w_pad_os_div2': w_pad_os_div2,
        'y_ifov': y_ifov, 'x_ifov': x_ifov,
        'y_ifov_os': y_ifov_os, 'x_ifov_os': x_ifov_os,
        'h_fpa_os': h_fpa_os, 'w_fpa_os': w_fpa_os,
        'h_fpa_pad_os': h_fpa_pad_os, 'w_fpa_pad_os': w_fpa_pad_os,
        'h_sub': h_sub, 'w_sub': w_sub,
        'render_mode': render_mode,
    }


def setup_background(ssp, h, w):
    """Setup background signal from config.

    Args:
        ssp: Config dict.
        h: Image height.
        w: Image width.

    Returns:
        Background array or scalar.
    """
    bg = pydash.get(ssp, 'background.galactic', 0.0)
    if isinstance(bg, (int, float)):
        return np.full((h, w), bg, dtype=np.float32)
    return np.array(bg, dtype=np.float32)
