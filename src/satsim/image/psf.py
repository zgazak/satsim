"""Point spread function generation using PyTorch.

Gaussian PSF uses torch ops. POPPY PSF remains pure numpy.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import erfinv


def eod_to_sigma(eod: float, osf: int) -> float:
    """Calculate Gaussian sigma from energy-on-detector.

    Args:
        eod: Desired maximum energy on detector (0 < eod < 1).
        osf: Desired PSF oversample factor.

    Returns:
        Estimated Gaussian sigma in real pixel space.
    """
    return osf / (2.0 * math.sqrt(2.0) * erfinv(math.sqrt(eod)))


def gen_gaussian(height: int, width: int, sigma: float, dtype=torch.float32) -> torch.Tensor:
    """Generate a 2D Gaussian point spread function.

    Args:
        height: Height of the output PSF.
        width: Width of the output PSF.
        sigma: Standard deviation of the Gaussian.
        dtype: Output dtype.

    Returns:
        2D Gaussian PSF tensor of shape [height, width].
    """
    escale = 1.0 / (2.0 * sigma * sigma)
    gscale = escale / math.pi

    r = height / 2.0 - 0.5
    c = width / 2.0 - 0.5

    rr = torch.arange(-r, r + 1, dtype=dtype)
    cc = torch.arange(-c, c + 1, dtype=dtype)

    # Create 2D grid
    rrr, ccc = torch.meshgrid(rr, cc, indexing='ij')

    return torch.exp(-(ccc * ccc + rrr * rrr) * escale) * gscale


def gen_from_poppy(optical_system, wavelengths=None, weights=None):
    """Generate PSF from a POPPY optical system.

    Args:
        optical_system: POPPY OpticalSystem.
        wavelengths: Array of wavelengths to simulate.
        weights: Array of weights (same length as wavelengths).

    Returns:
        PSF as a numpy array.
    """
    if wavelengths is None:
        wavelengths = [600e-9]
    if weights is None:
        weights = [1]

    with np.errstate(divide='ignore'):
        psf = optical_system.calc_psf(normalize='last', source={
            'wavelengths': wavelengths,
            'weights': weights,
            'oversample': 2,
        })
        return psf[0].data


def gen_from_poppy_configuration(height, width, y_ifov, x_ifov, s_osf, config):
    """Generate PSF from a POPPY configuration dict.

    Args:
        height: Height of the output PSF.
        width: Width of the output PSF.
        y_ifov: Field of view of a pixel in y (degrees).
        x_ifov: Field of view of a pixel in x (degrees).
        s_osf: Oversample factor.
        config: Dict describing the optical system.

    Returns:
        PSF as a numpy array.
    """
    from astropy import units as u
    import poppy
    import importlib
    module = importlib.import_module('poppy')

    ifov = (x_ifov * u.deg / u.pixel).to(u.arcsec / u.pixel)
    optical_system_config = config['optical_system']

    osys = poppy.OpticalSystem(oversample=s_osf, npix=None)

    for c in optical_system_config:
        if c['type'] == 'CompoundAnalyticOptic':
            element_list = []
            for e in c['opticslist']:
                element_list.append(getattr(module, e['type'])(**e['kwargs']))
            osys.add_pupil(poppy.CompoundAnalyticOptic(opticslist=element_list))
        else:
            osys.add_pupil(getattr(module, c['type'])(**c['kwargs']))

    # fix for misspelled key
    if 'turbulant_atmosphere' in config and 'turbulent_atmosphere' not in config:
        config['turbulent_atmosphere'] = config['turbulant_atmosphere']

    if 'turbulent_atmosphere' in config:
        turbulent_atmosphere = config['turbulent_atmosphere']
        Cn2 = turbulent_atmosphere['Cn2'] * u.m**(-2 / 3)
        L = turbulent_atmosphere['propagation_distance'] * u.m
        nz = turbulent_atmosphere['zones']
        dz = L / nz
        for i in range(nz + 1):
            if i == 0 or i == nz:
                phase_screen = poppy.KolmogorovWFE(Cn2=Cn2, dz=dz / 2)
            else:
                phase_screen = poppy.KolmogorovWFE(Cn2=Cn2, dz=dz)
            osys.add_pupil(phase_screen)

    if 'size' in config:
        osys.add_detector(pixelscale=ifov.value, fov_pixels=config['size'])
        m = max(config['size'])
        npix = _calc_npix(osys, ifov * m * u.pixel, config['wavelengths'])
        osys.npix = npix

        h_pad = int((height - config['size'][0]) / 2 * s_osf)
        w_pad = int((width - config['size'][1]) / 2 * s_osf)

        psf = gen_from_poppy(osys, config['wavelengths'], config['weights'])
        psf_pad = np.pad(psf, ((h_pad, h_pad), (w_pad, w_pad)))
        return psf_pad
    else:
        osys.add_detector(pixelscale=ifov.value, fov_pixels=[height, width])
        m = max([height, width])
        npix = _calc_npix(osys, ifov * m * u.pixel, config['wavelengths'])
        osys.npix = npix
        return gen_from_poppy(osys, config['wavelengths'], config['weights'])


def _calc_npix(optical_system, fov, wavelengths):
    from astropy import units as u
    det_fov = fov.to(u.radian).value
    det_fov = det_fov * 1.1

    optimal_npix = []
    diam = optical_system.planes[0].pupil_diam
    for wl in wavelengths:
        optimal_npix.append(int(((det_fov / 2.0) * (diam / wl)).value + 1))

    npix = max(optimal_npix)

    if npix % 2 != 0:
        npix += 1

    return npix
