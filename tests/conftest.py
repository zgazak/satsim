"""Shared test fixtures and configuration for SatSim tests."""

import pytest
import torch
import numpy as np

from satsim.device import set_device, get_device


@pytest.fixture(autouse=True)
def _reset_device():
    """Ensure CPU device and clean state for each test."""
    set_device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    set_device("cpu")


@pytest.fixture
def gpu_device():
    """Skip test if no CUDA available, otherwise return cuda device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda", 0)


@pytest.fixture
def minimal_config():
    """Minimal valid SatSim config dict."""
    return {
        "version": "2.0",
        "sim": {
            "spacial_osf": 1,
            "temporal_osf": 1,
            "padding": 10,
            "samples": 1,
            "show_obs_boxes": True,
            "show_star_boxes": False,
            "save_movie": False,
            "save_czml": False,
            "save_pickle": False,
            "save_jpeg": False,
            "fits_compression": None,
            "star_render_mode": "transform",
            "render_mode": "full",
            "calculate_snr": True,
            "enable_shot_noise": True,
        },
        "fpa": {
            "height": 32,
            "width": 32,
            "y_fov": 1.0,
            "x_fov": 1.0,
            "dark_current": 0.0,
            "zeropoint": 20.0,
            "num_frames": 2,
            "time": {"exposure": 1.0},
            "a2d": {"gain": 1.0, "fwc": 100000.0, "bias": 0, "dtype": "uint16"},
            "noise": {"read": 5.0, "electronic": 0.0},
            "psf": {"mode": "gaussian", "eod": 0.8},
        },
        "background": {"galactic": 0.0},
        "geometry": {
            "site": {"lat": 0.0, "lon": 0.0, "alt": 0.0},
            "obs": {"mode": "list", "list": []},
            "stars": {"mode": "none"},
            "time": {},
            "track": {"mode": "rate"},
        },
    }


@pytest.fixture
def sample_image():
    """A small sample 2D tensor for testing image operations."""
    return torch.randn(64, 64)


@pytest.fixture
def sample_psf():
    """A small normalized PSF tensor."""
    from satsim.image.psf import gen_gaussian
    return gen_gaussian(64, 64, 3.0)
