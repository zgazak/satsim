"""Tests for image.noise module."""

import torch
import pytest

from satsim.image.noise import add_photon_noise, add_read_noise


class TestPhotonNoise:
    def test_nonnegative(self):
        fpa = torch.ones(32, 32) * 100.0
        noisy = add_photon_noise(fpa)
        assert noisy.shape == (32, 32)
        assert (noisy >= 0).all()

    def test_zero_input(self):
        fpa = torch.zeros(32, 32)
        noisy = add_photon_noise(fpa)
        assert (noisy == 0).all()

    def test_multi_sample_averaging(self):
        fpa = torch.ones(32, 32) * 1000.0
        noisy = add_photon_noise(fpa, samples=10)
        # Mean should be close to input
        assert abs(noisy.mean().item() - 1000.0) < 50.0

    def test_preserves_dtype(self):
        fpa = torch.ones(32, 32, dtype=torch.float32) * 100.0
        noisy = add_photon_noise(fpa)
        assert noisy.dtype == torch.float32


class TestReadNoise:
    def test_output_shape(self):
        fpa = torch.ones(32, 32) * 100.0
        result, noise = add_read_noise(fpa, 10.0)
        assert result.shape == (32, 32)
        assert noise.shape == (32, 32)

    def test_zero_noise(self):
        fpa = torch.ones(32, 32) * 100.0
        result, noise = add_read_noise(fpa, 0.0, 0.0)
        torch.testing.assert_close(result, fpa)

    def test_noise_magnitude(self):
        fpa = torch.zeros(1000, 1000)
        _, noise = add_read_noise(fpa, 10.0)
        assert abs(noise.std().item() - 10.0) < 1.0
