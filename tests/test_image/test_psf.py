"""Tests for image.psf module."""

import torch
import pytest

from satsim.image.psf import eod_to_sigma, gen_gaussian


class TestEodToSigma:
    def test_high_eod(self):
        sigma = eod_to_sigma(0.9, 11)
        assert sigma > 0

    def test_low_eod(self):
        sigma_low = eod_to_sigma(0.1, 11)
        sigma_high = eod_to_sigma(0.9, 11)
        assert sigma_low > sigma_high


class TestGenGaussian:
    def test_shape(self):
        psf = gen_gaussian(64, 64, 3.0)
        assert psf.shape == (64, 64)

    def test_normalized(self):
        psf = gen_gaussian(101, 101, 5.0)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.01)

    def test_center_peak(self):
        psf = gen_gaussian(101, 101, 5.0)
        center = psf[50, 50]
        assert center == psf.max()

    def test_symmetry(self):
        psf = gen_gaussian(101, 101, 5.0)
        torch.testing.assert_close(psf[40, 50], psf[60, 50], atol=1e-6, rtol=1e-6)
