"""Tests for math.fft module."""

import torch
import numpy as np
import pytest

from satsim.math.fft import fftshift, fftconv2, fftconv2p, clear_psf_cache


class TestFftshift:
    def test_2d_shift(self):
        x = torch.arange(16.0).reshape(4, 4)
        shifted = fftshift(x, dims=2)
        assert shifted.shape == (4, 4)
        # Center should be at [0,0] after shift
        assert shifted[0, 0] == x[2, 2]

    def test_1d_shift(self):
        x = torch.tensor([0., 1., 2., 3., 4., -5., -4., -3., -2., -1.])
        shifted = fftshift(x, dims=1)
        assert shifted[0].item() == pytest.approx(-5.0)

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            fftshift(torch.zeros(4, 4), dims=3)


class TestFftconv2:
    def test_delta_convolution(self):
        """Convolving with delta function should return the original."""
        n = 33  # odd size so center is unambiguous
        x = torch.randn(n, n)
        delta = torch.zeros(n, n)
        delta[n // 2, n // 2] = 1.0
        result = fftconv2(x, delta)
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_commutative(self):
        x = torch.randn(16, 16)
        y = torch.randn(16, 16)
        r1 = fftconv2(x, y)
        r2 = fftconv2(y, x)
        torch.testing.assert_close(r1, r2, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)
        result = fftconv2(x, y)
        assert result.shape == (32, 32)


class TestFftconv2p:
    def test_output_shape(self):
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)
        result = fftconv2p(x, y, pad=4)
        assert result.shape == (32, 32)

    def test_cache_consistency(self):
        clear_psf_cache()
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)
        r1 = fftconv2p(x, y, pad=4, cache_last_y=True)
        r2 = fftconv2p(x, y, pad=4, cache_last_y=True)
        torch.testing.assert_close(r1, r2)

    def test_no_cache(self):
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)
        r1 = fftconv2p(x, y, pad=4, cache_last_y=False)
        r2 = fftconv2p(x, y, pad=4, cache_last_y=True)
        torch.testing.assert_close(r1, r2, atol=1e-5, rtol=1e-5)
