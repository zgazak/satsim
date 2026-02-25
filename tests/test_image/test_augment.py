"""Tests for image.augment module."""

import torch
import pytest

from satsim.image.augment import flip, crop_and_resize, resize, null, pow


class TestFlip:
    def test_no_flip(self):
        img = torch.arange(16.0).reshape(4, 4)
        result = flip(img, 0)
        torch.testing.assert_close(result, img)

    def test_up_down(self):
        img = torch.arange(16.0).reshape(4, 4)
        result = flip(img, 0, up_down=True)
        assert result[0, 0].item() == img[3, 0].item()

    def test_left_right(self):
        img = torch.arange(16.0).reshape(4, 4)
        result = flip(img, 0, left_right=True)
        assert result[0, 0].item() == img[0, 3].item()


class TestNull:
    def test_identity(self):
        img = torch.randn(10, 10)
        result = null(img, 0)
        assert result is img


class TestResize:
    def test_output_shape(self):
        img = torch.randn(32, 32)
        result = resize(img, 0, 64, 64)
        assert result.shape == (64, 64)


class TestPow:
    def test_identity(self):
        img = torch.ones(10, 10) * 2.0
        result = pow(img, 0, exponent=1)
        torch.testing.assert_close(result, img, atol=1e-5, rtol=1e-5)

    def test_square(self):
        img = torch.ones(10, 10) * 4.0
        result = pow(img, 0, exponent=2, normalize=False)
        torch.testing.assert_close(result, torch.ones(10, 10) * 16.0, atol=1e-5, rtol=1e-5)
