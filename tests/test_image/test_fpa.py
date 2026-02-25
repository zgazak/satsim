"""Tests for image.fpa module."""

import torch
import numpy as np
import pytest

from satsim.image.fpa import (
    downsample, crop, analog_to_digital, mv_to_pe, pe_to_mv,
    add_counts, transform_and_add_counts,
)


class TestDownsample:
    def test_pool_method(self):
        fpa = torch.ones(32, 32)
        result = downsample(fpa, 2, method='pool')
        assert result.shape == (16, 16)
        # Each pixel should sum 4 oversampled pixels
        assert result[0, 0].item() == pytest.approx(4.0)

    def test_conv2d_method(self):
        fpa = torch.ones(32, 32)
        result = downsample(fpa, 2, method='conv2d')
        assert result.shape == (16, 16)
        assert result[0, 0].item() == pytest.approx(4.0)


class TestCrop:
    def test_basic_crop(self):
        fpa = torch.arange(100.0).reshape(10, 10)
        cropped = crop(fpa, 2, 2, 5, 5)
        assert cropped.shape == (5, 5)
        assert cropped[0, 0].item() == fpa[2, 2].item()


class TestA2D:
    def test_basic_conversion(self):
        fpa = torch.ones(10, 10) * 1000.0
        result = analog_to_digital(fpa, gain=2.0, fwc=50000.0)
        assert result.shape == (10, 10)
        assert result[0, 0].item() == 500.0

    def test_saturation(self):
        fpa = torch.ones(10, 10) * 1e9
        result = analog_to_digital(fpa, gain=1.0, fwc=100000.0, dtype='uint16')
        assert result.max().item() <= 65535.0

    def test_negative_clipping(self):
        fpa = torch.ones(10, 10) * -100.0
        result = analog_to_digital(fpa, gain=1.0, fwc=100000.0)
        assert (result >= 0).all()


class TestMagnitudeConversion:
    def test_zeropoint_identity(self):
        pe = mv_to_pe(20.0, 20.0)
        assert pe == pytest.approx(1.0)

    def test_round_trip(self):
        zp = 20.0
        mv = 15.0
        pe = mv_to_pe(zp, mv)
        mv_back = pe_to_mv(zp, pe)
        assert mv_back == pytest.approx(mv)


class TestAddCounts:
    def test_single_point(self):
        fpa = torch.zeros(10, 10)
        result = add_counts(fpa, [5], [5], [100.0])
        assert result[5, 5].item() == pytest.approx(100.0)
        assert result.sum().item() == pytest.approx(100.0)

    def test_out_of_bounds(self):
        fpa = torch.zeros(10, 10)
        result = add_counts(fpa, [-1, 5], [5, 5], [100.0, 50.0])
        assert result[5, 5].item() == pytest.approx(50.0)
        assert result.sum().item() == pytest.approx(50.0)
