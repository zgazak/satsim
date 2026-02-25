"""Tests for geometry.transform module."""

import torch
import pytest

from satsim.geometry.transform import rotate_and_translate


class TestRotateAndTranslate:
    def test_no_rotation_no_translation(self):
        r = torch.tensor([5.0])
        c = torch.tensor([5.0])
        rr, cc = rotate_and_translate(10.0, 10.0, r, c, 0.0, 0.0, [0.0, 0.0])
        torch.testing.assert_close(rr, r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cc, c, atol=1e-5, rtol=1e-5)

    def test_pure_translation(self):
        r = torch.tensor([5.0])
        c = torch.tensor([5.0])
        rr, cc = rotate_and_translate(10.0, 10.0, r, c, 1.0, 0.0, [2.0, 3.0])
        assert rr[0].item() == pytest.approx(7.0)
        assert cc[0].item() == pytest.approx(8.0)

    def test_multiple_points(self):
        r = torch.tensor([1.0, 2.0, 3.0])
        c = torch.tensor([4.0, 5.0, 6.0])
        rr, cc = rotate_and_translate(10.0, 10.0, r, c, 0.0, 0.0, [0.0, 0.0])
        assert len(rr) == 3
        assert len(cc) == 3
