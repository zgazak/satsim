"""Tests for image.render module."""

import torch
import pytest

from satsim.image.render import render_full
from satsim.image.psf import gen_gaussian


class TestRenderFull:
    def test_empty_scene(self):
        h_os, w_os = 64, 64
        pad = 20
        osf = 1
        psf = gen_gaussian(h_os + 2 * pad, w_os + 2 * pad, 3.0)

        star, targ, _, _, _ = render_full(
            h_os, w_os, h_os + 2 * pad, w_os + 2 * pad, pad, pad,
            osf, psf,
            torch.tensor([]), torch.tensor([]), torch.tensor([]),
            torch.tensor([]), torch.tensor([]), torch.tensor([]),
            0.0, 1.0, 1, 0.0, [0.0, 0.0],
        )
        assert star.shape == (h_os, w_os)
        assert (star == 0).all()

    def test_single_target(self):
        h_os, w_os = 64, 64
        pad = 20
        osf = 1

        star, targ, _, _, _ = render_full(
            h_os, w_os, h_os + 2 * pad, w_os + 2 * pad, pad, pad,
            osf, None,
            torch.tensor([32.0 + pad]), torch.tensor([32.0 + pad]),
            torch.tensor([1000.0]),
            torch.tensor([]), torch.tensor([]), torch.tensor([]),
            0.0, 1.0, 1, 0.0, [0.0, 0.0],
        )
        assert targ.sum().item() > 0

    def test_no_psf_mode(self):
        h_os, w_os = 32, 32
        pad = 10
        osf = 1

        star, targ, _, _, _ = render_full(
            h_os, w_os, h_os + 2 * pad, w_os + 2 * pad, pad, pad,
            osf, None,
            torch.tensor([]), torch.tensor([]), torch.tensor([]),
            torch.tensor([]), torch.tensor([]), torch.tensor([]),
            0.0, 1.0, 1, 0.0, [0.0, 0.0],
        )
        assert star.shape == (h_os, w_os)
