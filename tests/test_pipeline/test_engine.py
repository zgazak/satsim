"""Tests for pipeline.engine module (integration)."""

import pytest
import torch

from satsim.pipeline.engine import generate_frames
from satsim.pipeline.frame import FrameResult


@pytest.mark.integration
class TestGenerateFrames:
    def test_basic_generation(self, minimal_config):
        frames = list(generate_frames(minimal_config))
        assert len(frames) == 2
        for frame in frames:
            assert isinstance(frame, FrameResult)
            assert frame.fpa_digital is not None
            assert frame.fpa_digital.shape == (32, 32)

    def test_frame_numbers(self, minimal_config):
        frames = list(generate_frames(minimal_config))
        assert frames[0].frame_num == 0
        assert frames[1].frame_num == 1

    def test_to_numpy(self, minimal_config):
        frames = list(generate_frames(minimal_config))
        frame_np = frames[0].to_numpy()
        assert not isinstance(frame_np.fpa_digital, torch.Tensor)
