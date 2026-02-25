"""Tests for config.loading module."""

import os
import json
import pytest

from satsim.config.loading import load_json, load_yaml, realize, transform


@pytest.fixture
def static_config_path():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'config_static.json')
    if not os.path.exists(path):
        pytest.skip("config_static.json not found in test data")
    return path


class TestLoadJson:
    def test_load_valid(self, tmp_path):
        config = {"version": "2.0", "sim": {"samples": 1}}
        path = tmp_path / "test.json"
        path.write_text(json.dumps(config))
        result = load_json(str(path))
        assert result["version"] == "2.0"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/path.json")


class TestRealize:
    def test_static_config(self):
        config = {
            "version": "1.0",
            "sim": {"samples": 1},
            "fpa": {"height": 512},
        }
        result = realize(config)
        assert result["fpa"]["height"] == 512

    def test_sample_resolution(self):
        config = {
            "version": "1.0",
            "sim": {"samples": 1},
            "fpa": {
                "height": {"$sample": "random.uniform", "low": 100, "high": 100, "seed": 42},
            },
        }
        result = realize(config)
        assert result["fpa"]["height"] == pytest.approx(100.0)
