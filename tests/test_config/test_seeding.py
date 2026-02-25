"""Tests for config.seeding module."""

import pytest
import numpy as np

from satsim.config.seeding import SeedTree


class TestSeedTree:
    def test_deterministic(self):
        tree = SeedTree(42)
        rng1 = tree.rng("fpa.y_fov")
        val1 = rng1.uniform()
        rng2 = SeedTree(42).rng("fpa.y_fov")
        val2 = rng2.uniform()
        assert val1 == val2

    def test_different_paths(self):
        tree = SeedTree(42)
        val1 = tree.rng("fpa.y_fov").uniform()
        val2 = tree.rng("fpa.x_fov").uniform()
        assert val1 != val2

    def test_none_seed(self):
        tree = SeedTree(None)
        rng = tree.rng("anything")
        assert isinstance(rng, np.random.RandomState)

    def test_child(self):
        tree = SeedTree(42)
        child = tree.child("geometry")
        val = child.rng("site.lat").uniform()
        assert isinstance(val, float)
