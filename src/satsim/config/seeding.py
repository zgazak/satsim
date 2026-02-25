"""Deterministic seeding for config field sampling.

SeedTree provides per-field deterministic RNG based on dotted config paths.
Changing one field's sampling doesn't affect others.
"""

from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass


@dataclass
class SeedTree:
    """Deterministic RNG keyed by config field path.

    Example:
        tree = SeedTree(42)
        rng1 = tree.rng("fpa.y_fov")  # always same RNG for this path
        rng2 = tree.rng("fpa.x_fov")  # different RNG, independent
    """
    root_seed: int | None = None

    def rng(self, path: str) -> np.random.RandomState:
        """Get a deterministic RNG for the given dotted path."""
        if self.root_seed is None:
            return np.random.RandomState()
        # Hash path with root seed for deterministic but independent seeds
        h = hashlib.sha256(f"{self.root_seed}:{path}".encode()).digest()
        seed = int.from_bytes(h[:4], 'little') % (2**31)
        return np.random.RandomState(seed)

    def child(self, prefix: str) -> SeedTree:
        """Create a child SeedTree with a path prefix baked in."""
        if self.root_seed is None:
            return SeedTree(None)
        h = hashlib.sha256(f"{self.root_seed}:{prefix}".encode()).digest()
        new_seed = int.from_bytes(h[:4], 'little') % (2**31)
        return SeedTree(new_seed)
