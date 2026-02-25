"""Typed configuration dataclasses for SatSim.

These replace the JSON DSL ($sample/$ref/$generator/$function/$compound/$import).
Users write Python config or YAML that maps to these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sampled:
    """A value that will be resolved by realize() via random sampling.

    Replaces the $sample JSON keyword.

    Example:
        Sampled("uniform", low=5.0, high=22.0) -> random float in [5, 22]
    """
    distribution: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def __init__(self, distribution: str, seed: int | None = None, **kwargs):
        self.distribution = distribution
        self.seed = seed
        self.kwargs = kwargs

    def draw(self, rng=None):
        """Draw a sample from this distribution."""
        import numpy as np
        from satsim.math.random import gen_sample
        if rng is not None:
            return gen_sample(self.distribution, seed=None, **self.kwargs)
        return gen_sample(self.distribution, seed=self.seed, **self.kwargs)


@dataclass
class TimeConfig:
    exposure: float = 1.0
    gap: float = 0.0


@dataclass
class A2DConfig:
    gain: float = 1.0
    fwc: float = 100000.0
    bias: float = 0.0
    dtype: str = "uint16"


@dataclass
class NoiseConfig:
    read: float = 0.0
    electronic: float = 0.0
    photon: bool = True
    shot_noise_samples: int | None = None


@dataclass
class GaussianPSF:
    mode: str = "gaussian"
    eod: float = 0.8


@dataclass
class PoppyPSF:
    mode: str = "poppy"
    optical_system: list[dict] = field(default_factory=list)
    wavelengths: list[float] = field(default_factory=lambda: [600e-9])
    weights: list[float] = field(default_factory=lambda: [1.0])
    size: list[int] | None = None
    turbulent_atmosphere: dict | None = None


@dataclass
class NoPSF:
    mode: str = "none"


@dataclass
class FPAConfig:
    height: int = 512
    width: int = 512
    y_fov: float = 1.0
    x_fov: float = 1.0
    dark_current: float | Any = 0.0
    gain: float = 1.0
    bias: float = 0.0
    zeropoint: float = 20.0
    num_frames: int = 1
    time: TimeConfig = field(default_factory=TimeConfig)
    a2d: A2DConfig = field(default_factory=A2DConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    psf: GaussianPSF | PoppyPSF | NoPSF = field(default_factory=GaussianPSF)
    crop: dict | None = None


@dataclass
class StarMotion:
    mode: str = "none"


@dataclass
class SiteConfig:
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    name: str = "default"


@dataclass
class TrackConfig:
    mode: str = "rate"
    tle: list[str] | None = None
    ra: float | None = None
    dec: float | None = None
    az: float | None = None
    el: float | None = None


@dataclass
class ObsObject:
    mode: str = "list"
    list: list[dict] = field(default_factory=list)


@dataclass
class StarsConfig:
    mode: str = "sstr7"
    mv: float = 11.0
    motion: StarMotion = field(default_factory=StarMotion)
    catalog: str | None = None


@dataclass
class GeometryConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    obs: ObsObject = field(default_factory=ObsObject)
    stars: StarsConfig = field(default_factory=StarsConfig)
    time: dict = field(default_factory=dict)


@dataclass
class BackgroundConfig:
    galactic: float | Any = 0.0
    stray_light: Any = None


@dataclass
class SimConfig:
    spacial_osf: int = 1
    temporal_osf: int = 1
    padding: int = 0
    samples: int = 1
    show_obs_boxes: bool = True
    show_star_boxes: bool = False
    save_movie: bool = False
    save_czml: bool = False
    save_pickle: bool = False
    save_jpeg: bool = True
    fits_compression: str | None = None
    star_render_mode: str = "transform"
    render_mode: str = "full"
    render_size: list[int] | None = None
    calculate_snr: bool = True
    analytical_obs: bool = False
    apply_star_wrap_around: bool = False
    num_target_samples: int = 0
    enable_shot_noise: bool = True
    ground_truth: dict | None = None
    segmentation: dict | None = None


@dataclass
class SatSimConfig:
    """Root configuration for a SatSim simulation."""
    version: str = "2.0"
    sim: SimConfig = field(default_factory=SimConfig)
    fpa: FPAConfig = field(default_factory=FPAConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    seed: int | None = None
