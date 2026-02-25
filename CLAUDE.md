# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SatSim is a high-fidelity space surveillance optical scene simulation environment built on PyTorch for GPU-accelerated (CUDA) synthetic image generation. It produces labeled training data for deep learning satellite detection systems. Licensed under AFRL Public Release #AFRL-2022-1116.

The codebase was rewritten from TensorFlow to PyTorch (v2.0). Legacy TF-based tests are preserved in `tests/_legacy/` for reference.

## Build & Development Commands

```bash
pip install -e ".[dev]"       # Install in dev mode with test dependencies
pip install -e ".[dev,vision]" # Also include torchvision/kornia
```

**Running tests:**
```bash
pytest                                      # All tests (excludes legacy)
pytest tests/test_image/ -v                 # Single test directory
pytest tests/test_config/test_loading.py    # Single test file
pytest tests/test_config/test_loading.py::TestRealize::test_static_config  # Single test
pytest -m "not gpu and not slow"            # Skip GPU/slow tests (CI default)
pytest -m gpu                               # GPU-only tests
```

**Linting:**
```bash
ruff check src/ tests/                      # Lint
ruff format src/ tests/                     # Format
```

**CLI usage:**
```bash
satsim run config.json                          # Default device (auto)
satsim run --device cuda:0 config.json          # Specific GPU
satsim run --device cpu config.json             # CPU only
satsim run --seed 42 -o output/ config.json     # With seed and output dir
satsim --debug INFO run config.json             # With logging
```

## Architecture

### Source Layout

Uses `src/` layout with `pyproject.toml` (hatchling build system, hatch-vcs versioning):

```
src/satsim/
  __init__.py          # Public API: load_yaml, load_json, realize, generate, generate_frames, run, set_device
  _version.py          # Version from importlib.metadata
  device.py            # PyTorch device management (DeviceConfig, get_device, set_device, ensure_tensor)
  cli.py               # Click CLI
  config/              # Config loading: dataclasses.py, loading.py, seeding.py
  pipeline/            # Rendering: engine.py (generate_frames), scene.py, frame.py (FrameResult)
  image/               # FPA, PSF, noise, render, augment (all PyTorch)
  math/                # FFT conv (PyTorch), angle, const, interpolate, random (numpy)
  geometry/            # Astrometric, ephemeris, WCS, star catalogs, transforms (mostly numpy)
  io/                  # Writers, SatNet, FITS, CZML, analytical
  generator/obs/       # Object generators (cone, circle, sphere, CSO, breakup)
  radar/               # Radar simulator (separate from EO pipeline)
  vecmath/             # Cartesian, Matrix, Quaternion
  time/                # UTC time utilities
  pipeline_functions/  # Dynamic parameter functions (constant, sin, cos, poly, glint)
  util/                # Python, thread, timer utilities
```

### Image Generation Pipeline

```
load_yaml/load_json() → realize() → generate_frames() → [yields FrameResult]
                                                                ↓ (optional)
                                                          write_collection()
```

1. **Config Loading** (`config/loading.py`) — Parse JSON/YAML, resolve `$sample`/`$ref`/`$generator`/`$function`/`$compound`/`$import` keywords
2. **Config Realization** (`config/loading.py: realize()`) — Resolve `Sampled` values using `SeedTree` for deterministic per-field RNG
3. **Scene Setup** (`pipeline/scene.py`) — Calculate satellite/star positions via Skyfield/SGP4
4. **Frame Rendering** (`pipeline/engine.py: generate_frames()`) — Yields `FrameResult` dataclasses, zero I/O
5. **Image Rendering** (`image/render.py`) — Render objects onto padded FPA via `render_full`/`render_piecewise`
6. **PSF Convolution** (`image/psf.py` + `math/fft.py`) — FFT-based convolution with cached PSF transforms
7. **Noise & Digitization** (`image/fpa.py`, `image/noise.py`) — Photon noise (`torch.poisson`), read noise, dark current, A/D conversion
8. **Output** (`io/writers.py`) — Consumes `FrameResult`, writes SatNet/FITS/CZML

### Device Management (`device.py`)

CPU-first design. Global device with `set_device()`/`get_device()`. `ensure_tensor()` replaces all `tf.cast()` calls. `DeviceConfig` dataclass resolves `auto`/`cpu`/`cuda` to a `torch.device`.

### Configuration System (`config/`)

- `dataclasses.py` — Typed config: `SatSimConfig`, `FPAConfig`, `GeometryConfig`, etc. `Sampled` wraps deferred random values.
- `loading.py` — Backward-compatible with JSON DSL keywords. `transform()` for legacy configs, `realize()` for new typed configs.
- `seeding.py` — `SeedTree` uses SHA-256 path hashing for deterministic per-field RNG. Changing one field doesn't affect others.

### Key PyTorch Migration Patterns

- `tf.signal.rfft2d` → `torch.fft.rfft2`; `F.pad(x, (left, right, top, bottom))` (reversed from TF)
- `tf.nn.avg_pool` → `F.avg_pool2d` (needs unsqueeze/squeeze for batch/channel dims)
- `tf.tensor_scatter_nd_add` → `scatter_add_` (non-deterministic on GPU with duplicate indices)
- TFA translate/rotate → `F.affine_grid` + `F.grid_sample` (always pass `align_corners=False`)
- `tf.compat.v1.random.poisson` → `torch.poisson(x.clamp(min=0))` (use float64 for multi-sample path)
- PSF FFT cache: `dict` keyed by `id(tensor)` (can't hash PyTorch tensors)

## Testing

- Framework: pytest with markers: `gpu`, `slow`, `integration`
- New tests: `tests/test_cli.py`, `tests/test_math/`, `tests/test_image/`, `tests/test_geometry/`, `tests/test_config/`, `tests/test_pipeline/`
- Existing pure-numpy tests: `tests/test_astrometric.py`, `tests/test_cartesian.py`, `tests/test_satnet.py`, etc.
- Legacy TF tests: `tests/_legacy/` (not run)
- Test configs: `tests/config_*.json` and `tests/config.yml`
- conftest.py: `_reset_device` (autouse), `gpu_device`, `minimal_config`, `sample_image`, `sample_psf` fixtures

## Linting

Ruff (replaces flake8): target Python 3.12, line length 120, selects E/F/W/I/UP, ignores E501.

## Version Management

Uses `hatch-vcs` (git tags). Fallback version: `2.0.0.dev0`. Read at runtime via `importlib.metadata`.

## JSON Schema

Configuration schema lives in `schema/v1/` with `Document.json` as root.
