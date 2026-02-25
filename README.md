SatSim
======

**SatSim source code was developed under contract with AFRL/RDSM, and is approved for public release under Public Affairs release approval
#AFRL-2022-1116.**

SatSim is a high-fidelity space surveillance optical scene simulation environment built on PyTorch. It generates synthetic labeled observation data for training and evaluating deep learning satellite detection models. SatSim supports systematic variation of scene parameters (object brightness, star drift rate, noise levels, etc.) and outputs in SatNet, FITS, CZML, and analytical formats.

![SatSim Example](satsim.jpg)

Quick Start
-----------

### Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. GPU acceleration is optional — SatSim runs on CPU by default.

### Run from the command line

```bash
# basic run
satsim run config.json

# specify output directory and device
satsim run --device cuda:0 --seed 42 -o output/ config.json

# CPU only
satsim run --device cpu config.json

# with debug logging
satsim --debug INFO run config.json
```

### Use as a Python library

```python
import satsim

# load and run a config file
config = satsim.load_json('config.json')
for frame in satsim.generate(config, seed=42):
    image = frame.fpa_digital.numpy()  # use as tensor or array

# or run directly to disk
satsim.run('config.json', output_dir='output/')

# opt-in GPU acceleration
satsim.set_device('cuda:0')
```

### Run tests

```bash
pytest                               # all tests
pytest -m "not gpu and not slow"     # skip GPU/slow tests
pytest tests/test_image/ -v          # single directory
```

Configuration
-------------

SatSim uses JSON or YAML configuration files. Most numeric parameters can use `$sample` for random sampling from any NumPy distribution:

```json
{
    "mv": { "$sample": "random.uniform", "low": 5.0, "high": 22.0 }
}
```

Other dynamic config keywords:
- **`$ref`** — reference another config path (e.g., `"$ref": "#/fpa/zeropoint"`)
- **`$generator`** — call a Python generator function to produce config
- **`$function`** — call an arbitrary Python function
- **`$compound`** — combine values with arithmetic operators
- **`$import`** — import settings from an external JSON file with optional overrides

See `schema/v1/Document.json` for the full configuration schema and `tests/config_static.json` for a complete example.

Documentation
-------------

* [CLAUDE.md](CLAUDE.md) — architecture overview and developer guide
* [History](HISTORY.md)

Versions
--------

* [History](HISTORY.md)
