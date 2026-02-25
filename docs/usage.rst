=====
Usage
=====

Python Module
-------------

To use SatSim in a Python project:

.. code-block:: python

    import satsim

    # load a config file
    config = satsim.load_json('input/config.json')

    # generate frames as an iterator
    for frame in satsim.generate(config, seed=42):
        image = frame.fpa_digital.numpy()       # final digitized image
        stars = frame.fpa_star.numpy()           # star signal layer
        targets = frame.fpa_target.numpy()       # target signal layer

    # or run directly to disk
    satsim.run('input/config.json', output_dir='output/')

    # opt-in GPU acceleration
    satsim.set_device('cuda:0')

Command Line Interface
----------------------

To run SatSim from the command line:

.. code-block:: bash

    # show help menu
    $ satsim --help

    # show run help menu
    $ satsim run --help

    # generate images with default device (auto-selects GPU if available)
    $ satsim run -o output/ input/config.json

    # specify GPU device
    $ satsim run --device cuda:0 -o output/ input/config.json

    # CPU only
    $ satsim run --device cpu -o output/ input/config.json

    # set random seed for reproducibility
    $ satsim run --seed 42 -o output/ input/config.json

    # enable debug logging
    $ satsim --debug INFO run input/config.json

Parameters and Configuration
----------------------------

SatSim uses JSON or YAML configuration files. See ``tests/config_static.json``
for a complete example and ``schema/v1/Document.json`` for the full schema.

Most numeric parameters can be replaced with a ``$sample`` key to randomly
sample a value from any NumPy distribution:

.. code-block:: python

    {
        "mv": { "$sample": "random.uniform", "low": 5.0, "high": 22.0 }
    }

You can also provide a ``seed`` inside a ``$sample`` dictionary to make that
field deterministic without affecting other samples:

.. code-block:: python

    {
        "mv": { "$sample": "random.uniform", "low": 5.0, "high": 22.0, "seed": 123 }
    }

Other dynamic configuration keywords:

- **$ref** — reference another config path (e.g., ``"$ref": "#/fpa/zeropoint"``)
- **$generator** — call a Python generator function to produce config dynamically
- **$function** — call an arbitrary Python function with keyword arguments
- **$compound** — combine multiple values with arithmetic operators
- **$import** — import settings from an external JSON file with optional overrides

Here is a complete SatSim parameter example:

.. code-block:: python

    {
        "version": 1,
        "sim": {
            "mode": "fftconv2p",       # convolution mode
            "spacial_osf": 15,         # spatial oversampling factor
            "temporal_osf": 100,       # temporal oversampling factor
            "padding": 100,            # pixels to pad each side
            "samples": 1               # number of sets to generate
        },
        "fpa": {
            "height": 512,             # image height in pixels
            "width": 512,              # image width in pixels
            "y_fov": 0.308312,         # vertical field of view in degrees
            "x_fov": 0.308312,         # horizontal field of view in degrees
            "dark_current": 0.3,       # dark current in pe/second
            "gain": 1,                 # pixel response
            "bias": 0,                 # pixel bias in pe
            "zeropoint": 20.6663,      # instrument zeropoint
            "a2d": {
                "response": "linear",  # A/D converter response
                "fwc": 100000,         # full well capacity in pe
                "gain": 1.5,           # digital gain
                "bias": 1500           # readout bias in digital counts
            },
            "noise": {
                "read": 9,             # RMS read noise in pe
                "electronic": 0        # RMS electronic noise in pe
            },
            "psf": {
                "mode": "gaussian",    # PSF type (gaussian, poppy, none)
                "eod": 0.15            # energy on detector (0.0-1.0)
            },
            "time": {
                "exposure": 5.0,       # integration time in seconds
                "gap": 2.5             # gap between frames in seconds
            },
            "num_frames": 6            # frames per set
        },
        "background": {
            "stray": { "mode": "none" },
            "galactic": 19.5           # background in mv/arcsec^2/sec
        },
        "geometry": {
            "stars": {
                "mode": "bins",
                "mv": {
                    "bins": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                    "density": [0.019,0,0.006,0.017,0.036,0.039,0.097,0.669,2.478,5.003,10.269,24.328,35.192,60.017,110.06,180.28,285.53,446.14]
                },
                "motion": {
                    "mode": "affine",
                    "rotation": 0,
                    "translation": [0.4, 7.0]
                }
            },
            "obs": {
                "mode": "list",
                "list": {
                    "$sample": "random.list",
                    "length": { "$sample": "random.randint", "low": 0, "high": 15 },
                    "value": {
                        "mode": "line",
                        "origin": [
                            { "$sample": "random.uniform", "low": 0.1, "high": 0.9 },
                            { "$sample": "random.uniform", "low": 0.1, "high": 0.9 }
                        ],
                        "velocity": [
                            { "$sample": "random.uniform", "low": -0.01, "high": 0.01 },
                            { "$sample": "random.uniform", "low": -0.01, "high": 0.01 }
                        ],
                        "mv": { "$sample": "random.uniform", "low": 5.0, "high": 22.0 }
                    }
                }
            }
        }
    }

Generator Example
-----------------

SatSim's ``$generator`` feature calls Python functions to dynamically produce
configuration. Here is an example using the built-in ``cone`` generator:

.. code-block:: python

    "obs": {
        "generator": {
            "module": "satsim.generator.obs.geometry",
            "function": "cone",
            "kwargs": {
                "n": { "$sample": "random.randint", "low": 500, "high": 1000 },
                "t": 0,
                "direction": [
                    { "$sample": "random.uniform", "low": 0, "high": 360 },
                    { "$sample": "random.uniform", "low": 10, "high": 30 }
                ],
                "velocity": [
                    { "$sample": "random.uniform", "low": 1.0, "high": 5.0 },
                    { "$sample": "random.uniform", "low": 3.0, "high": 5.0 }
                ],
                "origin": [
                    { "$sample": "random.uniform", "low": 0.2, "high": 0.8 },
                    { "$sample": "random.uniform", "low": 0.2, "high": 0.8 }
                ],
                "mv": [15.0, 17.0]
            }
        }
    }

Import Example
--------------

Import partial configurations from external files with optional overrides:

.. code-block:: python

    "fpa": {
        "$import": "../common/random_raven_fpa.json",
        "override": {
            "num_frames": 10
        }
    }
