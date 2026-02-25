.. highlight:: shell

============
Installation
============


From PyPI
---------

.. code-block:: console

    $ pip install satsim

Requires Python 3.10+. PyTorch is installed automatically as a dependency.


From Source
-----------

Clone the repository:

.. code-block:: console

    $ git clone https://github.com/zgazak/satsim.git
    $ cd satsim

Install in development mode:

.. code-block:: console

    $ pip install -e ".[dev]"

This installs the package in editable mode along with test dependencies
(pytest, ruff, mypy).

To also install optional vision dependencies (torchvision, kornia):

.. code-block:: console

    $ pip install -e ".[dev,vision]"


GPU Support
-----------

SatSim runs on CPU by default. To use GPU acceleration, ensure you have a
CUDA-compatible PyTorch installation:

.. code-block:: console

    $ pip install torch --index-url https://download.pytorch.org/whl/cu121

Then select the device at runtime:

.. code-block:: console

    $ satsim run --device cuda:0 config.json

Or in Python:

.. code-block:: python

    import satsim
    satsim.set_device('cuda:0')
