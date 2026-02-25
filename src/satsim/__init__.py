"""SatSim - Satellite observation and scene simulator.

GPU-accelerated synthetic image generation using PyTorch.
"""

from satsim._version import __version__
from satsim.config.loading import load_json, load_yaml, realize
from satsim.pipeline.engine import generate_frames
from satsim.pipeline.frame import FrameResult
from satsim.device import get_device, set_device, DeviceConfig


def generate(config, seed=None):
    """Generate frames from a config dict.

    Args:
        config: Config dict (from load_yaml/load_json) or SatSimConfig.
        seed: Optional seed for reproducibility.

    Yields:
        FrameResult for each frame.
    """
    from satsim.config.loading import realize as _realize
    if isinstance(config, dict):
        realized = _realize(config, seed=seed)
    else:
        realized = config
    yield from generate_frames(realized)


def run(config_path, output_dir="./", seed=None, device="auto"):
    """High-level convenience: load config, generate, and write output.

    Args:
        config_path: Path to YAML or JSON config file.
        output_dir: Output directory.
        seed: Optional seed for reproducibility.
        device: Device string ('auto', 'cpu', 'cuda', 'cuda:0').
    """
    set_device(device if device != "auto" else
               "cuda" if __import__("torch").cuda.is_available() else "cpu")

    if config_path.endswith(('.yml', '.yaml')):
        config = load_yaml(config_path)
    else:
        config = load_json(config_path)

    from satsim.io.writers import write_collection
    write_collection(config, output_dir, seed=seed)


__all__ = [
    "__version__",
    "load_json",
    "load_yaml",
    "realize",
    "generate",
    "generate_frames",
    "FrameResult",
    "run",
    "get_device",
    "set_device",
    "DeviceConfig",
]
