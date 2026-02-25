"""PyTorch device management.

CPU-first design with global device and context manager for overrides.
Replaces satsim.util.system (TensorFlow GPU configuration).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

_global_device: torch.device = torch.device("cpu")


@dataclass
class DeviceConfig:
    """Configuration for PyTorch device selection.

    Args:
        device_type: "auto", "cpu", or "cuda".
        device_id: CUDA device index (ignored for CPU).
        memory_limit_mb: Optional GPU memory limit in MB.
    """
    device_type: str = "auto"
    device_id: int = 0
    memory_limit_mb: int | None = None

    def resolve(self) -> torch.device:
        if self.device_type == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda", self.device_id)
            else:
                dev = torch.device("cpu")
        elif self.device_type == "cpu":
            dev = torch.device("cpu")
        elif self.device_type == "cuda":
            dev = torch.device("cuda", self.device_id)
        else:
            raise ValueError(f"Unknown device_type: {self.device_type!r}")

        if dev.type == "cuda" and self.memory_limit_mb is not None:
            fraction = self.memory_limit_mb / (torch.cuda.get_device_properties(dev).total_mem / 1024 / 1024)
            torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0), dev)

        logger.info("Resolved device: %s", dev)
        return dev


def get_device() -> torch.device:
    """Return the current global device."""
    return _global_device


def set_device(device: torch.device | str) -> None:
    """Set the global device."""
    global _global_device
    if isinstance(device, str):
        device = torch.device(device)
    _global_device = device
    logger.info("Global device set to: %s", device)


@contextmanager
def device_context(device: torch.device | str):
    """Context manager to temporarily override the global device."""
    global _global_device
    old = _global_device
    if isinstance(device, str):
        device = torch.device(device)
    _global_device = device
    try:
        yield device
    finally:
        _global_device = old


def ensure_tensor(x, dtype=torch.float32, device: torch.device | None = None) -> torch.Tensor:
    """Convert input to a torch.Tensor on the specified device.

    Handles numpy arrays, Python scalars, lists, and existing tensors.
    """
    if device is None:
        device = _global_device

    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def is_running_on_cpu() -> bool:
    """Return True if the global device is CPU."""
    return _global_device.type == "cpu"
