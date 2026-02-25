import numpy as np
from astropy.io import fits

import torch
from satsim.device import ensure_tensor


def load_sprite_from_file(filename, normalize=True, dtype=torch.float32):
    """Load an image or sprite from file based on filename extension.
    Supported formats: FITS

    Args:
        filename: `str`, the image file name.
        normalize: `boolean`, normalize the sprite to 1. Default=True.
        dtype: torch dtype for the output tensor.

    Returns:
        A `Tensor`, the image, None if file type is not recognized
    """
    if filename.endswith('.fits'):

        hdul = fits.open(filename)
        img = ensure_tensor(hdul[0].data, dtype=dtype)

        if normalize:
            img = (img / img.sum()).to(dtype)

        return img

    else:

        return None
