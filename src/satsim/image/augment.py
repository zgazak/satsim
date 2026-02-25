"""Image augmentation using PyTorch.

Replaces TFA translate/rotate and tf.image ops with
torch.nn.functional operations (grid_sample, affine_grid, flip, interpolate).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from satsim.device import ensure_tensor
from satsim.geometry.sprite import load_sprite_from_file
from satsim.math.fft import fftconv2p
from satsim.image.fpa import add_counts


def _translate_image(image: torch.Tensor, dx: float, dy: float,
                     interpolation: str = 'nearest') -> torch.Tensor:
    """Translate a 2D image by (dx, dy) pixels using grid_sample.

    Args:
        image: 2D tensor [H, W].
        dx: Horizontal shift in pixels.
        dy: Vertical shift in pixels.
        interpolation: 'nearest' or 'bilinear'.

    Returns:
        Translated 2D tensor.
    """
    h, w = image.shape
    # Normalize shifts to [-1, 1] range
    shift_x = 2.0 * dx / w
    shift_y = 2.0 * dy / h

    # Build affine matrix: identity + translation
    theta = torch.tensor([[1, 0, -shift_x],
                          [0, 1, -shift_y]], dtype=torch.float32)
    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, [1, 1, h, w], align_corners=False)
    img_4d = image.float().unsqueeze(0).unsqueeze(0)

    mode = 'nearest' if interpolation == 'nearest' else 'bilinear'
    out = F.grid_sample(img_4d, grid, mode=mode, padding_mode='zeros', align_corners=False)
    return out.squeeze(0).squeeze(0)


def _rotate_image(image: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate a 2D image by angle (radians) using grid_sample.

    Args:
        image: 2D tensor [H, W].
        angle_rad: Rotation angle in radians.

    Returns:
        Rotated 2D tensor.
    """
    h, w = image.shape
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    theta = torch.tensor([[cos_a, -sin_a, 0],
                          [sin_a,  cos_a, 0]], dtype=torch.float32)
    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, [1, 1, h, w], align_corners=False)
    img_4d = image.float().unsqueeze(0).unsqueeze(0)
    out = F.grid_sample(img_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return out.squeeze(0).squeeze(0)


def scatter_shift_random(image, t, loc, scale, length, spacial_osf=1,
                         normalize=True, mode='fft', interpolation='nearest',
                         dtype=torch.float32):
    """Shift image randomly in multiple directions and combine.

    Args:
        image: Image to shift.
        t: Simulation time (unused, required by interface).
        loc: Mean of normal distribution for shift magnitude.
        scale: Standard deviation of normal distribution.
        length: Number of random shifts.
        spacial_osf: Oversample factor.
        normalize: Normalize output sum to match input.
        mode: Algorithm: 'fft', 'roll', or 'shift'.
        interpolation: Interpolation mode: 'bilinear' or 'nearest'.
        dtype: Output dtype.

    Returns:
        Shifted 2D image.
    """
    if length <= 0:
        return ensure_tensor(image, dtype=dtype)

    mag = np.random.normal(loc, scale, size=length)
    angle = np.random.uniform(0, 360, size=length)

    return scatter_shift_polar(image, t, mag, angle, 1.0,
                               spacial_osf=spacial_osf, normalize=normalize,
                               mode=mode, interpolation=interpolation, dtype=dtype)


def scatter_shift_polar(image, t, mag, angle, weights, spacial_osf=1,
                        normalize=True, mode='fft', interpolation='nearest',
                        dtype=torch.float32):
    """Shift image by polar coordinates and combine.

    Args:
        image: Image to shift.
        t: Simulation time (unused, required by interface).
        mag: Shift magnitude in pixels.
        angle: Shift direction in degrees.
        weights: Scale factor for each shift.
        spacial_osf: Oversample factor.
        normalize: Normalize output sum to match input.
        mode: Algorithm: 'fft', 'roll', or 'shift'.
        interpolation: Interpolation mode.
        dtype: Output dtype.

    Returns:
        Shifted 2D image.
    """
    image = ensure_tensor(image, dtype=dtype)
    mag = ensure_tensor(mag, dtype=dtype)
    angle = ensure_tensor(angle, dtype=dtype)

    angle_rad = angle * math.pi / 180.0
    x = mag * torch.sin(angle_rad)
    y = mag * torch.cos(angle_rad)

    return scatter_shift(image, t, y, x, weights, spacial_osf, normalize,
                         mode=mode, interpolation=interpolation, dtype=dtype)


def scatter_shift(image, t, y, x, weights, spacial_osf=1, normalize=True,
                  mode='fft', interpolation='nearest', dtype=torch.float32):
    """Shift image by cartesian coordinates and combine.

    Args:
        image: Image to shift.
        t: Simulation time (unused, required by interface).
        y: Vertical shift in pixels.
        x: Horizontal shift in pixels.
        weights: Scale factor for each shift.
        spacial_osf: Oversample factor.
        normalize: Normalize output sum to match input.
        mode: Algorithm: 'fft', 'roll', or 'shift'.
        interpolation: Interpolation mode.
        dtype: Output dtype.

    Returns:
        Shifted 2D image.
    """
    orig_image = ensure_tensor(image, dtype=dtype)
    result = torch.zeros_like(orig_image)
    y = ensure_tensor(y * spacial_osf, dtype=dtype)
    x = ensure_tensor(x * spacial_osf, dtype=dtype)
    weights_t = ensure_tensor(weights, dtype=dtype).expand_as(y)

    if mode == 'roll':
        y_int = y.int()
        x_int = x.int()
        for i in range(len(y_int)):
            shifted = torch.roll(orig_image, shifts=(int(y_int[i]), int(x_int[i])), dims=(0, 1))
            result = result + shifted * weights_t[i]
    elif mode == 'fft':
        h, w = orig_image.shape
        rr = (y + h / 2).int()
        cc = (x + w / 2).int()
        delta = add_counts(torch.zeros_like(orig_image), rr - 1, cc - 1, weights_t)
        result = fftconv2p(delta, orig_image, pad=1)
    else:
        for i in range(len(y)):
            shifted = _translate_image(orig_image, float(x[i]), float(y[i]), interpolation)
            result = result + shifted * weights_t[i]

    if normalize:
        orig_sum = orig_image.sum()
        result_sum = result.sum()
        if result_sum != 0:
            result = result * orig_sum / result_sum

    return result


def crop_and_resize(image, t, y_start, x_start, y_box_size, x_box_size):
    """Crop image and resize to original size.

    Args:
        image: Image to crop.
        t: Simulation time (unused, required by interface).
        y_start: Starting y in normalized coordinates [0,1].
        x_start: Starting x in normalized coordinates [0,1].
        y_box_size: Box height in normalized coordinates.
        x_box_size: Box width in normalized coordinates.

    Returns:
        Cropped and resized 2D image.
    """
    image = ensure_tensor(image)
    h, w = image.shape

    y1 = y_start
    x1 = x_start
    y2 = min(y1 + y_box_size, 1.0)
    x2 = min(x1 + x_box_size, 1.0)

    # Convert normalized coords to pixel coords
    py1 = int(y1 * h)
    py2 = int(y2 * h)
    px1 = int(x1 * w)
    px2 = int(x2 * w)

    cropped = image[py1:py2, px1:px2]

    # Resize back to original size
    resized = F.interpolate(
        cropped.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )
    return resized.squeeze(0).squeeze(0)


def flip(image, t, up_down=False, left_right=False):
    """Flip image about y and/or x axis.

    Args:
        image: Image to flip.
        t: Simulation time (unused, required by interface).
        up_down: Flip about x axis.
        left_right: Flip about y axis.

    Returns:
        Flipped 2D image.
    """
    if not up_down and not left_right:
        return image

    image = ensure_tensor(image)

    if up_down:
        image = torch.flip(image, [0])
    if left_right:
        image = torch.flip(image, [1])

    return image


def null(image, t):
    """Identity function.

    Args:
        image: Input image.
        t: Simulation time.

    Returns:
        Same as input image.
    """
    return image


def resize(image, t, height, width, spacial_osf=1, normalize=True, dtype=torch.float32):
    """Resize image.

    Args:
        image: Input image.
        t: Simulation time.
        height: New height.
        width: New width.
        spacial_osf: Multiply height and width by this.
        normalize: Normalize output sum to match input.
        dtype: Output dtype.

    Returns:
        Resized 2D image.
    """
    orig_image = ensure_tensor(image, dtype=dtype)
    resized = F.interpolate(
        orig_image.unsqueeze(0).unsqueeze(0),
        size=(height * spacial_osf, width * spacial_osf),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    if normalize:
        orig_sum = orig_image.sum()
        resized_sum = resized.sum()
        if resized_sum != 0:
            resized = resized * orig_sum / resized_sum

    return resized


def rotate(image, t, angle=0, rate=0):
    """Rotate image.

    Args:
        image: Input image.
        t: Simulation time in seconds from epoch.
        angle: Angle to rotate in degrees.
        rate: Rotation rate in degrees per second.

    Returns:
        Rotated 2D image.
    """
    image = ensure_tensor(image)
    angle_rad = (angle + rate * t) * math.pi / 180.0
    return _rotate_image(image, angle_rad)


def load_from_file(image, t, filename, normalize=True, dtype=torch.float32):
    """Load image from file.

    Args:
        image: Base image (ignored).
        t: Simulation time (unused).
        filename: File to load.
        normalize: Normalize so sum equals 1.
        dtype: Output dtype.

    Returns:
        Loaded image as 2D tensor.
    """
    return load_sprite_from_file(filename, normalize, dtype)


def pow(image, t, exponent=1, normalize=True, dtype=torch.float32):
    """Raise image to a power.

    Args:
        image: Base image.
        t: Simulation time.
        exponent: Power exponent.
        normalize: Normalize output sum to match input.
        dtype: Output dtype.

    Returns:
        Image raised to power.
    """
    orig_image = ensure_tensor(image, dtype=dtype)
    result = orig_image ** exponent

    if normalize:
        orig_sum = orig_image.sum()
        result_sum = result.sum()
        if result_sum != 0:
            result = result * orig_sum / result_sum

    return result
