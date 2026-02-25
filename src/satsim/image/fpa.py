"""Focal Plane Array simulation using PyTorch.

Replaces TF ops (avg_pool, conv2d, scatter_nd_add, while_loop) with
PyTorch equivalents (F.avg_pool2d, F.conv2d, scatter_add_, for loops).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from satsim.device import ensure_tensor, is_running_on_cpu
from satsim.geometry.transform import rotate_and_translate
from satsim.math.fft import fftconv2p


MAX_PIXEL_VALUE = {
    'int8': 127.0,
    'uint8': 255.0,
    'int16': 32767.0,
    'uint16': 65535.0,
    'int32': 2147483647.0,
    'uint32': 4294967295.0,
}


def downsample(fpa: torch.Tensor, osf: int, method: str = 'conv2d') -> torch.Tensor:
    """Downsample a 2D oversampled image tensor.

    Args:
        fpa: Input image as a 2D tensor in oversampled pixel space.
        osf: Oversample factor.
        method: 'conv2d' or 'pool'.

    Returns:
        Downsampled 2D image tensor.
    """
    if method == 'pool':
        n = float(osf * osf)
        # F.avg_pool2d needs [N, C, H, W]
        out = F.avg_pool2d(fpa.unsqueeze(0).unsqueeze(0), osf, stride=osf, padding=0)
        return (out.squeeze(0).squeeze(0)) * n
    else:
        filt = torch.ones((1, 1, osf, osf), dtype=fpa.dtype, device=fpa.device)
        out = F.conv2d(
            fpa.unsqueeze(0).unsqueeze(0),
            filt,
            stride=osf,
            padding=0
        )
        return out.squeeze(0).squeeze(0)


def crop(fpa: torch.Tensor, y_pad: int, x_pad: int, y_size: int, x_size: int) -> torch.Tensor:
    """Crop a 2D image tensor.

    Args:
        fpa: Input image tensor.
        y_pad: Starting y pixel (number of pad pixels at top).
        x_pad: Starting x pixel (number of pad pixels on left).
        y_size: Total y pixels to keep.
        x_size: Total x pixels to keep.

    Returns:
        Cropped 2D image tensor.
    """
    y_pad = int(y_pad)
    x_pad = int(x_pad)
    y_size = int(y_size)
    x_size = int(x_size)
    return fpa[y_pad:y_pad + y_size, x_pad:x_pad + x_size]


def analog_to_digital(fpa: torch.Tensor, gain: float, fwc: float, bias: float = 0,
                       dtype: str = 'uint16', saturated_pixel_model: str = 'max') -> torch.Tensor:
    """Convert photoelectron counts to digital counts.

    Args:
        fpa: Input image in real pixel space (photoelectrons).
        gain: Digital to analog multiplier.
        fwc: Full well capacity in photoelectrons.
        bias: Bias to add in digital counts.
        dtype: Pixel data type string.
        saturated_pixel_model: 'max' clips to dtype max value.

    Returns:
        2D image tensor as digital counts.
    """
    fpa_with_bias = fpa + (bias * gain)
    fpa_digital = torch.where(fpa_with_bias < fwc, fpa_with_bias, torch.full_like(fpa, fwc))
    fpa_digital = torch.floor(fpa_digital / gain)
    fpa_digital = torch.where(fpa_digital > 0, fpa_digital, torch.zeros_like(fpa))

    if saturated_pixel_model == 'max':
        if dtype in MAX_PIXEL_VALUE:
            max_val = MAX_PIXEL_VALUE[dtype]
            fpa_digital = torch.where(fpa_digital < max_val, fpa_digital, torch.full_like(fpa, max_val))

    return fpa_digital


def mv_to_pe(zeropoint, mv):
    """Convert visual magnitude to photoelectrons per second.

    Args:
        zeropoint: Zeropoint of the FPA.
        mv: Visual magnitude.

    Returns:
        Photoelectrons per second.
    """
    zp = np.asarray(zeropoint)
    mv = np.asarray(mv)
    return 10 ** ((zp - mv) / 2.5)


def pe_to_mv(zeropoint, pe):
    """Convert photoelectrons per second to visual magnitude.

    Args:
        zeropoint: Zeropoint of the FPA.
        pe: Brightness in photoelectrons per second.

    Returns:
        Visual magnitude.
    """
    zp = np.asarray(zeropoint)
    pe = np.asarray(pe)
    return -2.5 * np.log10(pe) + zp


def add_patch(fpa: torch.Tensor, r, c, cnt, patch: torch.Tensor,
              r_offset: int = 0, c_offset: int = 0, mode: str = 'fft') -> torch.Tensor:
    """Add a patch to the image centered about each (row, col) coordinate.

    Args:
        fpa: Input image as a 2D tensor.
        r: Row pixel coordinates.
        c: Column pixel coordinates.
        cnt: Absolute counts (dn or pe).
        patch: Patch image as a 2D tensor.
        r_offset: Offset to add to r values.
        c_offset: Offset to add to c values.
        mode: Render mode: 'fft' or 'overlay'.

    Returns:
        Modified image.
    """
    patch_rows, patch_cols = patch.shape
    patch_rows_div2 = patch_rows // 2
    patch_cols_div2 = patch_cols // 2

    r_offset = float(r_offset)
    c_offset = float(c_offset)

    # Expand fpa to fit image and patch
    fpa = F.pad(fpa, (patch_cols, patch_cols, patch_rows, patch_rows))
    fpa_rows, fpa_cols = fpa.shape

    rr = (ensure_tensor(r) + r_offset).int()
    cc = (ensure_tensor(c) + c_offset).int()

    if mode == 'fft':
        patch_full = _to_shape(patch, fpa)
        delta = add_counts(torch.zeros_like(fpa), rr + 1, cc + 1, cnt)
        fpa = fftconv2p(delta, patch_full, pad=1)
    else:
        cnt_t = ensure_tensor(cnt)
        for i in range(len(rr)):
            ri = int(rr[i])
            ci = int(cc[i])
            r_end = ri + patch_rows
            c_end = ci + patch_cols

            overlay = cnt_t[i] * patch

            if ri >= 0 and ci >= 0 and r_end < fpa_rows and c_end < fpa_cols:
                fpa[ri:r_end, ci:c_end] = fpa[ri:r_end, ci:c_end] + overlay

    # Crop so patches are centered
    fpa = fpa[patch_rows_div2:fpa_rows - patch_rows + patch_rows_div2,
              patch_cols_div2:fpa_cols - patch_cols + patch_cols_div2]

    return fpa


def add_counts(fpa: torch.Tensor, r, c, cnt, r_offset: int = 0, c_offset: int = 0) -> torch.Tensor:
    """Add counts (dn, pe) to the input image at specified pixel locations.

    Args:
        fpa: Input image as a 2D tensor.
        r: Row pixel coordinates.
        c: Column pixel coordinates.
        cnt: Absolute counts.
        r_offset: Offset for r values.
        c_offset: Offset for c values.

    Returns:
        Modified image.
    """
    r = ensure_tensor(r, dtype=torch.int64) + int(r_offset)
    c = ensure_tensor(c, dtype=torch.int64) + int(c_offset)
    cnt = ensure_tensor(cnt, dtype=fpa.dtype)

    h, w = fpa.shape

    # Bounds checking
    valid = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    r = r[valid]
    c = c[valid]
    cnt = cnt[valid]

    flat_idx = r * w + c
    result = fpa.clone()
    result.view(-1).scatter_add_(0, flat_idx, cnt)

    return result


def transform_and_fft(fpa: torch.Tensor, r, c, cnt, t_start, t_end, t_osf,
                       rotation, translation) -> torch.Tensor:
    """Apply transformations to center point and smear all points with FFT.

    Creates a motion blur PSF from the center point transformation and
    applies it to all point sources via FFT convolution.

    Args:
        fpa: Input image as a 2D tensor.
        r: Row pixel coordinates.
        c: Column pixel coordinates.
        cnt: Absolute counts.
        t_start: Start time in seconds from epoch.
        t_end: End time in seconds from epoch.
        t_osf: Temporal oversample factor.
        rotation: Clockwise rotation rate in radians/sec.
        translation: [row, col] translation rate in pixels/sec.

    Returns:
        Modified image.
    """
    h, w = fpa.shape
    h_f = float(h)
    w_f = float(w)
    h_minus_1 = h_f - 1.0
    w_minus_1 = w_f - 1.0
    r = ensure_tensor(r)
    c = ensure_tensor(c)
    cnt = ensure_tensor(cnt)

    h_mid = h_minus_1 / 2.0
    w_mid = w_minus_1 / 2.0

    # Create PSF by transforming a center point source
    blur = transform_and_add_counts(
        torch.zeros_like(fpa), [h_mid], [w_mid], [1.0],
        t_start, t_end, int(t_osf), rotation, translation
    )

    # Create delta functions at mid-exposure positions
    (rr, cc) = rotate_and_translate(h_minus_1, w_minus_1, r, c, 0.0, rotation, translation)
    delta = add_counts(torch.zeros_like(fpa), rr, cc, cnt)

    return fftconv2p(delta, blur, pad=1)


def transform_and_add_counts(fpa: torch.Tensor, r, c, cnt, t_start, t_end, t_osf,
                              rotation, translation, batch_size: int = 500,
                              filter_out_of_bounds: bool = True) -> torch.Tensor:
    """Apply discrete rotation/translation transformations to points.

    Smears points between t_start and t_end with t_osf discrete steps.
    Total energy is conserved.

    Args:
        fpa: Input image as a 2D tensor.
        r: Row pixel coordinates.
        c: Column pixel coordinates.
        cnt: Absolute counts.
        t_start: Start time in seconds from epoch.
        t_end: End time in seconds from epoch.
        t_osf: Temporal oversample factor.
        rotation: Clockwise rotation rate in radians/sec.
        translation: [row, col] translation rate in pixels/sec.
        batch_size: Points to process together.
        filter_out_of_bounds: Remove points fully outside bounds.

    Returns:
        Modified image.
    """
    h, w = fpa.shape
    h_f = float(h)
    w_f = float(w)
    h_minus_1 = h_f - 1.0
    w_minus_1 = w_f - 1.0
    r = ensure_tensor(r)
    c = ensure_tensor(c)
    cnt = ensure_tensor(cnt)
    t_osf = int(t_osf)

    cnt_os = cnt / float(t_osf)

    if filter_out_of_bounds:
        (rr0, cc0) = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_start, rotation, translation)
        (rr1, cc1) = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_end, rotation, translation)
        rr_lt_0 = (rr0 < 0) & (rr1 < 0)
        rr_gt_h = (rr0 > h_f) & (rr1 > h_f)
        cc_lt_0 = (cc0 < 0) & (cc1 < 0)
        cc_gt_w = (cc0 > w_f) & (cc1 > w_f)
        out_of_bounds = (rr_lt_0 | rr_gt_h) | (cc_lt_0 | cc_gt_w)
        in_bounds = ~out_of_bounds

        r = r[in_bounds]
        c = c[in_bounds]
        cnt_os = cnt_os[in_bounds]

    n_points = r.shape[0]
    if n_points == 0:
        return fpa

    batch_size = min(batch_size, n_points)

    # Batch the points
    r_batch = _to_batch_1d(r, batch_size)
    c_batch = _to_batch_1d(c, batch_size)
    cnt_os_batch = _to_batch_1d(cnt_os, batch_size)

    image = fpa
    tt = torch.linspace(t_start, t_end, t_osf, dtype=torch.float32, device=fpa.device)

    for i in range(r_batch.shape[0]):
        # Vectorize: repeat time steps for each point in batch
        tt_rep = tt.repeat_interleave(batch_size)
        rr = r_batch[i].repeat(t_osf)
        cc = c_batch[i].repeat(t_osf)
        cnt_os_rep = cnt_os_batch[i].repeat(t_osf)

        (rr_t, cc_t) = rotate_and_translate(h_minus_1, w_minus_1, rr, cc, tt_rep, rotation, translation)
        image = add_counts(image, rr_t, cc_t, cnt_os_rep)

    return image


def _to_batch_1d(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reshape 1D tensor into batches, padding if necessary."""
    n = x.shape[0]
    remainder = batch_size - (n % batch_size)
    if remainder != batch_size:
        x = F.pad(x, (0, remainder))
    num_batches = x.shape[0] // batch_size
    return x.reshape(num_batches, batch_size)


def _to_shape(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pad tensor a to match shape of tensor b."""
    y_, x_ = b.shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return F.pad(a, (x_pad // 2, x_pad // 2 + x_pad % 2, y_pad // 2, y_pad // 2 + y_pad % 2))
