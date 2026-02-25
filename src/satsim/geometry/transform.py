"""Coordinate transformations using PyTorch.

Replaces tf.math.sin/cos with torch.sin/cos.
"""

from __future__ import annotations

import torch

from satsim.device import ensure_tensor


def rotate_and_translate(h_m1, w_m1, r, c, t, rotation, translation):
    """Rotate and translate a list of input pixel locations.

    Typically used to transform stars to simulate image drift and rotation
    during sidereal and rate track.

    Args:
        h_m1: Image height in pixels minus 1.
        w_m1: Image width in pixels minus 1.
        r: Row pixel coordinates (tensor or list).
        c: Column pixel coordinates (tensor or list).
        t: Time elapsed from epoch.
        rotation: Rotation rate in radians/sec clockwise.
        translation: [row, col] translation rate in pixels/sec.

    Returns:
        Tuple of (rr, cc) transformed pixel locations.
    """
    angles = rotation * t
    if isinstance(angles, torch.Tensor):
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
    else:
        sina = torch.sin(ensure_tensor(angles))
        cosa = torch.cos(ensure_tensor(angles))

    rr = sina * c + cosa * r + (h_m1 - (sina * w_m1 + cosa * h_m1)) * 0.5 + translation[0] * t
    cc = cosa * c - sina * r + (w_m1 - (cosa * w_m1 - sina * h_m1)) * 0.5 + translation[1] * t

    return [rr, cc]


def apply_wrap_around(height, width, r, c, t_start, t_end, rotation, translation, wrap_around):
    """Apply wrap-around boundary conditions to transformed coordinates.

    Args:
        height: Height of array.
        width: Width of array.
        r: Row pixel coordinates.
        c: Column pixel coordinates.
        t_start: Start time in seconds from epoch.
        t_end: End time in seconds from epoch.
        rotation: Clockwise rotation rate in radians/sec.
        translation: [row, col] translation rate in pixels/sec.
        wrap_around: [[row_lo, row_hi], [col_lo, col_hi], [center_r, center_c]]
            or None.

    Returns:
        Tuple of (r, c, wrap_around) with updated positions and bounds.
    """
    h = ensure_tensor(height - 1)
    w = ensure_tensor(width - 1)
    r = ensure_tensor(r)
    c = ensure_tensor(c)

    if wrap_around is not None:
        wrap_around = ensure_tensor(wrap_around)
        (r1, c1) = rotate_and_translate(h, w, ensure_tensor(0.0), ensure_tensor(0.0), t_start, rotation, translation)

        r2 = r1 - wrap_around[2][0]
        c2 = c1 - wrap_around[2][1]
        rwrap = wrap_around[0] - r2
        cwrap = wrap_around[1] - c2

        ri0 = torch.where(r < wrap_around[0][0])[0].unsqueeze(1)
        ri1 = torch.where(r > wrap_around[0][1])[0].unsqueeze(1)
        delta_r0 = (rwrap[1] - rwrap[0]).expand(ri0.shape[0])
        delta_r1 = (rwrap[0] - rwrap[1]).expand(ri1.shape[0])
        r = r.scatter_add(0, ri0.squeeze(1), delta_r0)
        r = r.scatter_add(0, ri1.squeeze(1), delta_r1)

        ci0 = torch.where(c < wrap_around[1][0])[0].unsqueeze(1)
        ci1 = torch.where(c > wrap_around[1][1])[0].unsqueeze(1)
        delta_c0 = (cwrap[1] - cwrap[0]).expand(ci0.shape[0])
        delta_c1 = (cwrap[0] - cwrap[1]).expand(ci1.shape[0])
        c = c.scatter_add(0, ci0.squeeze(1), delta_c0)
        c = c.scatter_add(0, ci1.squeeze(1), delta_c1)

        wrap_around = [rwrap, cwrap, [r1.squeeze(), c1.squeeze()]]

    return r, c, wrap_around
