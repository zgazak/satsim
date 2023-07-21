from __future__ import division, print_function, absolute_import
from copy import copy
from os import listdir
from os.path import isfile, join, splitext

import numpy as np
import matplotlib.image as mpimg

from skimage.draw import rectangle_perimeter, line


def get_shape(fpa_np):
    try:
        (h, w) = fpa_np.shape
        c = 1
    except:
        h, w, c = fpa_np.shape

    return h, w, c


def save(
    filename,
    fpa,
    vauto=False,
    vmin=None,
    vmax=None,
    cmap="gray",
    annotation=None,
    pad=5,
    show_obs_boxes=True,
    show_star_boxes=[],
    show_star_lines=[],
):
    """Save an array as an image file.

    Args:
        filename: `string`, the image filename.
        fpa: `np.array`, input image as a 2D numpy array.
        vauto: vmin is set to min value, and vmax is set to 2*median value
        vmin, vmax: `int`, vmin and vmax set the color scaling for the image by
            fixing the values that map to the colormap color limits. If either
            vmin or vmax is None, that limit is determined from the arr min/max
            value.
        cmap: `str`, A Colormap instance or registered colormap name.
        annotation: `dict`, annotation object created from
            `satsim.io.satnet.set_frame_annotation` used to place a box around
            objects in the image
        pad: `int`, pad length in pixels to add to each side of the annotation
            box
    """
    fpa_copy = copy(fpa)
    fpa_flat = fpa_copy.flatten()
    min_val = vmin or np.min(fpa_flat)
    max_val = vmax or np.max(fpa_flat)

    if vauto:
        max_val = (np.median(fpa_flat) - min_val) * 4 + min_val

    fpa_np = fpa_copy
    if annotation is not None and show_obs_boxes is True:
        h, w, c = get_shape(fpa_np)

        for a in annotation:
            start = (a["box"][1] - pad, a["box"][0] - pad)
            end = (
                a["box"][1] + a["box"][3] + pad,
                a["box"][0] + a["box"][2] + pad,
            )
            rr, cc = rectangle_perimeter(start, end=end, shape=fpa_np.shape)
            fpa_np[rr, cc] = max_val

    for box in show_star_boxes:
        pad = 0
        start = (box[1] - pad, box[0] - pad)
        end = (box[1] + box[3] + pad, box[0] + box[2] + pad)
        rr, cc = rectangle_perimeter(start, end=end, shape=fpa_np.shape)
        fpa_np[rr, cc] = max_val

    for lin in show_star_lines:
        rr, cc = line(int(lin[1]), int(lin[0]), int(lin[3]), int(lin[2]))

        try:
            r1, c1 = zip(
                *(
                    (rr, cc)
                    for rr, cc in zip(rr, cc)
                    if (
                        rr > 0
                        and cc > 0
                        and rr < fpa_np.shape[0]
                        and cc < fpa_np.shape[1]
                    )
                )
            )

            fpa_np[r1, c1] = min_val
        except:
            pass

    mpimg.imsave(
        filename,
        fpa_np if c == 1 else np.median(fpa_np, 2),
        vmin=min_val,
        vmax=max_val,
        cmap=cmap,
    )


def savecubeover(
    filename,
    fpa,
    vauto=False,
    vmin=None,
    vmax=None,
    cmap="gray",
    annotation=None,
    pad=5,
    show_obs_boxes=True,
):
    """Save an array as an image file.

    Args:
        filename: `string`, the image filename.
        fpa: `np.array`, input image as a 2D numpy array.
        vauto: vmin is set to min value, and vmax is set to 2*median value
        vmin, vmax: `int`, vmin and vmax set the color scaling for the image by
            fixing the values that map to the colormap color limits. If either
            vmin or vmax is None, that limit is determined from the arr min/max
            value.
        cmap: `str`, A Colormap instance or registered colormap name.
        annotation: `dict`, annotation object created from
            `satsim.io.satnet.set_frame_annotation` used to place a box around
            objects in the image
        pad: `int`, pad length in pixels to add to each side of the annotation
            box
    """

    fpa_flat = fpa.flatten()
    min_val = vmin or np.min(fpa_flat)
    max_val = vmax or np.max(fpa_flat)

    if vauto:
        max_val = (np.median(fpa_flat) - min_val) * 4 + min_val

    fpa_np = fpa

    if annotation is not None and show_obs_boxes is True:
        h, w, c = get_shape(fpa_np)
        for nobj, a in enumerate(annotation):
            if nobj == 0:
                """
                start = (a["y_min"] * h - pad, a["x_min"] * w - pad)
                end = (a["y_max"] * h + pad, a["x_max"] * w + pad)
                rr, cc = rectangle_perimeter(start, end=end, shape=fpa_np.shape)
                fpa_np[rr, cc] = max_val
                """

                slice = fpa_np[
                    int(a["y_min"] * h - 3 * pad) : int(a["y_max"] * h + 3 * pad),
                    int(a["x_min"] * w - 3 * pad) : int(a["x_max"] * w + 3 * pad),
                    :,
                ]
                print(
                    int(a["x_min"] * w - 2 * pad),
                    int(a["x_max"] * w + 2 * pad),
                    int(a["y_min"] * w - 2 * pad),
                    int(a["y_max"] * w + 2 * pad),
                )
                print(h, w, c)
                print(slice.shape)

                num_per_row = round(np.sqrt(15) + 0.5)
                size = slice.shape[0]
                flat_img = (
                    np.ones(
                        (
                            num_per_row * size + num_per_row + 1,
                            num_per_row * size + num_per_row + 1,
                        )
                    )
                    * max_val
                )
                row = 0
                for idx in range(slice.shape[-1]):
                    col = idx % num_per_row

                    flat_img[
                        size * col + col + 1 : size * (col + 1) + col + 1,
                        size * row + row + 1 : size * (row + 1) + row + 1,
                    ] = slice[:size, :size, idx]

                    if col == num_per_row - 1:
                        row += 1

            flat_frame = np.median(fpa_np, -1)

            flat_frame[
                a["pixels"][0][0]
                - round(flat_img.shape[0] / 2) : a["pixels"][0][0]
                - round(flat_img.shape[0] / 2)
                + flat_img.shape[0],
                a["pixels"][0][1]
                - round(flat_img.shape[1] / 2) : a["pixels"][0][1]
                - round(flat_img.shape[1] / 2)
                + flat_img.shape[1],
            ] = flat_img

            mpimg.imsave(
                filename,
                flat_frame,
                vmin=min_val,
                vmax=max_val,
                cmap=cmap,
            )


def savecube(
    filename,
    fpa,
    vauto=False,
    vmin=None,
    vmax=None,
    cmap="gray",
    annotation=None,
    pad=5,
    show_obs_boxes=True,
):
    """Save an array as an image file.

    Args:
        filename: `string`, the image filename.
        fpa: `np.array`, input image as a 2D numpy array.
        vauto: vmin is set to min value, and vmax is set to 2*median value
        vmin, vmax: `int`, vmin and vmax set the color scaling for the image by
            fixing the values that map to the colormap color limits. If either
            vmin or vmax is None, that limit is determined from the arr min/max
            value.
        cmap: `str`, A Colormap instance or registered colormap name.
        annotation: `dict`, annotation object created from
            `satsim.io.satnet.set_frame_annotation` used to place a box around
            objects in the image
        pad: `int`, pad length in pixels to add to each side of the annotation
            box
    """

    fpa_flat = fpa.flatten()
    min_val = vmin or np.min(fpa_flat)
    max_val = vmax or np.max(fpa_flat)

    if vauto:
        max_val = (np.median(fpa_flat) - min_val) * 4 + min_val

    fpa_np = fpa

    if annotation is not None and show_obs_boxes is True:
        h, w, c = get_shape(fpa_np)
        for nobj, a in enumerate(annotation):
            if nobj == 0:
                """
                start = (a["y_min"] * h - pad, a["x_min"] * w - pad)
                end = (a["y_max"] * h + pad, a["x_max"] * w + pad)
                rr, cc = rectangle_perimeter(start, end=end, shape=fpa_np.shape)
                fpa_np[rr, cc] = max_val
                """

                slice = fpa_np[
                    int(a["y_min"] * h - 3 * pad) : int(a["y_max"] * h + 3 * pad),
                    int(a["x_min"] * w - 3 * pad) : int(a["x_max"] * w + 3 * pad),
                    :,
                ]
                print(
                    int(a["x_min"] * w - 2 * pad),
                    int(a["x_max"] * w + 2 * pad),
                    int(a["y_min"] * w - 2 * pad),
                    int(a["y_max"] * w + 2 * pad),
                )
                print(h, w, c)
                print(slice.shape)

                num_per_row = round(np.sqrt(15) + 0.5)
                size = slice.shape[0]
                flat_img = (
                    np.ones(
                        (
                            num_per_row * size + num_per_row + 1,
                            num_per_row * size + num_per_row + 1,
                        )
                    )
                    * max_val
                )
                row = 0
                for idx in range(slice.shape[-1]):
                    col = idx % num_per_row

                    flat_img[
                        size * col + col + 1 : size * (col + 1) + col + 1,
                        size * row + row + 1 : size * (row + 1) + row + 1,
                    ] = slice[0:size, 0:size, idx]

                    if col == num_per_row - 1:
                        row += 1

            mpimg.imsave(
                filename,
                flat_img,
                vmin=min_val,
                vmax=max_val,
                cmap=cmap,
            )


def save_apng(dirname, filename):
    """Combine all jpg and png image files in the specified directory into an
    animated PNG file. Useful to view images in a web browser.

    Args:
        dirname: `string`, directory containing image files to combine.
        filename: `string`, file name of the animated PNG.
    """
    from apng import APNG

    files = [
        join(dirname, f)
        for f in sorted(listdir(dirname))
        if isfile(join(dirname, f))
        and (splitext(f)[1] == ".png" or splitext(f)[1] == ".jpg")
    ]

    APNG.from_files(files, delay=100).save(join(dirname, filename))
