from __future__ import division, print_function, absolute_import

import os
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np

from satsim.io import fits, image
from satsim.config import save_json


def init_annotation(dirname, sequence, height, width, y_ifov, x_ifov):
    """Init annotation object for SatNet.

    Args:
        dirname: `string`, the name of the directory (not full path) the SatNet
            data files will be saved to.
        sequence: `int`, sequence
        height: `int`, image height in pixels
        width: `int`, image width in pixels
        y_ifov: `float`, pixel fov in degrees
        x_ifov: `float`, pixel fov in degrees

    Returns:
        A `dict`, the data object for `set_frame_annotation`
    """
    data = OrderedDict()

    data["data"] = {
        "file": {"dirname": dirname, "sequence": sequence},
        "sensor": {"height": height, "width": width, "iFOVy": y_ifov, "iFOVx": x_ifov},
        "objects": [],
    }

    return data


def set_frame_annotation(
    data,
    frame_num,
    height,
    width,
    obs,
    box_size=None,
    box_pad=0,
    filter_ob=False,
    snr=None,
    obs_annot=None,
    star_annot=None,
    star_lines=None,
):
    """Set frame data on annotation object for SatNet.

    Args:
        data: `dict`, object returned by `init_annotation`.
        frame_num: `int`, the current frame number
        height: `int`, image height in pixels
        width: `int`, image width in pixels
        obs: `list`, list of SatSim obs
        box_size: `[int, int]`, box size in row,col pixels
        box_pad: `int`, amount of pad to add to each side of box
        filter_ob: `boolean`, bounds min and max and remove out of bounds

    Returns:
        A `dict`, the data object for `set_frame_annotation`
    """
    os_factor = width / data["data"]["sensor"]["width"]
    data["data"]["file"]["filename"] = "undefined"
    data["data"]["file"]["sequence_id"] = frame_num
    data["data"]["stats"] = {"num_obs_initial": len(obs), "num_obs": len(obs)}
    data["data"]["star_annotation_channel"] = 1
    data["data"]["noise_annotation_channel"] = 0

    objs = data["data"]["objects"] = []
    data["data"]["star_boxes"] = star_annot if star_annot is not None else []
    data["data"]["star_lines"] = star_lines if star_lines is not None else []

    def is_ob(a, b):
        if a > 1 and b > 1 or a < 0 and b < 0:
            return True
        else:
            return False

    if snr is not None:
        snra = snr.numpy()

    for idx, o in enumerate(obs):
        # add 0.5 since the corner of array 0,0, thus middle of first pixel is 0.5,0.5
        rr_norm = (o["rr"] + 0.5) / height
        cc_norm = (o["cc"] + 0.5) / width

        y_min_true = np.min(rr_norm)
        x_min_true = np.min(cc_norm)
        y_max_true = np.max(rr_norm)
        x_max_true = np.max(cc_norm)
        y_center_true = (y_max_true + y_min_true) / 2  # TODO assumes linear
        x_center_true = (x_max_true + x_min_true) / 2

        if filter_ob:
            if is_ob(y_min_true, y_max_true) or is_ob(x_min_true, x_max_true):
                continue
            rr = rr_norm[np.logical_and(rr_norm >= 0, rr_norm < 1)]
            cc = cc_norm[np.logical_and(cc_norm >= 0, cc_norm < 1)]
        else:
            rr = rr_norm
            cc = cc_norm

        # calculate inbound coordinates
        y_min = np.min(rr)
        x_min = np.min(cc)
        y_max = np.max(rr)
        x_max = np.max(cc)
        y_center = (y_max + y_min) / 2
        x_center = (x_max + x_min) / 2

        if box_size is None:
            bbox_height = y_max - y_min + box_pad * 2 / height * os_factor
            bbox_width = x_max - x_min + box_pad * 2 / width * os_factor
        else:
            bbox_height = (box_size[0] + box_pad * 2) / height * os_factor
            bbox_width = (box_size[1] + box_pad * 2) / width * os_factor

        osnr = []
        opix = []
        if snr is not None:
            rrr = o["rrr"].astype(int)
            rcc = o["rcc"].astype(int)
            upix, uidx = np.unique(
                np.column_stack((rrr, rcc)), axis=0, return_index=True
            )
            rrr = rrr[uidx]
            rcc = rcc[uidx]
            uidx = np.logical_and(
                np.logical_and(rrr >= 0, rrr < snra.shape[0]),
                np.logical_and(rcc >= 0, rcc < snra.shape[1]),
            )
            osnr = snra[rrr[uidx], rcc[uidx]].tolist()
            osnr = list(map(lambda x: float("%.2f" % (x)), osnr))
            opix = upix.tolist()

        objs.append(
            OrderedDict(
                {
                    "class_type": "Satellite",
                    "class_name": obs_annot[idx]["annotation"]["class_name"]
                    if obs_annot is not None
                    else 0,
                    "class_id": int(obs_annot[idx]["annotation"]["class_id"])
                    if obs_annot is not None
                    else 0,
                    "obstime": obs_annot[idx]["annotation"]["obstime"]
                    if obs_annot is not None
                    else 0,
                    "range": float(obs_annot[idx]["annotation"]["range"])
                    if obs_annot is not None
                    else 0,
                    "source_spectrum": obs_annot[idx]["annotation"]["spectrum"]
                    if obs_annot is not None
                    else 0,
                    "phase_angle": float(obs_annot[idx]["annotation"]["phase_angle"])
                    if obs_annot is not None
                    else 0,
                    "projected_meters_sq": float(
                        obs_annot[idx]["annotation"]["projected_meters_sq"]
                    )
                    if obs_annot is not None
                    else 0,
                    "annotation_channel": idx + 2,  # 0 is noise, 1 is stars
                    "y_min": y_min,
                    "x_min": x_min,
                    "y_max": y_max,
                    "x_max": x_max,
                    "y_center": y_center,
                    "x_center": x_center,
                    "bbox_height": bbox_height,
                    "bbox_width": bbox_width,
                    "box": [
                        int((x_min - bbox_width / 2) * data["data"]["sensor"]["width"]),
                        int(
                            (y_min - bbox_height / 2) * data["data"]["sensor"]["height"]
                        ),
                        int(bbox_width * data["data"]["sensor"]["width"]),
                        int(bbox_height * data["data"]["sensor"]["height"]),
                    ],
                    "source": "satsim",
                    "magnitude": o["mv"],
                    "pe_per_sec": o["pe"],
                    "y_start": rr_norm[0],
                    "x_start": cc_norm[0],
                    "y_mid": y_center_true,
                    "x_mid": x_center_true,
                    "y_end": rr_norm[-1],
                    "x_end": cc_norm[-1],
                    "pixels": opix,
                    "snr": osnr,
                }
            )
        )

    return data


def write_frame(
    dir_name,
    sat_name,
    fpa_digital,
    meta_data,
    frame_num,
    exposure_time,
    time_stamp,
    ssp,
    show_obs_boxes=True,
    show_star_boxes=True,
    astrometrics=None,
    save_pickle=False,
    dtype="uint16",
    save_jpeg=True,
    seg_map=None,
):
    """Write image and annotation files compatible with SatNet. In addition,
    writes annotated images and SatSim configuration file for reference.

    Args:
        dir_name: `string`, directory to save files to
        sat_name: `string`, satellite name
        fpa_digital: `array`, image as a numpy array
        meta_data: `dict`, annotation data generated by `init_annotation`
        frame_num: `int`, current frame number in set
        exposure_time: `float`, current frame exposure time in seconds
        time_stamp: `datetime`, reference time
        ssp: `dict`: SatSim parameters to be saved to JSON file
        dtype: `string`: Data type to save FITS pixel data as
    """
    print("writing frame")
    file_name = "{}.{:04d}".format(sat_name, frame_num)

    meta_data["data"]["file"]["dirname"] = Path(dir_name).name
    meta_data["data"]["file"]["filename"] = "{}.fits".format(file_name)

    annotation_dir = os.path.join(dir_name, "Annotations")
    image_dir = os.path.join(dir_name, "ImageFiles")
    annotatedimg_dir = os.path.join(dir_name, "AnnotatedImages")
    annotatedimgstar_dir = os.path.join(dir_name, "AnnotatedImagesStars")
    annotatedimgcube_dir = os.path.join(dir_name, "AnnotatedImagesCube")
    annotatedimgcubeo_dir = os.path.join(dir_name, "AnnotatedImagesCubeO")

    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir, exist_ok=True)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
    if not os.path.exists(annotatedimg_dir):
        os.makedirs(annotatedimg_dir, exist_ok=True)
    if not os.path.exists(annotatedimgstar_dir):
        os.makedirs(annotatedimgstar_dir, exist_ok=True)
    if not os.path.exists(annotatedimgcube_dir):
        os.makedirs(annotatedimgcube_dir, exist_ok=True)
    if not os.path.exists(annotatedimgcubeo_dir):
        os.makedirs(annotatedimgcubeo_dir, exist_ok=True)

    # save fits
    fits.save(
        os.path.join(image_dir, "{}.fits".format(file_name)),
        fpa_digital,
        exposure_time,
        time_stamp,
        overwrite=True,
        astrometrics=astrometrics,
        dtype=dtype,
        # seg_map=seg_map,
    )

    # save annotation
    with open(
        os.path.join(annotation_dir, "{}.json".format(file_name)), "w"
    ) as json_file:
        json.dump(meta_data, json_file, indent=None, separators=(",", ":"))

    # save annotated images
    if save_jpeg:
        """
        try:
            image.savecube(
                os.path.join(annotatedimgcube_dir, "{}.jpg".format(file_name)),
                fpa_digital,
                vauto=True,
                annotation=meta_data["data"]["objects"],
                show_obs_boxes=show_obs_boxes,
            )
        except Exception:
            pass
        """
        try:
            image.save(
                os.path.join(annotatedimg_dir, "{}.jpg".format(file_name)),
                fpa_digital,
                vauto=True,
                pad=5,
                annotation=meta_data["data"]["objects"],
                show_obs_boxes=show_obs_boxes,
            )
        except Exception:
            pass
        """
        try:
            image.save(
                os.path.join(annotatedimgstar_dir, "{}.jpg".format(file_name)),
                fpa_digital,
                vauto=True,
                pad=0,
                annotation=meta_data["data"]["objects"],
                show_obs_boxes=show_obs_boxes,
                show_star_boxes=meta_data["data"]["star_boxes"],
                show_star_lines=meta_data["data"]["star_lines"],
            )
        except Exception as e:
            raise e
            pass

        image.savecubeover(
            os.path.join(annotatedimgcubeo_dir, "{}.jpg".format(file_name)),
            fpa_digital,
            vauto=True,
            annotation=meta_data["data"]["objects"],
            show_obs_boxes=show_obs_boxes,
        )"""

    # save sim config
    save_json(os.path.join(dir_name, "config.json"), ssp, save_pickle=save_pickle)
