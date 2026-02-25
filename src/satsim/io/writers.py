"""Output writers for SatSim.

Consumes FrameResult objects. Completely optional - rendering works without it.
Extracted from gen_images() in satsim.py.
"""

from __future__ import annotations

import os
import copy
import logging
from datetime import datetime

import numpy as np

from satsim.pipeline.engine import generate_frames
from satsim.pipeline.frame import FrameResult
from satsim.io.satnet import write_frame, set_frame_annotation, init_annotation
from satsim.config.loading import save_json

logger = logging.getLogger(__name__)


def write_collection(config: dict, output_dir: str = './', seed: int | None = None) -> str:
    """Generate and write a full collection of frames.

    Args:
        config: Resolved config dict.
        output_dir: Root output directory.
        seed: Optional seed for reproducibility.

    Returns:
        Directory path where output was written.
    """
    from satsim.config.loading import realize

    ssp = realize(config, seed=seed)

    h = ssp['fpa']['height']
    w = ssp['fpa']['width']
    s_osf = ssp['sim']['spacial_osf']
    y_ifov = ssp['fpa']['y_fov'] / h
    x_ifov = ssp['fpa']['x_fov'] / w
    num_frames = ssp['fpa']['num_frames']
    a2d_dtype = ssp['fpa']['a2d'].get('dtype', 'uint16')
    t_exposure = ssp['fpa']['time']['exposure']

    dt = datetime.now()
    dir_name = os.path.join(output_dir, dt.isoformat().replace(':', '-'))
    set_name = 'sat_00000'

    os.makedirs(dir_name, exist_ok=True)

    meta_data = init_annotation(
        'dir.name',
        ['{}.{:04d}.json'.format(set_name, x) for x in range(num_frames)],
        h, w, y_ifov, x_ifov,
    )

    for frame in generate_frames(ssp):
        frame_np = frame.to_numpy()
        if frame_np.fpa_digital is not None:
            write_single_frame(
                frame_np, dir_name, set_name, meta_data,
                t_exposure, dt, ssp, a2d_dtype,
            )

    return dir_name


def write_single_frame(frame: FrameResult, dir_name: str, set_name: str,
                        meta_data: dict, t_exposure: float,
                        time_stamp: datetime, ssp: dict,
                        dtype: str = 'uint16') -> None:
    """Write a single frame to disk.

    Args:
        frame: FrameResult (should be numpy).
        dir_name: Output directory.
        set_name: Dataset name.
        meta_data: Annotation metadata dict.
        t_exposure: Exposure time.
        time_stamp: Timestamp.
        ssp: Config dict.
        dtype: Output dtype string.
    """
    write_frame(
        dir_name=dir_name,
        sat_name=set_name,
        fpa_digital=frame.fpa_digital,
        meta_data=copy.deepcopy(meta_data),
        frame_num=frame.frame_num,
        exposure_time=t_exposure,
        time_stamp=time_stamp,
        ssp=ssp,
        show_obs_boxes=ssp['sim'].get('show_obs_boxes', True),
        show_star_boxes=ssp['sim'].get('show_star_boxes', False),
        astrometrics=frame.astrometrics,
        save_pickle=ssp['sim'].get('save_pickle', False),
        dtype=dtype,
        fits_compression=ssp['sim'].get('fits_compression', None),
        save_jpeg=ssp['sim'].get('save_jpeg', True),
        ground_truth=frame.ground_truth,
        segmentation=frame.segmentation,
    )
