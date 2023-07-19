import pdb
import math
from glob import glob
import os
import json
from tqdm import tqdm
from astropy.io import fits
import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map

from satsim.geometry.wcs import get_min_max_ra_dec

DEFAULT_GAIA_PATH = "/data/share/gaia_fits"


def struct_to_dataframe(index_struct):
    """Convert saved index to pandas dataframes

    Args:
        index_struct (_type_): _description_
    """

    return pd.DataFrame(index_struct["ra_zone_map"]), pd.DataFrame(
        index_struct["dec_zone_map"]
    )


def get_zones_from_values(ras, decs, entry):
    ra_zone = ras[(ras["min"] <= entry["RA"]) & (ras["max"] > entry["RA"])]["idx"].iloc[
        0
    ]

    dec_zone = decs[(decs["min"] <= entry["Dec"]) & (decs["max"] > entry["Dec"])][
        "idx"
    ].iloc[0]

    return ra_zone, dec_zone


def tag_from_zones(ra_zone, dec_zone):
    return f"{ra_zone:d}_{dec_zone:d}"


def map_single_entry_to_zone(tup_args):
    ras, decs, entry, fn, idx = tup_args
    ra_zone, dec_zone = get_zones_from_values(ras, decs, entry)
    zone_tag = tag_from_zones(ra_zone, dec_zone)

    return (zone_tag, fn, idx)


def build_gaia3_index(gaia_path=DEFAULT_GAIA_PATH, ra_zones=60, dec_zones=1800):
    """Read through all gaia3 files and determine the ra/dec limits for each file
    so later we can target which files to actually load for a target simulation.

    Args:
        gaia_path (_type_, optional): _description_. Defaults to DEFAULT_GAIA_PATH.
    """

    zone_dRA = 360 / ra_zones
    zone_dDec = 180 / dec_zones

    ra_lims = np.linspace(0, 360, ra_zones + 1)
    dec_lims = np.linspace(-90, 90, dec_zones + 1)

    index_struct = {
        "settings": {
            "ra_zones": ra_zones,
            "dec_zones": dec_zones,
            "dRA": zone_dRA,
            "dDec": zone_dDec,
        },
        "ra_zone_map": [],
        "dec_zone_map": [],
        "file_zone_map": {},
    }

    for idx in range(len(ra_lims) - 1):
        # min and max RA in zone idx:
        index_struct["ra_zone_map"].append(
            {"idx": idx, "min": ra_lims[idx], "max": ra_lims[idx + 1]}
        )

    for idx in range(len(dec_lims) - 1):
        # min max dec zones
        index_struct["dec_zone_map"].append(
            {"idx": idx, "min": dec_lims[idx], "max": dec_lims[idx + 1]}
        )

    ras, decs = struct_to_dataframe(index_struct)

    for file in tqdm(glob(gaia_path + "/*.fits"), ascii=True, desc="preparing index"):
        # sep fits name
        fn = os.path.split(file)[-1]
        _, tbl = fits.open(file)

        entrylist = []
        for count, entry in enumerate(tbl.data):
            entrylist.append((ras, decs, entry, fn, count))

        entries = process_map(
            map_single_entry_to_zone, entrylist, max_workers=500, chunksize=1000
        )

        for entry in tqdm(entries, ascii=True, desc="unroll"):
            zone_tag, fn, count = entry
            if zone_tag not in index_struct["file_zone_map"]:
                index_struct["file_zone_map"][zone_tag] = {}
                index_struct["file_zone_map"][zone_tag][fn] = []
            elif fn not in index_struct["file_zone_map"][zone_tag]:
                index_struct["file_zone_map"][zone_tag][fn] = []

            # add the row in this table that aligns with the zone
            # satsim can then load this file and quickly extract rows matching the zone_tag
            index_struct["file_zone_map"][zone_tag][fn].append(count)

        # just dump this as we go to monitor
        with open(os.path.join(gaia_path, "index.json"), "w") as fp:
            json.dump(index_struct, fp)


def select_zones(ra_min, ra_max, dec_min, dec_max, zoneInfo):
    """Select a list of regions that intersect with the rectangular coordinate
    bounds

    Args:
        ra_min: `float`, min RA bounds
        ra_max: `float`, max RA bounds
        dec_min: `float`, min dec bounds
        dec_max: `float`, max dec bounds
        zoneIndex: `list`, list of zones

    Returns:
        A `list`, list of dictionaries with zone position and length that
            encompass the ra/dec min max
    """
    ra_grid, dec_grid = struct_to_dataframe(zoneInfo)

    # all ra zones that ra_min and ra_max contain:
    ra_zones = ra_grid[
        (ra_grid["min"] > ra_min - zoneInfo["settings"]["dRA"])
        & (ra_grid["max"] < ra_max + zoneInfo["settings"]["dRA"])
    ]["idx"].values

    # all zones that dec_min and dec_max cover:
    dec_zones = dec_grid[
        (dec_grid["min"] > dec_min - zoneInfo["settings"]["dDec"])
        & (dec_grid["max"] < dec_max + zoneInfo["settings"]["dDec"])
    ]["idx"].values

    zones = []
    for ra in ra_zones:
        for dec in dec_zones:
            zones.append(tag_from_zones(ra, dec))

    return zones


def query_by_los(
    height,
    width,
    y_fov,
    x_fov,
    ra,
    dec,
    rot=0,
    rootPath=DEFAULT_GAIA_PATH,
    pad_mult=0,
    origin="center",
    filter_ob=True,
    flipud=False,
    fliplr=False,
    filter_center=None,
):
    """Query the catalog based on focal plane parameters and ra and dec line
    of sight vector. Line of sight vector is defined as the top left corner
    of the focal plane array.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        ra: `float`, right ascension of top left corner of array, [0,0]
        dec: `float`, declination of top left corner of array, [0,0]
        rot: `float`, focal plane rotation (not implemented)
        rootPath: path to root directory. default: environment variable SATSIM_SSTR7_PATH
        pad_mult: `float`, padding multiplier
        origin: `string`, if `center`, rr and cc will be defined where the line of sight is at the center of
            the focal plane array. default='center'
        filter_ob: `boolean`, remove stars outside pad
        flipud: `boolean`, flip row coordinates
        fliplr: `boolean`, flip column coordinates

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row pixel locations
            cc: `list`, list of column pixel locations
            mv: `list`, list of visual magnitudes
    """

    cmin, cmax, w = get_min_max_ra_dec(
        height, width, y_fov / height, x_fov / width, ra, dec, rot, pad_mult, origin
    )

    # no need our gaia in deg
    # cmin = np.radians(cmin)
    # cmax = np.radians(cmax)
    stars = query_by_min_max(
        cmin[0], cmax[0], cmin[1], cmax[1], rootPath, filter_center=filter_center
    )

    rra = np.array([s["ra"] for s in stars])
    ddec = np.array([s["dec"] for s in stars])
    mm = np.array([s["m_g"] for s in stars])
    spt = [s["spt"] for s in stars]

    cc, rr = w.wcs_world2pix(np.degrees(rra), np.degrees(ddec), 0)

    if filter_ob:
        hp = height * (1 + pad_mult)
        wp = width * (1 + pad_mult)
        in_bounds = np.logical_and.reduce([rr <= hp, rr >= -hp, cc <= wp, cc >= -wp])
        rr = rr[in_bounds]
        cc = cc[in_bounds]
        mm = mm[in_bounds]
        spt = [sp for sp, in_bound in zip(spt, list(in_bounds)) if in_bound]

    if origin == "center":
        rr += height / 2.0
        cc += width / 2.0

    if flipud:
        rr = height - rr

    if fliplr:
        cc = width - cc

    return rr, cc, mm, spt


def query_by_min_max(
    ra_min,
    ra_max,
    dec_min,
    dec_max,
    rootPath=DEFAULT_GAIA_PATH,
    clip_min_max=True,
    filter_center=None,
):
    """Query the catalog based on focal plane parameters and minimum and
    maximum right ascension and declination.

    Args:
        ra_min: `float`, min RA bounds [deg]
        ra_max: `float`, max RA bounds [deg]
        dec_min: `float`, min dec bounds [deg]
        dec_max: `float`, max dec bounds [deg]
        rootPath: `string`, path to root directory. default: environment
            variable SATSIM_SSTR7_PATH
        clip_min_max: `boolean`, clip stars outsize of `ra_min` and `ra_max`

    Returns:
        A `list`, stars within the bounds of input parameters
    """
    mas_yr_to_rad_sec = 3.154e7 / 4.8481368e-9  # sec/year / rad/mas

    zoneInfo = json.load(open(os.path.join(rootPath, "index.json"), mode="r"))

    zones_to_load = select_zones(ra_min, ra_max, dec_min, dec_max, zoneInfo)
    files_to_load = {}
    for zone in zones_to_load:
        for file in zoneInfo["file_zone_map"][zone].keys():
            if file not in files_to_load:
                files_to_load[file] = zoneInfo["file_zone_map"][zone][file]
            else:
                files_to_load[file] = (
                    files_to_load[file] + zoneInfo["file_zone_map"][zone][file]
                )

    stars = []

    for file in files_to_load:
        _, tbl = fits.open(os.path.join(rootPath, file))

        # narrow by precomputed index locations of "in zone" stars
        for star in tbl.data[files_to_load[file]]:
            # it isn't clear, but it looks like satsim wants ra,dec in radians, pms in rad/sec
            # gaia PMs are mas/yr
            # epoch is 2016
            stars.append(
                {
                    "ra": np.deg2rad(star["RA"]),
                    "dec": np.deg2rad(star["Dec"]),
                    "ra_pm": star["RA_PM"] * mas_yr_to_rad_sec,
                    "dec_pm": star["DEC_PM"] * mas_yr_to_rad_sec,
                    "parallax": 0,  # forgot to have billy add this in, not sure if it's used
                    "m_g": star["G_MAG"],
                    "m_gbp": star["GBP_MAG"],
                    "m_grp": star["GRP_MAG"],
                    "spt": None if star["SPTYPE"] == "null" else star["SPTYPE"],
                }
            )

    return stars


if __name__ == "__main__":
    # to build initial index file, run:
    # python -m satsim.geometry.gaia3
    # which does this:
    if not os.path.exists(os.path.join(DEFAULT_GAIA_PATH, "index.json")):
        build_gaia3_index()

    rr, cc, mm, spt = query_by_los(512, 512, 0.5, 0.5, 42.0, 11.0, 32)
