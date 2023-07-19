import pdb
import math
from glob import glob
import os
from astropy.io import fits

import numpy as np

# from satsim.geometry.wcs import get_min_max_ra_dec

DEFAULT_GAIA_PATH = "/data/share/gaia_fits"


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

    index_struct = {"ra_zone_map": {}, "dec_zone_map": {}, "file_zone_map": {}}

    for idx in len(ra_lims) - 1:
        # min and max RA in zone idx:
        index_struct["ra_zone_map"][idx] = [ra_lims[idx], ra_lims[idx + 1]]

    for idx in len(dec_lims) - 1:
        # min and max RA in zone idx:
        index_struct["dec_zone_map"][idx] = [dec_lims[idx], dec_lims[idx + 1]]

    for file in glob(gaia_path + "/*.fits"):
        _, tbl = fits.open(file)
        ra_range = [np.min(tbl.data["RA"]), np.max(tbl.data["RA"])]
        dec_range = [np.min(tbl.data["dec"]), np.max(tbl.data["dec"])]

        print(ra_range)
        print(dec_range)

        pdb.set_trace()


if __name__ == "__main__":
    build_gaia3_index()
