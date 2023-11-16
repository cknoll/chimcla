import os
import sys
import numpy as np
import glob
import itertools as it
import collections
import argparse
from colorama import Fore, Style

import dill

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()

from stage_2a_bar_selection import (
    Container,
)

import stage_2a_bar_selection as bs
import asyncio_tools as aiot


exclude_cell_keys = [("a", "1"), ("b", "1"), ("c", "1")]


ERROR_CMDS = []


parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description='This program creates copies or symlinks of files to enable simple sorting wrt. one numeric feature',
)


parser.add_argument(
    "--create-feature-dirs",
    "-cfd",
    help="create direectories with file names containing feature values; e.g. cell_img_dir = critical_hist_chunk002",
    metavar="cell_img_dir"
)


args = parser.parse_args()


def get_dirname(order):
    dirname = "__".join(order)
    return dirname


def create_feature_dirs():
    basedir = args.create_feature_dirs
    assert os.path.isdir(basedir)
    fpaths = list(sorted(glob.glob(f"{basedir}/*.jpg")))

    assert fpaths, "unexpected empty list"

    # the key should be kept in sync with the library (see comment `# anchor::db_keys`)

    prefixes = {
        "crit_pix_nbr": "nbr",
        "crit_pix_mean": "avg",
        "crit_pix_median": "med",
        "crit_pix_q95": "qnt",
        "score_str": "A",
    }

    orders = [
        ("score_str", "crit_pix_nbr", "crit_pix_mean"),
        ("crit_pix_nbr", "crit_pix_mean"),
        ("crit_pix_mean", "crit_pix_nbr"),
        ("crit_pix_median", "crit_pix_nbr"),
        ("crit_pix_q95", "crit_pix_nbr"),
    ]

    # create all dirs
    for order in orders:
        os.makedirs(os.path.join(basedir, get_dirname(order)), exist_ok=True)

    for fpath in fpaths:
        fname = os.path.split(fpath)[-1]
        data = bs.db.get(fname)

        if data is None:
            print(f"could not find key {fname} in db")
            continue

        for order in orders:
            new_fname_parts = []
            for key in order:
                value = data.get(key, "XXX")
                if isinstance(value, np.ndarray):
                    arr = np.atleast_1d(value)
                    assert len(arr) == 1
                    value = arr[0]
                if isinstance(value, float):
                    value = f"{value:.1f}"
                if value is None:
                    value = "0X"

                assert isinstance(value, (str, int))
                prefix = prefixes[key]
                new_fname_parts.append(f"{prefix}{value}")

            new_fname = "_".join(new_fname_parts) + f"__{fname}"

            new_path = os.path.join(os.path.join(basedir, get_dirname(order)), new_fname)

            try:
                os.symlink(os.path.join("..", fname), new_path)
            except FileExistsError:
                pass

            # print("created:",  new_path)

def main():

    if args.create_feature_dirs:
        create_feature_dirs()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()