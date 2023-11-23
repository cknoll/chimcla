import os
import sys
import numpy as np
import glob
import itertools as it
import collections
import argparse
from colorama import Fore, Style
from matplotlib import pyplot as plt

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
    help="create directories with file names containing feature values; e.g. cell_img_dir = critical_hist_chunk002",
    metavar="cell_img_dir"
)


parser.add_argument(
    "--create-stats",
    "-cs",
    help="create statistics on how the different features behave",
    metavar="cell_img_dir"
)

parser.add_argument(
    "--create-json-data",
    "-cj",
    help="create a json file which contains information about the critical cells",
    metavar="cell_img_dir"
)


args = parser.parse_args()


def get_dirname(keys):

    prefixes = [PREFIXES[key] for key in keys]
    dirname = "__".join(prefixes)
    return dirname


# the keys should be kept in sync with the library (see comment `# anchor::db_keys`)

PREFIXES = {
    "crit_pix_nbr": "nbr",
    "crit_pix_mean": "avg",
    "crit_pix_median": "med",
    "crit_pix_q95": "qnt",
    "score_str": "A",
}


def create_feature_dirs():
    basedir = args.create_feature_dirs
    assert os.path.isdir(basedir)
    fpaths = list(sorted(glob.glob(f"{basedir}/*.jpg")))

    assert fpaths, "unexpected empty list"

    orders = [
        ("score_str", "crit_pix_nbr", "crit_pix_mean"),
        ("crit_pix_nbr", "crit_pix_mean", "score_str"),
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
                    value = f"{value:05.1f}"
                if isinstance(value, int):
                    value = f"{value:04d}"
                if value is None:
                    value = "0X"

                assert isinstance(value, str)
                prefix = PREFIXES[key]
                new_fname_parts.append(f"{prefix}{value}")

            new_fname = "_".join(new_fname_parts) + f"__{fname}"

            new_path = os.path.join(os.path.join(basedir, get_dirname(order)), new_fname)

            try:
                os.symlink(os.path.join("..", fname), new_path)
            except FileExistsError:
                pass

            # print("created:",  new_path)


def create_stats():
    basedir = args.create_stats
    assert os.path.isdir(basedir)
    fpaths = list(sorted(glob.glob(f"{basedir}/*.jpg")))

    dirname = "_stats"
    os.makedirs(os.path.join(basedir, dirname), exist_ok=True)

    keys = list(PREFIXES.keys())

    stat_combinations = [
        ("crit_pix_nbr", "crit_pix_mean"),
        ("crit_pix_nbr", "crit_pix_median"),
        ("crit_pix_nbr", "crit_pix_q95"),
        ("crit_pix_mean", "crit_pix_median"),
        ("crit_pix_mean", "crit_pix_q95"),
        ("crit_pix_mean", "score_str"),
        ("crit_pix_nbr", "score_str"),
    ]

    num_data = collections.defaultdict(list)
    for fpath in fpaths:
        fname = os.path.split(fpath)[-1]
        data = bs.db.get(fname)

        if data is None:
            print(f"could not find key {fname} in db")
            continue

        if data["crit_pix_nbr"] < 5:
            # image is not interesting for stats
            continue

        for key in keys:
            value = data[key]

            if isinstance(value, str):
                value = int(value)

            num_data[key].append(value)

    for x_key, y_key in stat_combinations:
        xx = num_data[x_key]
        yy = num_data[y_key]

        xlabel = PREFIXES[x_key]
        ylabel = PREFIXES[y_key]

        plt.figure()
        plt.plot(xx, yy, ".")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{xlabel} vs {ylabel}")

        fpath = os.path.join(basedir, dirname, f"{xlabel}_{ylabel}.png")

        plt.savefig(fpath)


def create_json_data(basedir: str):

    res = {}

    assert os.path.isdir(basedir)
    fpaths = list(sorted(glob.glob(f"{basedir}/*.jpg")))

    for fpath in fpaths:
        fname = os.path.split(fpath)[-1]
        data = bs.db.get(fname)

        if data is None:
            print(f"could not find key {fname} in db")
            continue

        res[fname] = {"citicality_score": int(data["score_str"])}

    import json

    jfpath = os.path.join(basedir, "hist_eval.json")
    with open(jfpath, "w") as fp:
        json.dump(res, fp, indent=2)

    print(f"File written: {jfpath}")




def main():

    if args.create_feature_dirs:
        create_feature_dirs()

    elif args.create_stats:
        create_stats()
    elif args.create_json_data:
        create_json_data(basedir=args.create_json_data)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()