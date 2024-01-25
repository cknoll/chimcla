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
    description='This program a csv file for a directory of images',
)


parser.add_argument(
    "dirname",
    help="directory to which to apply",
)



args = parser.parse_args()


def get_all_files_for_basename(dirname: str, basename: str):

    # example:
    # bn = '2023-06-26_09-05-30_C50.jpg'
    # res = [
    # 'experimental_imgs_psy01_bm0_bv60/P178_2023-06-26_09-05-30_C50.jpg',
    # 'experimental_imgs_psy01_bm0_bv60/P178_2023-06-26_09-05-30_C50_exp_hard_110.jpg',
    # 'experimental_imgs_psy01_bm0_bv60/P178_2023-06-26_09-05-30_C50_exp_soft_60.jpg'
    # ]

    res = glob.glob(f"{dirname}/*{os.path.splitext(basename)[0]}*")

    # drop the dirname
    res = [os.path.split(elt)[1] for elt in res]

    assert len(res) == 3
    return res


def main():

    dn = args.dirname
    paths = glob.glob(f"{dn}/*jpg")
    base_names0 = [p.replace(f"{dn}/", "") for p in paths if "exp_" not in p]
    base_names = ["_".join(bn.split("_")[1:]) for bn in base_names0]
    import pandas as pd

    data = {
        "fname_raw": [],
        "fname_orig": [],
        "fname_hard": [],
        "fname_soft": [],
        "crit_cells": [], "crit_pix": [], "max_q95": [], "pix_above_q95": [],
        "crit_pix_mean": [],
        "crit_pix_std": [],
        "crit_pix_median": [],
        "crit_pix_q95": [],
        "crit_pix_q05": [],
    }

    df = pd.DataFrame(data)

    for bn in base_names:
        fnames = get_all_files_for_basename(dirname=dn, basename=bn)
        try:
            s = summary = Container(**bs.db["criticality_summary"][bn[:-4]])
        except KeyError as ex:
            print(f"KeyError for {str(ex)}")
            continue
        df.loc[len(df.index)] = [
            bn, *fnames, s.crit_cell_number, s.crit_pix_number, np.max(s.q95_list), s.crit_pix_above_q95_num,
            s.crit_pix_mean, s.crit_pix_std, s.crit_pix_median, s.crit_pix_q95, s.crit_pix_q05,
        ]


    csv_fname = os.path.join(dn, "_data.csv")

    # tab-separated csv
    df.to_csv(csv_fname, sep='\t')
    print(f"file written {csv_fname}")

if __name__ == "__main__":
    main()