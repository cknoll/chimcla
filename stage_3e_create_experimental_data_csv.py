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


def main():

    dn = args.dirname
    paths = glob.glob(f"{dn}/*jpg")
    base_names0 = [p.replace(f"{dn}/", "") for p in paths if "exp_" not in p]
    base_names = ["_".join(bn.split("_")[1:]) for bn in base_names0]
    import pandas as pd


    data = {"fname": [], "crit_cells": [], "crit_pix": [], "max_q95": [], "pix_above_q95": []}

    df = pd.DataFrame(data)

    for bn in base_names:
        try:
            s = summary = Container(**bs.db["criticality_summary"][bn[:-4]])
        except KeyError as ex:
            print(str(ex))
            continue
        df.loc[len(df.index)] = [bn, s.crit_cell_number, s.crit_pix_number, np.max(s.q95_list), s.crit_pix_above_q95_num]


    csv_fname = os.path.join(dn, "_data.csv")
    df.to_csv(csv_fname)
    print(f"file written {csv_fname}")

if __name__ == "__main__":
    main()