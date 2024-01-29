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
    description='This program creates a csv file for a directory of images and other useful data',
)


parser.add_argument(
    "dirname",
    help="directory to which to apply",
    nargs="?",
)


parser.add_argument(
    "--csv",
    help="create general csv data",
    action="store_true",
)


parser.add_argument(
    "-dia",
    "--diagram",
    help="create overview diagram",
    action="store_true",
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



def get_base_names_in_dir(dirpath):
    paths = glob.glob(f"{dirpath}/*jpg")
    base_names0 = [p.replace(f"{dirpath}/", "") for p in paths if "exp_" not in p]
    base_names = ["_".join(bn.split("_")[1:]) for bn in base_names0]
    return base_names

def generate_csv():

    dn = args.dirname
    base_names = get_base_names_in_dir(dn)
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

def generate_diagram():

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = ['tab:green', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    alphas = [1, .5, .5] + [0.5]*10

    subdir_name_list = os.listdir(args.dirname)
    subdir_names = {}

    base_names_color_tuples = []
    for i in range(10):

        # determine subdir name
        try:
            sdn = [n for n in subdir_name_list if n.startswith(f"{i}_")][0]
        except IndexError:
            # no subdir starting with {i}_ was found -> break
            break
        subdir_names[i] = sdn

        dn = os.path.join(args.dirname, sdn)
        base_names = get_base_names_in_dir(dn)
        if not base_names:
            break
        base_names_color_tuples.append((base_names, colors.pop(0)))


    plt.figure(figsize=(10, 10))

    for i, (base_names, color) in enumerate(base_names_color_tuples):
        crit_pix_numbers = []
        brightness = []
        sdn = subdir_names[i]
        dn = os.path.join(args.dirname, sdn)
        # create a new dir for the copies
        new_dir = os.path.join(args.dirname, f"new_{sdn}")
        os.makedirs(new_dir, exist_ok=True)

        print(sdn, end=": ")

        for bn in base_names:
            fnames = get_all_files_for_basename(dirname=dn, basename=bn)
            try:
                s = summary = Container(**bs.db["criticality_summary"][bn[:-4]])
            except KeyError as ex:
                print(f"KeyError for {str(ex)}")
                continue

            crit_pix_numbers.append(s.crit_pix_number)
            brightness.append(s.crit_pix_mean)

            # create renamed copies
            for fn in fnames:
                path_src = os.path.join(dn, fn)


                # create new fname "P103_2023-06-27_02-41-01_C0.jpg" -> "P103_x123-y456_2023..."
                parts = fn.split("_")
                parts.insert(1, f"x{s.crit_pix_number}-y{round(s.crit_pix_mean)}")
                fn_new = "_".join(parts)

                path_dst = os.path.join(new_dir, fn_new)
                cmd = f"cp {path_src} {path_dst}"
                os.system(cmd)
                print(".", end="")

        # print a newline after each sdn-loop
        print()

        plt.plot(crit_pix_numbers, brightness, "o", color=color, label=sdn, alpha=alphas.pop(0))
    plt.xlabel("number of critical pixels")
    plt.ylabel("mean brightness of critical pixels")
    plt.legend()
    plt.tight_layout()
    figname = os.path.join(args.dirname, "overview.png")
    plt.savefig(figname)
    print(figname, "written")
    # plt.show()


    # IPS()

def main():

    if args.csv:
        generate_csv()
    elif args.diagram:
        generate_diagram()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()