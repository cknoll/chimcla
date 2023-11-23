"""
This file serves for creating images for Romys Chocolate Experiment (as discussed in Nov. 2023)

"""

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

import stage_2a_bar_selection as bs
import asyncio_tools as aiot


exclude_cell_keys = [("a", "1"), ("b", "1"), ("c", "1")]


ERROR_CMDS = []


parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description='This program creates histogram style plots for every unusual cell (also considering the area)',
)

parser.add_argument(
    '--img_dir',
    help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0",
    default=None,
)

parser.add_argument(
    '--img',
    help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0/2023-06-26_06-17-41_C50.jpg",
    default=None,
)

parser.add_argument(
    "--suffix",
    "-s",
    help="specify suffix for output folder",
    default="_chunk_psy_test",
)


parser.add_argument(
    "--no-parallel",
    "-np",
    help="sequential mode (no parallelization)",
    action="store_true",
)

parser.add_argument(
    "--limit",
    help="limit the number of files",
    default=None,
    type=int,
)

args = parser.parse_args()


def process_img(img_fpath):
    original_img_fpath = bs.get_original_image_fpath(img_fpath)
    IPS()


from collections import defaultdict
import re


def update_cell_mappings():

    pat = re.compile(".*(_.*?)\.jpg")
    cell_mappings = defaultdict(list)

    for key in bs.db.keys():

        if not key.endswith(".jpg"):
            continue

        cell_ = pat.match(key).group(1)
        fname = key.replace(cell_, "")
        cell_mappings[fname].append(key)

    bs.db["cell_mappings"] = cell_mappings
    bs.db.commit()


@aiot.background
def run_this_script(img_path, **kwargs):

    options = kwargs.get("options", {})
    option_str_elements = []
    for option, value in options.items():
        if value is True:
            option_str_elements.append(option)

    option_str = " ".join(option_str_elements)

    cmd = f"{sys.executable} {__file__} --img {img_path} --suffix {args.suffix} {option_str}".strip()
    print(cmd)
    res = os.system(cmd)
    res = 0

    if res != 0:
        ERROR_CMDS.append(cmd)


def aio_main():

    update_cell_mappings()

    arg_list = bs.get_img_list(args.img_dir)[:args.limit]
    func = run_this_script

    # prepare options for passing to individual calls
    options = {}

    if args.no_parallel:
        for arg in arg_list:
            func(arg, options=options)
    else:
        aiot.run(aiot.main(func=func, arg_list=arg_list, options=options))

    if ERROR_CMDS:
        print(
            f"\n{Fore.RED}There where errors with the following commands:\n\n",
            Style.RESET_ALL,
            "\n".join(ERROR_CMDS),
        )

def main():

    if args.img:
        process_img(args.img)
    elif args.img_dir:
        aio_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()