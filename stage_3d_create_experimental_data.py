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

from stage2 import stage_2a_bar_selection as bs

from stage2 import asyncio_tools as aiot


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

parser.add_argument(
    "--blend-value",
    "-bv",
    default=120,
    type=int,
)

parser.add_argument(
    "--blend-mode",
    "-bm",
    help= "0 (soft, default) or 1 (hard)",
    default=0,
    type=int,
)

parser.add_argument(
    "--print-std-deviation",
    "-std",
    help= "print standard deviation of critical pixels for each critical cell onto the image",
    action="store_true",
)

args = parser.parse_args()


def process_img(img_fpath):

    # ignore cells where only a few pixels are critical

    # original_img_fpath = bs.get_original_image_fpath(img_fpath)

    bs.CRIT_PIX_THRESHOLD = 15


    fname = os.path.split(img_fpath)[-1]

    # if f"{he.img_basename}{he.img_ext}" not in FILES:
    # if fname not in FILES:
    #     # save time: only process known images
    #     print("nope")
    #     return
    he = bs.HistEvaluation(img_fpath, suffix=args.suffix, ev_crit_pix=True)

    he.initialize_hist_cache()

    # default values:
    save_options = {
        "create_experimental_img": True,
        "blend_hard": args.blend_mode == 1,
        "blend_value": args.blend_value,
        "print_std_deviation": args.print_std_deviation,
        "crit_score_thresh": 80,
    }

    err_list = []
    try:
        crit_cell_list = he.find_critical_cells_for_img(exclude_cell_keys=exclude_cell_keys, save_options=save_options)
    except Exception as ex:
        print(img_fpath, ex)
        img_fname = os.path.split(img_fpath)[-1]
        err_list.extend([img_fname, str(ex)])
        crit_cell_list = None
        raise

    if args.blend_mode:
        fsuffix = f"hard"
    else:
        fsuffix = f"soft"
    fsuffix = f"{fsuffix}_{args.blend_value}"
    summary = he.get_criticality_summary()
    if summary.crit_score_avg > 50:

        # bs.db["criticality_summary"][he.img_basename] = dict(summary.item_list())
        # bs.db.commit()
        bs.db.put("criticality_summary", he.img_basename, value=dict(summary.item_list()), commit=True)

        score_str = str(int(summary.crit_score_avg))
        he.save_experimental_img(fprefix=f"S{score_str}_", fsuffix=fsuffix)


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


# @aiot.background
def run_this_script(img_path, **kwargs):

    options = kwargs.get("options", {})
    option_str_elements = []
    for option, value in options.items():
        if value is True:
            option_str_elements.append(option)

    option_str = " ".join(option_str_elements)

    option_str += f"-bm {args.blend_mode} -bv {args.blend_value}"

    new_suffix = f"{args.suffix}_bm{args.blend_mode}_bv{args.blend_value}"

    cmd = f"{sys.executable} {__file__} --img {img_path} --suffix {new_suffix} {option_str}".strip()
    print(cmd)
    res = os.system(cmd)
    res = 0

    if res != 0:
        print("ERROR\n", cmd)
        ERROR_CMDS.append(cmd)


def aio_main():

    update_cell_mappings()

    arg_list = bs.get_img_list(args.img_dir, limit=args.limit)

    # prepare options for passing to individual calls
    options = {}

    if args.no_parallel:
        func = run_this_script
        for i, arg in enumerate(arg_list):
            # print(i)
            func(arg, options=options)
    else:
        wrapped_func=aiot.background(run_this_script)
        aiot.run(aiot.main(func=wrapped_func, arg_list=arg_list, options=options))

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