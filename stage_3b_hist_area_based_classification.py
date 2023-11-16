import os
import sys
import numpy as np
import glob
import itertools as it
import collections
import argparse
from colorama import Fore, Style

import dill

from ipydex import IPS

from stage_2a_bar_selection import (
    load_img,
    rgb,
    get_bbox_list,
    assign_row_col,
    index_combinations,
    find_missing_boxes,
    handle_missing_boxes,
    select_bar_from_file,
    get_raw_cell,
    get_angle,
    correct_angle,
    rotate_img,
    get_symlog_hist,
    Container,
)

import stage_2a_bar_selection as bs


import asyncio_tools as aiot


exclude_cell_keys = [("a", "1"), ("b", "1"), ("c", "1")]


ERROR_CMDS = []


parser = argparse.ArgumentParser(
    prog='stage_2b_bar_C0_hist_dict',
    description='This program creates histogram dicts of one or more files',
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
    default="_chunk_test",
)

parser.add_argument(
    "--no-parallel",
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



CELL_KEY_END = None
cell_keys = list(it.product("abc", np.array(range(1, 28), dtype=str)))[:CELL_KEY_END]


dict_dir = "dicts"
os.makedirs(dict_dir, exist_ok=True)


def process_img(img_fpath):

    hist_cache = collections.defaultdict(list)
    hist_cache["bad_cells"] = collections.defaultdict(list)

    # this will map the cell tup to the identified angle
    hist_cache["angles"] = {}

    # use the debug-container mechanism to extract the angle from the function
    # without changing the interface
    dc = Container()

    for cell_key in cell_keys:
        try:
            hist_raw, hist_smooth = get_symlog_hist(img_fpath, *cell_key, delta=1, dc=dc)
        except Exception as ex:
            hist_cache["bad_cells"][img_fpath].append(cell_key)
            print(f"{type(ex)}: bad cell {img_fpath.split('/')[-1]}: {cell_key}")
            hist_smooth = None
            dc.angle = None
        hist_cache[cell_key].append(hist_smooth)
        hist_cache["angles"][cell_key] = dc.angle

    # now we have a histogram for every cell of the image

    he = bs.HistEvaluation(suffix=args.suffix, ev_crit_pix=True)
    err_list = []
    try:
        crit_cell_list = he.find_critical_cells_for_hist_dict(
            hist_cache, img_fpath, exclude_cell_keys=exclude_cell_keys
        )
    except Exception as ex:
        print(img_fpath, ex)
        img_fname = os.path.split(img_fpath)[-1]
        err_list.extend([img_fname, str(ex)])
        crit_cell_list = None
    he.save_eval_res(img_fpath, crit_cell_list, err_list)

def get_img_list(img_dir):

    img_path_list = glob.glob(f"{img_dir}/*.jpg")
    img_path_list.sort()

    # omit C100 images

    img_path_list2 = []
    C100_list = []
    for img_fpath in img_path_list:
        # find out if C100 with same base name is in list
        first_parts = img_fpath.split("_")[:-1]
        checkpath = f"{'_'.join(first_parts)}_C100.jpg"
        if checkpath in img_path_list:
            C100_list.append(img_fpath)
        else:
            img_path_list2.append(img_fpath)

    return img_path_list2


@aiot.background
def run_this_script(img_path):
    cmd = f"{sys.executable} {__file__} --img {img_path} --suffix {args.suffix}"
    print(cmd)
    res = os.system(cmd)

    if res != 0:
        ERROR_CMDS.append(cmd)



def aio_main():

    arg_list = get_img_list(args.img_dir)[:args.limit]
    func = run_this_script
    if args.no_parallel:
        for arg in arg_list:
            func(arg)
    else:
        aiot.run(aiot.main(func=func, arg_list=arg_list))

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