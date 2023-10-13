import os
import sys
import numpy as np
import glob
import itertools as it
import collections
import argparse

import dill

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

args = parser.parse_args()



END = None
cell_tups = list(it.product("abc", np.array(range(1, 28), dtype=str)))[:END]


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

    for cell_tup in cell_tups:
        # print("".join(cell_tup), end="; ")
        try:
            hist_raw, hist_smooth = get_symlog_hist(img_fpath, *cell_tup, delta=1, dc=dc)
        except Exception as ex:
            hist_cache["bad_cells"][img_fpath].append(cell_tup)
            print(f"{type(ex)}: bad cell {img_fpath.split('/')[-1]}: {cell_tup}")
            hist_smooth = None
            dc.angle = None
        hist_cache[cell_tup].append(hist_smooth)
        hist_cache["angles"][cell_tup] = dc.angle

    # now we have a histogram for every cell of the image

    # TODO: replace hardcoded suffix
    he = bs.HistEvaluation(suffix="_chunk002")
    crit_cell_list = he.find_critical_cells_for_hist_dict(hist_cache, img_fpath)
    he.save_eval_res(img_fpath, crit_cell_list)

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
    cmd = f"{sys.executable} {__file__} --img {img_path}"
    # print(cmd)
    os.system(cmd)


def aio_main():

    img_path_list = get_img_list(args.img_dir)[:50]
    aiot.run(aiot.main(func=run_this_script, arg_list=img_path_list))


def main():

    if args.img:
        process_img(args.img)
    elif args.img_dir:
        aio_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()