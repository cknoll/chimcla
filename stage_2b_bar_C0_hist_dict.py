import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import importlib as il
import glob
import itertools as it
import random
import warnings
import collections

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
)


import asyncio_tools as aiot



img_dir = "/home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0"


@aiot.background
def process_img(img_fpath):
    for cell_tup in cell_tups:
        # print("".join(cell_tup), end="; ")
        try:
            hist_raw, hist_smooth = get_symlog_hist(img_fpath, *cell_tup)
        except RuntimeError:
            bad_cells[img_fpath].append(cell_tup)
        hist_cache[cell_tup].append(hist_smooth)


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


end = None


cell_tups = list(it.product("abc", np.array(range(1, 28), dtype=str)))[:end]

hist_cache = collections.defaultdict(list)
bad_cells = collections.defaultdict(list)


def _main():

    img_path_list = get_img_list(img_dir)[:end]
    for img_fpath in img_path_list:
        print(img_fpath)
        process_img(img_fpath)

        break

if __name__ == "__main__":

    img_path_list = get_img_list(img_dir)[:25]
    aiot.run(aiot.main(func=process_img, arg_list=img_path_list))


    hist_cache["bad_cells"] = bad_cells
    fname = "hist_cache.dill"
    with open(fname, "wb") as fp:
        dill.dump(hist_cache, fp)

    from ipydex import IPS
    IPS()