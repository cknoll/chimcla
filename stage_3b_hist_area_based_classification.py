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

# activate_ips_on_exception()

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
    default="_chunk_test",
)

parser.add_argument(
    "--generate-training-data",
    "-td",
    help="generate training data (store raw cell images, also process uncritical images)",
    action="store_true",
)

parser.add_argument(
    "--no-imgs",
    "-ni",
    help="do not create output images (only write to db-file)",
    action="store_true",
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
    "--adgen-mode",
    "-ad",
    help="activate annotation data generation mode (default is analysis mode)",
    action="store_true",
)

args = parser.parse_args()


dict_dir = "dicts"
os.makedirs(dict_dir, exist_ok=True)


def process_img(img_fpath):

    training_data_flag = args.generate_training_data

    # only for debugging
    # bs.PREPROCESS_BORDERS = True

    he = bs.HistEvaluation(
        img_fpath, suffix=args.suffix, ev_crit_pix=True, training_data_flag=training_data_flag,
    )
    he.initialize_hist_cache()

    # default values:
    save_options = {"save_plot": True, "push_db": True, "adgen_mode": args.adgen_mode}

    if args.no_imgs:
        save_options.pop("save_plot")

    err_list = []
    try:
        crit_cell_list = he.find_critical_cells_for_img(exclude_cell_keys=exclude_cell_keys, save_options=save_options)
    except Exception as ex:
        print(img_fpath, ex)
        img_fname = os.path.split(img_fpath)[-1]
        err_list.extend([img_fname, str(ex)])
        crit_cell_list = None
        raise
    he.save_eval_res(img_fpath, crit_cell_list, err_list)



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

    arg_list = bs.get_img_list(args.img_dir)[:args.limit]
    func = run_this_script

    # prepare options for passing to individual calls
    options = {}
    if args.generate_training_data:
        options["--generate-training-data"] =  True
    if args.no_imgs:
        options["--no-imgs"] =  True
    if args.adgen_mode:
        options["--adgen-mode"] =  True

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

    bs.HistEvaluation.reset_result_files(suffix=args.suffix)

    if args.img:
        process_img(args.img)
    elif args.img_dir:
        aio_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()