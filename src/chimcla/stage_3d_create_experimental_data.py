"""
This file serves for
- creating images for Romys Chocolate Experiment (as discussed in Nov. 2023)
- creating data (images and numerical) for history evaluation

"""

import os
import sys
import numpy as np
import glob
import itertools as it
import collections
import argparse
from colorama import Fore, Style
import pathlib

import pandas as pd

import ipydex
from ipydex import IPS, activate_ips_on_exception, set_trace

activate_ips_on_exception()

from chimcla import stage_2a_bar_selection as bs

from chimcla import asyncio_tools as aiot



def process_img(img_fpath):

    # ignore cells where only a few pixels are critical

    # original_img_fpath = bs.get_original_image_fpath(img_fpath)

    bs.CRIT_PIX_THRESHOLD = 40
    bs.PREPROCESS_BORDERS = True


    fname = os.path.split(img_fpath)[-1]

    # if f"{he.img_basename}{he.img_ext}" not in FILES:
    # if fname not in FILES:
    #     # save time: only process known images
    #     print("nope")
    #     return


    # default values:
    save_options = {
        "create_experimental_img": True,
        "blend_hard": mm.args.blend_mode == 1,
        "blend_value": mm.args.blend_value,
        "print_std_deviation": mm.args.print_std_deviation,
        "crit_score_thresh": 40,
        # determine how much the crit_score_thresh is lowered for every crit pixel
        "crit_score_slope": -.25,
        "paper_mode": True,
        "save_plot": True,
        "push_db": False,
        "adgen_mode": False,
        "desired_cells": ["b16", "b14"],

    }

    err_list = []
    img_fname = os.path.split(img_fpath)[-1]
    try:
        debug = True
        if not debug:
            ipydex.sys.excepthook = ipydex.sys_orig_excepthook
        he = bs.HistEvaluation(
            img_fpath, suffix=mm.args.suffix, ev_crit_pix=True, history_eval_flag=mm.args.history_evaluation,
        )
        he.initialize_hist_cache()
    except (bs.MissingBoundingBoxes, ValueError) as ex:
        path = os.path.join(*pathlib.Path(img_fpath).parts[-2:])
        print(bs.yellow(path), bs.bred(str(ex)))
        if debug:
            raise ex

    try:
        crit_cell_list = he.find_critical_cells_for_img(exclude_cell_keys=mm.exclude_cell_keys, save_options=save_options)
    except Exception as ex:
        print(img_fpath, ex)
        err_list.extend([img_fname, str(ex)])
        crit_cell_list = None
        raise

    if mm.args.blend_mode:
        fsuffix = f"hard"
    else:
        fsuffix = f"soft"
    fsuffix = f"{fsuffix}_{mm.args.blend_value}"
    summary = he.get_criticality_summary(save_to_db=True)
    if summary.crit_score_avg > 50:

        # bs.db["criticality_summary"][he.img_basename] = dict(summary.item_list())
        # bs.db.commit()
        bs.db.put("criticality_summary", he.img_basename, value=dict(summary.item_list()), commit=True)

        if mm.args.history_evaluation:
            sum_score_str = f"{(int(sum(summary.crit_score_list))):06d}"
            fprefix=f"S{sum_score_str}_"
            he.copy_original_image_to_output_folder(fprefix=fprefix)
        else:
            fprefix=f"P{summary.crit_pix_number}_"

            he.save_experimental_img(fprefix=fprefix, fsuffix=fsuffix)
    else:
        os.makedirs(he.output_dir, exist_ok=True)
        with open(os.path.join(he.output_dir, "_uncritical_form-images.txt"), "a") as fp:
            fp.write(f"{img_fname}\n")


from collections import defaultdict
import re


def update_cell_mappings():
    """
    creates a dict which maps filenames of form-images like "2023-06-26_08-47-55_C50.jpg"
    to a list of cells-images ["2023-06-26_08-47-55_C50_a9.jpg", ...]
    """

    pat = re.compile(".*(_.*?)\.jpg")

    # defaultdict with empty lists
    cell_mappings = defaultdict(list)

    for key in bs.db.keys():
        # example:
        # key = "2023-06-26_08-47-55_C50_a9.jpg"
        # fname = "2023-06-26_08-47-55_C50.jpg"

        if not key.endswith(".jpg"):
            continue

        cell_ = pat.match(key).group(1)
        fname = key.replace(cell_, "")

        # fill the list
        cell_mappings[fname].append(key)

    bs.db["cell_mappings"] = cell_mappings
    bs.db.commit()


# @aiot.background
def run_this_script(img_path, **kwargs):

    options = kwargs.get("options", {})


    if "--history-evaluation" in options:
        option_str_elements = []
        new_suffix = mm.args.suffix
    else:
        # original mode (for Romys experimental images)
        # these two options are default
        option_str_elements = [f"-bm {mm.args.blend_mode}", f"-bv {mm.args.blend_value}"]
        new_suffix = f"{mm.args.suffix}_bm{mm.args.blend_mode}_bv{mm.args.blend_value}"

    for option, value in options.items():
        if value is True:
            option_str_elements.append(option)

    option_str = " ".join(option_str_elements)

    # this is what is defined in pyproject.toml
    BASE_CMD = "chimcla_ced"
    # cmd = f"{sys.executable} {__file__} --img {img_path} --suffix {new_suffix} {option_str}".strip()
    cmd = f"{BASE_CMD} --img {img_path} --suffix {new_suffix} {option_str}".strip()
    print("\n")
    print(cmd)

    res = os.system(cmd)
    # res = 0  # turn safety check off in mass production
    if not res == 0:
        msg = f"The following command unexpectedly exited with a nonzero code: \n\n{cmd}"
        raise AssertionError(msg)

    if res != 0:
        print("ERROR\n", cmd)
        mm.ERROR_CMDS.append(cmd)


def aio_main():

    update_cell_mappings()

    if mm.args.img_dir:
        arg_list = bs.get_img_list(mm.args.img_dir, limit=mm.args.limit)

        # this once was a special use case (might become relevant again): only consider known critical images
        # arg_list = get_known_critical_images(bs.get_img_list(args.img_dir))[:args.limit]

    else:
        assert mm.args.img_dir_base
        assert mm.args.img_dir_src
        arg_list = get_img_list_from_src_dir()[:mm.args.limit]


    # prepare options for passing to individual calls

    # TODO: this could be more general but currently we only need few options
    options = {
        "--history-evaluation": mm.args.history_evaluation,
    }

    if mm.args.no_parallel:
        func = run_this_script
        for i, arg in enumerate(arg_list):
            # print(i)
            func(arg, options=options)
    else:
        wrapped_func=aiot.background(run_this_script)
        aiot.run(aiot.main(func=wrapped_func, arg_list=arg_list, options=options))

    if mm.args.history_evaluation:
        generate_criticality_list(arg_list)

    if mm.ERROR_CMDS:
        print(
            f"\n{Fore.RED}There where errors with the following commands:\n\n",
            Style.RESET_ALL,
            "\n".join(mm.ERROR_CMDS),
        )


def get_known_critical_images(img_fpaths):
    """
    This function is for debugging and development to handle only those images who are already known to be critical
    """
    res = []
    cs = bs.db["criticality_summary"]

    for fpath in img_fpaths:
        basename = os.path.splitext(os.path.split(fpath)[-1])[0]
        if basename in cs:
            res.append(fpath)
    return res


def generate_criticality_list(img_fpaths: list):

    assert img_fpaths
    cs = bs.db["criticality_summary"]
    csv_data = defaultdict(list)

    for fpath in img_fpaths:
        basename = os.path.splitext(os.path.split(fpath)[-1])[0]
        res = cs.get(basename)
        if res is None:
            continue

        csv_data["basename"].append(basename)
        csv_data["criticality"].append(np.round(sum(res["crit_score_list"]), 2))

    df = pd.DataFrame(csv_data)
    df2 = df.sort_values(by=['criticality'], ascending=False)

    output_dir = bs.db["meta_data"]["output_dir"]

    csv_fpath = os.path.join(output_dir, "_criticality_list.csv")
    df2.to_csv(csv_fpath)
    print(f"written: {csv_fpath}")


def get_img_list_from_src_dir():

    # taken from stage_3e
    dn = mm.args.img_dir_src
    # IPS()
    paths = glob.glob(f"{dn}/*jpg")
    base_names0 = [p.replace(f"{dn}/", "") for p in paths if "exp_" not in p]
    base_names = ["_".join(bn.split("_")[1:]) for bn in base_names0]
    base_names_set = set(base_names)

    dn2 = mm.args.img_dir_base
    original_img_paths = glob.glob(f"{dn2}/*_shading_corrected/*jpg")

    relevant_img_paths = []
    for p in original_img_paths:
        if os.path.split(p)[1] in base_names_set:
            relevant_img_paths.append(p)

    return relevant_img_paths



def main2():

    if mm.args.img:
        process_img(mm.args.img)
    elif mm.args.img_dir:
        aio_main()
    elif mm.args.img_dir_base:
        # only process images which have been found to be relevant by another run
        # used to create new color variations

        # py3 stage_3d_create_experimental_data.py --img_dir_base /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped --img_dir_src experimental_imgs_psy01_bm0_bv60 --suffix _psy01 -bm 0 -bv 60
        # py3 stage_3d_create_experimental_data.py --img_dir_base /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped --img_dir_src experimental_imgs_psy01_bm0_bv60 --suffix _psy01 -bm 1 -bv 110
        assert not mm.args.img_dir
        aio_main()
    else:
        mm.parser.print_help()


class MainManager:
    def __init__(self):
        self.exclude_cell_keys = [("a", "1"), ("b", "1"), ("c", "1")]
        self.ERROR_CMDS = []

        self.parser = argparse.ArgumentParser(
                prog=sys.argv[0],
                description='This program evaluates single or multiple chocolate images. See docstring for more details.',
            )
        self.args = None
mm = MainManager()


def main():
    # TODO: move this into ManagerClass

    mm.parser.add_argument(
        '--img_dir',
        help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0",
        default=None,
    )

    # see usage example below
    mm.parser.add_argument(
        '--img_dir_base',
        help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/",
        default=None,
    )


    # see usage example below
    mm.parser.add_argument(
        '--img_dir_src',
        help="directory with already processed images -> extract the basename",
        default=None,
    )

    mm.parser.add_argument(
        '--img',
        help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0/2023-06-26_06-17-41_C50.jpg",
        default=None,
    )

    mm.parser.add_argument(
        "--suffix",
        "-s",
        help="specify suffix for output folder",
        default="_chunk_psy_test",
    )


    mm.parser.add_argument(
        "--no-parallel",
        "-np",
        help="sequential mode (no parallelization)",
        action="store_true",
    )

    mm.parser.add_argument(
        "--limit",
        help="limit the number of processed files",
        default=None,
        type=int,
    )


    # parameters for Romys experiments
    mm.parser.add_argument(
        "--blend-value",
        "-bv",
        default=120,
        type=int,
    )

    mm.parser.add_argument(
        "--blend-mode",
        "-bm",
        help= "0 (soft, default) or 1 (hard)",
        default=0,
        type=int,
    )

    mm.parser.add_argument(
        "--print-std-deviation",
        "-std",
        help= "print standard deviation of critical pixels for each critical cell onto the image",
        action="store_true",
    )

    # parameters for history evaluation
    # This means: iterate over images
    # generate:
    # - _criticality_list.csv: col1 a sorted list (summed criticality score) col2: filename
    # - output directory with critical images with Filenames S2801_<basename>.jpg

    mm.parser.add_argument(
        "--history-evaluation",
        "-H",
        help= "generate data for history evaluation",
        action="store_true",
    )

    mm.args = mm.parser.parse_args()
    main2()


if __name__ == "__main__":#
    main()  # parse args, then call main
