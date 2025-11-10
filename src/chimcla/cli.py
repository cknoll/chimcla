"""
**Important**

Command line interface.
"""

import os
import sys
import argparse
import glob
import random
import tqdm

import addict

from ipydex import IPS, activate_ips_on_exception


# TODO: move this to chimcla_main interface
def create_groups():

    N = 500  # number of
    p = 0.4  # fraction of repeated images
    q = 1 - p  # fraction of group specific images
    Np = int(N*p)  # number of
    Nq = N - Np


    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=f'This program creates directories with {N} cell images each. It also creates a json report',
    )

    parser.add_argument(
        'dir',
        help="root directory",
    )

    args = parser.parse_args()
    dircontent = os.listdir(args.dir)

    subdirs = []
    elt: str
    for elt in os.listdir(args.dir):
        if elt.startswith("_"):
            continue
        path = os.path.join(args.dir, elt)
        if os.path.isdir(path):
            subdirs.append(path)

    image_paths = []
    for subdir in subdirs:
        imgs = glob.glob(f"{subdir}/s*.jpg")
        image_paths.extend(imgs)
    random.seed(1246)
    random.shuffle(image_paths)

    T = len(image_paths)
    nbr_of_groups = (T // Nq) + 1

    groups = []

    for i in range(nbr_of_groups):
        i0 = i*Nq
        i1 = (i+1)*Nq
        group = image_paths[i0:i1]
        rest  = image_paths[:i0] + image_paths[i1:]
        random.shuffle(rest)

        # the last group might be incomplete, thus we have to determine length every time
        L = len(group)
        group.extend(rest[:(N-L)])
        groups.append(group)

    for i, group in tqdm.tqdm(enumerate(groups, start=1)):
        gdir = os.path.join(args.dir, "_results", f"gruppe{i:04d}")
        os.makedirs(gdir, exist_ok=True)
        for fpath in group:
            fname = os.path.split(fpath)[1]
            cell_key = fname[:-4].split("_")[-1]
            target_path = os.path.join(gdir, f"{cell_key}_{fname}")
            cmd = f"cp {fpath} {target_path}"
            os.system(cmd)


def main():

    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="chimcla main command line interface "
        "(for all functions which do not have their own script)",
    )

    subparsers = parser.add_subparsers(dest="command", help="")
    parser_prepare_docs = subparsers.add_parser("prepare-docs", help="prepare automatic generation of docs")
    parser_build_docs = subparsers.add_parser(
        "build-docs", help="automatic generation of docs (after preparation)"
    )


    parser_build_docs.add_argument("docfile", type=str, help="optional file to build (otherwise: build all)", nargs='?', default=None)

    parser_continuously_build_docs = subparsers.add_parser(
        "continuously-build-docs", help="continuous automatic generation of docs (after preparation)"
    )
    parser_bgr_convert = subparsers.add_parser(
        "bgr-convert", help="convert jpg files from BGR to RGB and vice versa"
    )

    parser_bgr_convert.add_argument("img_dir", type=str, help="directory containing jpg files to convert")
    args = parser.parse_args()

    if args.command == "prepare-docs":
        from .util_doc import generate_module_docs
        generate_module_docs()
    elif args.command == "build-docs":
        from .util_doc import make_html_doc
        make_html_doc(args.docfile)
    elif args.command == "continuously-build-docs":
        from .util_doc import make_html_doc
        make_html_doc()
    elif args.command == "bgr-convert":
        from .util import bgr_convert
        bgr_convert(args.img_dir)
    else:
        msg = f"unknown chimcla command: {args.command}"
        print(msg)
        sys.exit(1)


def rename_cell_imgs():

    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description='This program renames image files, such that the cell comes first',
    )

    parser.add_argument(
        'dir',
        help="directory",
    )

    args = parser.parse_args()
    _rename_cell_imgs(args.dir)


def _rename_cell_imgs(dirname):
    fnames = os.listdir(dirname)

    fnames.sort()

    for fname in fnames:
        if not fname.endswith("jpg"):
            continue
        cell_key = fname[:-4].split("_")[-1]


        old_path = os.path.join(dirname, fname)
        new_path = os.path.join(dirname, f"{cell_key}_{fname}")


        cmd = f"mv {old_path} {new_path}"
        os.system(cmd)
        # print(cmd)
        # break


def create_work_images():
    """
    Perform stage 1 creating images for further research.
    """
    from . import stage_1a_preprocessing as s1a
    s1a.main()


def pipeline():
    """
    This is the entry point of the main processing pipeline
    """

    msg ="""
    Idea: it might be convenient to start the "whole" pipeline with one command
    However, currently it is unclear whether there will be a use case for this.
    """

    print(msg)
