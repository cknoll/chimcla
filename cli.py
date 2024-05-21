import os
import sys
import argparse
import glob
import random

import cv2

from ipydex import IPS


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

    for i, group in enumerate(groups, start=1):
        gdir = os.path.join(args.dir, "_results", f"group{i:04d}")
        os.makedirs(gdir, exist_ok=True)
        for fpath in group:
            fname = os.path.split(fpath)[1]
            target_path = os.path.join(gdir, fname)
            cmd = f"cp {fpath} {target_path}"
            os.system(cmd)


def bgr_convert():


    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description='This program converts jpg files from BGR TO RGB and vice versa.',
    )


    parser.add_argument(
        'dir',
        help="directory",
    )

    args = parser.parse_args()
    fnames = os.listdir(args.dir)

    fnames.sort()

    for fname in fnames:
        if not fname.lower().endswith("jpg"):
            continue
        fpath = os.path.join(args.dir, fname)
        img  = cv2.imread(fpath)
        if img is None:
            print(f"could not read {fpath}")
            continue
        try:
            pass
            res = cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        except Exception as e:
            print(f"!! {e}")
            continue
        print(f"{fname} done")


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
    fnames = os.listdir(args.dir)

    fnames.sort()

    for fname in fnames:
        if not fname.endswith("jpg"):
            continue
        cell_key = fname[:-4].split("_")[-1]


        old_path = os.path.join(args.dir, fname)
        new_path = os.path.join(args.dir, f"{cell_key}_{fname}")


        cmd = f"mv {old_path} {new_path}"
        os.system(cmd)
        # print(cmd)
        # break
