import os
import sys
import argparse
import glob

import cv2

from ipydex import IPS


def create_groups():

    N = 500

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

    IPS()


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
