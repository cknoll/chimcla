"""
This script converts jpg files from BGR TO RGB and vice versa.

Not yet included in cli.py.
"""

import os
import sys
import argparse

import cv2

from ipydex import IPS


parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description='This program converts jpg files from BGR TO RGB and vice versa.',
)


parser.add_argument(
    'dir',
    help="directory",
)


def main():

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


if __name__ == "__main__":
    main()