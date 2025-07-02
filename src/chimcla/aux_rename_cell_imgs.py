"""
This script renames image files, such that the cell comes first.
"""

import os
import sys
import argparse
from ipydex import IPS



def main():

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
        cell_key = fname[:-4].split("_")[-1]


        old_path = os.path.join(args.dir, fname)
        new_path = os.path.join(args.dir, f"{cell_key}_{fname}")


        cmd = f"mv {old_path} {new_path}"
        os.system(cmd)
        # print(cmd)
        # break

if __name__ == "__main__":
    main()