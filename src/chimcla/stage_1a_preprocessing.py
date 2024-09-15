"""
This module contains several preprocessing steps which were distributed over multiple scripts in earlier versions
"""

import os
import argparse
import glob
from ipydex import IPS

from .asyncio_tools import background, async_run

pjoin = os.path.join


class Stage1Preprocessor:
    def __init__(self, args):
        # see cli.py for arg-definitions
        self.args = args
        self.img_dir = args.img_dir.rstrip("/")
        self.jpg0_target_dir_path = os.path.join(self.img_dir, "..", args.target_rel_dir)
        assert os.path.exists(self.img_dir)

        os.makedirs(self.jpg0_target_dir_path, exist_ok=True)

        self.png_path_list = glob.glob(f"{self.img_dir}/*.png")

    def main(self):
        if self.args.no_parallel:
            for png_fpath in self.png_path_list:
                self.pipeline(png_fpath)

        else:
            # parallelization mode
            bg_pipeline = background(self.pipeline)
            async_run(bg_pipeline, self.png_path_list)

    def pipeline(self, fpath):
        self.step01_mogrify_1000jpg(fpath)

    def step01_mogrify_1000jpg(self, fpath):
        prefix, fname = os.path.split(fpath)

        cmd = f"mogrify -monitor -format jpg -resize 1000 -path {self.jpg0_target_dir_path} {fpath}"
        # print(cmd, "\n")
        os.system(cmd)


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(
            prog='stage_0f_resize_and_jpg',
            description='This program corrects resizes the original png files and converts to jpg',
        )

        parser.add_argument('img_dir', help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_roh_aus_peine_ab_2023-07-31")
        parser.add_argument(
            "target_rel_dir", help="target directory (relative to img_dir/..)", nargs="?", default="bilder_jpg0"
        )
        parser.add_argument(
            "--no-parallel",
            "-np",
            help="sequential mode (no parallelization)",
            action="store_true",
        )

        args = parser.parse_args()

    s1p = Stage1Preprocessor(args)
    s1p.main()
