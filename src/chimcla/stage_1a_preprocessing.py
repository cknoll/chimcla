"""
This module contains several preprocessing steps which were distributed over multiple scripts in earlier versions
"""

import os
import argparse
import glob

import cv2
import numpy as np

from ipydex import IPS

from .asyncio_tools import background, async_run
from .util import CHIMCLA_DATA

pjoin = os.path.join

_EMPTY_SLOT_REF_IMG_PATH = pjoin(CHIMCLA_DATA, "reference", "empty_slot.jpg")
# region of interest
_EMPTY_SLOT_REF_IMG_ROI = (30, 930, 85, 600)


class Stage1Preprocessor:
    def __init__(self, args):
        # see cli.py for arg-definitions
        # preparation for step 1
        self.args = args
        self.img_dir = args.img_dir.rstrip("/")
        self.jpg0_target_dir_path = os.path.join(self.img_dir, "..", args.target_rel_dir)
        assert os.path.exists(self.img_dir)

        os.makedirs(self.jpg0_target_dir_path, exist_ok=True)
        self.png_path_list = glob.glob(f"{self.img_dir}/*.png")

        # preparation for step 2
        self.empty_slot_ref_image = None
        self.WIDTH_COMP = 100
        self.HEIGHT_COMP = 67
        self.PIXELS = self.WIDTH_COMP * self.HEIGHT_COMP

    def main(self):
        if self.args.no_parallel:
            for png_fpath in self.png_path_list:
                self.pipeline(png_fpath)

        else:
            # parallelization mode: apply the background-decorator only if needed
            bg_pipeline = background(self.pipeline)
            async_run(bg_pipeline, self.png_path_list)

    def pipeline(self, fpath):
        # self.step01_mogrify_1000jpg(fpath)
        self.step02_empty_slot_detection_mockup(fpath)

    def step01_mogrify_1000jpg(self, fpath):
        prefix, fname = os.path.split(fpath)

        cmd = f"mogrify -monitor -format jpg -resize 1000 -path {self.jpg0_target_dir_path} {fpath}"
        # print(cmd, "\n")
        os.system(cmd)

    def step02_empty_slot_detection_mockup(self, fpath):
        pass
    def step02_empty_slot_detection(self, fpath):
        img1 = self._load_comp_img(fpath)
        corr = self._get_correlation(img1, img_ref)
        res = self._post_process(fpath, corr)
        if res is None:
            return

        print(res)
        res_list.append(res)


    def _load_comp_img(self):
        if not self.empty_slot_ref_image is not None:
            return self.empty_slot_ref_image

        ROI = _EMPTY_SLOT_REF_IMG_ROI

        # load the image and apply ROI; note that row index (y dimension comes first)
        image1  = cv2.imread(_EMPTY_SLOT_REF_IMG_PATH)[ROI[2]:ROI[3], ROI[0]:ROI[1], :]
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        self.empty_slot_ref_image = self._resize_for_comparison(img1)
        return self.empty_slot_ref_image

    def _resize_for_comparison(self, img):


        res = cv2.resize(img, dsize=(self.WIDTH_COMP, self.HEIGHT_COMP), interpolation=cv2.INTER_CUBIC)
        # res = img  # omit resizing
        return np.array(res, dtype=float)/255

    def _get_correlation(self, img, img_ref):
        assert img.shape == img_ref.shape
        abs_diff = np.abs(img - img_ref)

        # sum of abs diff per pixel
        sadpp = np.sum(abs_diff) / self.PIXELS
        corr = np.exp(-sadpp)
        return corr


    def _post_process(self, fpath, corr):
        """
        depending on corr apply the corresponding action of the CORR_ACTION_MAP
        """
        for thr, res_template in CORR_ACTION_MAP:
            if corr > thr:
                break
        else:
            # there was no break -> do nothing
            return

        prefix, fname = os.path.split(fpath)

        res = res_template.format(original_fname=fname, IMG_DIR=IMG_DIR, corr=corr)
        return res


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
