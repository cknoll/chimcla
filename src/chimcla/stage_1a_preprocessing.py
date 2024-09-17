"""
This module contains several preprocessing steps which were distributed over multiple scripts in earlier versions
"""

import os
import argparse
import glob
from typing import Dict
import collections

import cv2
import numpy as np
import scipy as sc
import time

from ipydex import IPS

from .asyncio_tools import background, async_run
from .util import CHIMCLA_DATA

pjoin = os.path.join

_EMPTY_SLOT_REF_IMG_PATH = pjoin(CHIMCLA_DATA, "reference", "empty_slot.jpg")
# region of interest
_EMPTY_SLOT_REF_IMG_ROI = (30, 930, 85, 600)

class ImageInfoContainer:
    """
    Class to track information about individual images
    """
    def __init__(self, original_fpath):
        self.original_fpath = original_fpath
        self.latest_fpath = original_fpath
        self.original_dirpath, self.fname = os.path.split(original_fpath)
        self.basename, _ = os.path.splitext(self.fname)
        self.fname_jpg = f"{self.basename}.jpg"

        self.step01_fpath = None
        self.step02_fpath = None
        self.step03_fpath = None
        self.step04_fpath = None
        self.step05_fpath = None

        self.error = None
        self.messages = []

    def __repr__(self):

        if self.error:
            err_flag = "(err) "
        else:
            err_flag = ""

        return f"<IIC {err_flag} {self.fname}>"


class Stage1Preprocessor:
    def __init__(self, args):
        # general preparations
        # see cli.py for arg-definitions
        self.args = args
        self.iic_map: Dict[str, ImageInfoContainer]= {}

        # self.prefix = args.prefix
        # preparation for step 1
        self.img_dir = args.img_dir.rstrip("/")
        assert os.path.exists(self.img_dir)
        self.png_path_list = glob.glob(f"{self.img_dir}/*.png")

        self.jpg0_target_dir_path = os.path.abspath(pjoin(self.img_dir, "..", args.target_rel_dir))
        os.makedirs(self.jpg0_target_dir_path, exist_ok=True)

        # preparation for step 2
        self.empty_slot_ref_image = None
        self.WIDTH_COMP = 100
        self.HEIGHT_COMP = 67
        self.PIXELS = self.WIDTH_COMP * self.HEIGHT_COMP
        self.img_ref = self.get_img_for_empty_slot_comp(_EMPTY_SLOT_REF_IMG_PATH)

        # preparation for step 3
        self.cropped_target_dir_path = f"{self.jpg0_target_dir_path}_cropped"
        os.makedirs(self.cropped_target_dir_path, exist_ok=True)

        # preparation for step 4
        self.shading_corrected_target_dir_path = f"{self.jpg0_target_dir_path}_shading_corrected"
        os.makedirs(self.shading_corrected_target_dir_path, exist_ok=True)


        # for debugging/experiments
        self.async_counter = 0
        self.async_res = []

    def main(self):
        if self.args.no_parallel:
            for png_fpath in self.png_path_list:
                self.pipeline(png_fpath)
        else:
            # parallelization mode: apply the background-decorator only if needed
            bg_pipeline = background(self.pipeline)
            async_run(bg_pipeline, self.png_path_list)

    def pipeline(self, fpath):
        # important: use iic as a local variable here because this method will be
        # run in parallel and write access to instance variables is shared
        iic = self.iic_map[fpath] = ImageInfoContainer(fpath)
        self.step01_mogrify_1000jpg(iic)
        self.step02_empty_slot_detection(iic)
        self.step03_cropping(iic)


    def step01_mogrify_1000jpg(self, iic: ImageInfoContainer):
        if iic.error is not None:
            return

        iic.step01_fpath = iic.latest_fpath
        cmd = f"mogrify -monitor -format jpg -resize 1000 -path {self.jpg0_target_dir_path} {iic.step01_fpath}"
        # print(cmd, "\n")
        res = os.system(cmd)
        if res != 0:
            iic.error = "nonzero exit code of mogrify"
            iic.messages.append(f"nonzero exit code of mogrify: {res}")
            return

        # generate the filename for the next step
        iic.latest_fpath = pjoin(self.jpg0_target_dir_path, iic.fname_jpg)

    def debug_step02_async_experiments(self, iic: ImageInfoContainer):
        """
        This method serves to experiment with the async execution.
        Results:
            - instance variables might be changed from outside
            - local variables remain unchanged
            - self.async_res.append(...) works (ordering is like it is)
        """
        self.async_counter +=1
        local_value = self.async_counter*1
        self.async_res.append(f"starting {local_value}, {self.async_counter}")
        time.sleep(6-self.async_counter*0.5)
        self.async_res.append(f"result {local_value}, {self.async_counter}")

    def step02_empty_slot_detection(self, iic: ImageInfoContainer):
        if iic.error is not None:
            return

        iic.step02_fpath = iic.latest_fpath

        # load that part of the image which is relevant to compare for empty slots
        img_empty_slot_comp_part = self.get_img_for_empty_slot_comp(iic.step02_fpath)
        corr = self._get_correlation(img_empty_slot_comp_part, self.img_ref)

        # post process:
        if corr > 0.95:
            iic.error = "empty slot detected via correlation"
            iic.messages.append(f"empty slot detected via correlation ({corr}>0.95)")
        elif corr > 0.65:
            iic.error = "empty slot probable via correlation"
            iic.messages.append(f"empty slot probable via correlation ({corr}>0.65)")

        # `iic.latest_fpath` remains unchanged in this step

    def step03_cropping(self, iic: ImageInfoContainer):
        if iic.error is not None:
            return

        # this is the complete area where the form can be (later there will be another ROI)
        ROI = (30, 930, 85, 600)
        # ROI = (0, None, 0, None)  # complete image

        # load the image and apply ROI; note that row index (y dimension comes first)
        img = cv2.imread(iic.latest_fpath)[ROI[2]:ROI[3], ROI[0]:ROI[1], :]
        LL = self._lightness_curve_of_y(img)
        y_idx_edge = self._get_lower_edge(LL)

        FORM_HEIGHT = 460  # (earlier: `HEIGHT_ROI`)

        cropped_img = img[y_idx_edge - FORM_HEIGHT:y_idx_edge, :]
        new_path = pjoin(self.cropped_target_dir_path, iic.fname_jpg)

        try:
            cv2.imwrite(new_path, cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        except cv2.error as ex:
            iic.error = "error during cropping"
            iic.messages.append(f"error during cropping: {ex}")

        iic.latest_fpath = new_path

    # auxiliary methods for step02

    def get_img_for_empty_slot_comp(self, img_fpath):
        ROI = _EMPTY_SLOT_REF_IMG_ROI

        # load the image and apply ROI; note that row index (y dimension comes first)
        image1  = cv2.imread(img_fpath)[ROI[2]:ROI[3], ROI[0]:ROI[1], :]
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

    @staticmethod
    def _lightness_curve_of_y(img: np.ndarray):
        """
        :param img:     ndarray with shape[2] = 3 (BGR color convention)
        """

        if img.dtype not in (np.uint8, int, np.int16):
            img = np.array(img*255, dtype=np.uint8)

        # assume BGR color convention

        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab_img)

        # average over all x values
        L_of_y = np.average(L, axis=1)
        return L_of_y

    @staticmethod
    def _get_lower_edge(LL):
        """
        :param LL:  1d lightness arrow (averaged of x direction) in dependence of y
        """

        # lower 20% of the image
        idx1 = int(len(LL)*0.8)
        part1 = LL[idx1:]

        # Index of the darkest point of the curve
        idx2 = idx1 + np.argmin(part1)

        # this is the part where it gets darker
        part2 = LL[idx1:idx2]

        part2_diff = np.diff(part2)

        # empirically determined; might depend on resolution
        sigma = 2
        part2_diff_filter = sc.ndimage.gaussian_filter1d(part2_diff, sigma=sigma)

        # get the point of the maximum negative change rate of lightness
        idx3 = np.argmin(part2_diff_filter)
        return idx1 + idx3

    # general evaluation methods

    def get_error_report(self):
        """
        return a dict which maps from error messages to iic
        used for testing
        """
        res = collections.defaultdict(list)
        for iic in self.iic_map.values():
            res[iic.error].append(iic)

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

    # return the omo-like object for testing
    return s1p
