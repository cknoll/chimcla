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

from ipydex import IPS, set_trace

from .asyncio_tools import background, async_run
from .util import CHIMCLA_DATA, ImageInfoContainer, handle_error
from . import util

pjoin = os.path.join


# the presence of this file indicates that its parent directory is the root
# of all relevant chimcla-data (see README.md)
CHIMCLA_DATA_INDICATOR_FNAME = "__chimcla_data__.txt"

_EMPTY_SLOT_REF_IMG_PATH = pjoin(CHIMCLA_DATA, "reference", "empty_slot.jpg")
# region of interest
_EMPTY_SLOT_REF_IMG_ROI = (30, 930, 85, 600)



class Stage1Preprocessor:
    def __init__(self, args):
        # general preparations
        # see cli.py for arg-definitions
        self.args = args
        self.iic_map: Dict[str, ImageInfoContainer] = {}

        self.prefix = args.prefix
        self.original_img_dir = args.img_dir.rstrip("/")

        self.data_base_dir = None
        self.rel_part_path = None
        self._rel_part_path_list = []
        self.get_data_base_dir(start=self.original_img_dir)

        # assume something like `['lots', '2024-09-17', 'part000']`
        assert len(self._rel_part_path_list) > 1
        self.output_base = pjoin(
            self.data_base_dir, f"{self.prefix}result", *self._rel_part_path_list[1:]
        )

        # preparation for step 1
        assert os.path.exists(self.original_img_dir)

        # note: during the course of the project raw image format switched from png to jpg
        self.orig_img_path_list = util.get_png_or_jpg_list(self.original_img_dir)

        self.jpg0_target_dir_path = pjoin(self.output_base, "jpg0")
        os.makedirs(self.jpg0_target_dir_path, exist_ok=True)

        # preparation for step 2
        self.empty_slot_ref_image = None
        self.WIDTH_COMP = 100
        self.HEIGHT_COMP = 67
        self.PIXELS = self.WIDTH_COMP * self.HEIGHT_COMP
        self.img_ref = self.get_img_for_empty_slot_comp(_EMPTY_SLOT_REF_IMG_PATH)

        # preparation for step 3
        self.cropped_target_dir_path = pjoin(self.output_base, "cropped")
        os.makedirs(self.cropped_target_dir_path, exist_ok=True)

        # preparation for step 4
        self.shading_correction_matrix_fpath = pjoin(CHIMCLA_DATA,"shading_correction_matrix.npy")
        self.shading_corrected_target_dir_path = pjoin(self.output_base, "shading_corrected")
        os.makedirs(self.shading_corrected_target_dir_path, exist_ok=True)

        # for debugging/experiments
        self.async_counter = 0
        self.async_res = []

    def main(self):
        if self.args.no_parallel:
            for png_fpath in self.orig_img_path_list:
                self.pipeline(png_fpath)
        else:
            # parallelization mode: apply the background-decorator only if needed
            bg_pipeline = background(self.pipeline)
            async_run(bg_pipeline, self.orig_img_path_list)

    def pipeline(self, fpath):
        # important: use iic as a local variable here because this method will be
        # run in parallel and write access to instance variables is shared
        iic = self.iic_map[fpath] = ImageInfoContainer(fpath, data_base_dir=self.data_base_dir)
        self.step01_mogrify_1000jpg(iic)
        self.step02_empty_slot_detection(iic)
        self.step03_cropping(iic)
        self.step04_shading_correction(iic)

    @handle_error
    def step01_mogrify_1000jpg(self, iic: ImageInfoContainer):

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
        self.async_counter += 1
        local_value = self.async_counter * 1
        self.async_res.append(f"starting {local_value}, {self.async_counter}")
        time.sleep(6 - self.async_counter * 0.5)
        self.async_res.append(f"result {local_value}, {self.async_counter}")

    @handle_error
    def step02_empty_slot_detection(self, iic: ImageInfoContainer):
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

    @handle_error
    def step03_cropping(self, iic: ImageInfoContainer):

        # this is the complete area where the form can be (later there will be another ROI)
        ROI = (30, 930, 85, 600)
        # ROI = (0, None, 0, None)  # complete image

        # load the image and apply ROI; note that row index (y dimension comes first)
        img = cv2.imread(iic.latest_fpath)[ROI[2] : ROI[3], ROI[0] : ROI[1], :]
        LL = self._lightness_curve_of_y(img)
        y_idx_edge = self._get_lower_edge(LL)

        FORM_HEIGHT = 460  # (earlier: `HEIGHT_ROI`)

        cropped_img = img[y_idx_edge - FORM_HEIGHT : y_idx_edge, :]
        new_path = pjoin(self.cropped_target_dir_path, iic.fname_jpg)

        try:
            cv2.imwrite(new_path, cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        except cv2.error as ex:
            iic.error = "error during cropping"
            iic.messages.append(f"error during cropping: {ex}")

        iic.latest_fpath = new_path

    @handle_error
    def step04_shading_correction(self, iic: ImageInfoContainer):
        """
        Apply a predefined correction matrix to the lightness channel
        """
        correction_matrix = np.load(self.shading_correction_matrix_fpath)

        img = cv2.imread(iic.latest_fpath)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab_img)

        L_new = np.array(np.clip(L*correction_matrix, 0, 255), dtype=np.uint8)
        img_new = cv2.cvtColor(cv2.merge((L_new, a, b)), cv2.COLOR_LAB2BGR)

        new_path = pjoin(self.shading_corrected_target_dir_path, iic.fname_jpg)
        res = cv2.imwrite(new_path, img_new, [cv2.IMWRITE_JPEG_QUALITY, 98])

        assert res, f"Something went wrong during the creation of {new_path}"

        iic.latest_fpath = new_path

    # auxiliary methods for step02

    def get_img_for_empty_slot_comp(self, img_fpath):
        ROI = _EMPTY_SLOT_REF_IMG_ROI

        # load the image and apply ROI; note that row index (y dimension comes first)
        image1 = cv2.imread(img_fpath)[ROI[2] : ROI[3], ROI[0] : ROI[1], :]
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        self.empty_slot_ref_image = self._resize_for_comparison(img1)
        return self.empty_slot_ref_image

    def _resize_for_comparison(self, img):

        res = cv2.resize(img, dsize=(self.WIDTH_COMP, self.HEIGHT_COMP), interpolation=cv2.INTER_CUBIC)
        # res = img  # omit resizing
        return np.array(res, dtype=float) / 255

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
            img = np.array(img * 255, dtype=np.uint8)

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
        idx1 = int(len(LL) * 0.8)
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

    def get_data_base_dir(self, start: str):
        start = os.path.abspath(start)
        assert os.path.isdir(start)

        if os.path.isfile(pjoin(start, CHIMCLA_DATA_INDICATOR_FNAME)):
            return start

        new_start, segment = os.path.split(start)
        self._rel_part_path_list.insert(0, segment)
        if new_start == start:
            msg = (
                f"Unexpectedly could not find {CHIMCLA_DATA_INDICATOR_FNAME} "
                f"(starting with {self.original_img_dir})"
            )
            raise FileNotFoundError(msg)

        # recursively call this function
        res = self.get_data_base_dir(start=new_start)
        self.data_base_dir = res
        self.rel_part_path = pjoin(*self._rel_part_path_list)
        return res


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(
            prog="stage_0f_resize_and_jpg",
            description="This program corrects resizes the original png files and converts to jpg",
        )

        parser.add_argument("img_dir", help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_roh_aus_peine_ab_2023-07-31")
        parser.add_argument(
            "prefix", help="prefix for newly created dirs during preprocessing", nargs="?", default="pp_"
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