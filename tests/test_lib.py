import os
import unittest
import glob
import shutil

import pytest

# Note: for performance reasons some imports are moved inside the tests

from ipydex import IPS, Container

# run eg with `pytest -s -k test01``

pjoin = os.path.join

dir_of_this_file = os.path.dirname(__file__)
TEST_DATA_DIR = pjoin(dir_of_this_file, "testdata")

TEST_PREFIX = "tmp_pp_"

TEST_LOT_01 = pjoin(TEST_DATA_DIR, "lots", "2024-09-17", "part000")


class TestCases1(unittest.TestCase):

    # this is to access custom command line options passed to pytest
    # see also conftest.py
    @pytest.fixture(autouse=True)
    def setup(self, keep_data, no_parallel):
        # this is not to be confused with TestCase.setUp
        self.keep_data = keep_data
        self.no_parallel = no_parallel

    def tearDown(self):
        if self.keep_data:
            return

        dir_list = self.get_tmp_data_dirs()
        for dir in dir_list:
            shutil.rmtree(dir, ignore_errors=True)

    def get_tmp_data_dirs(self):
        dir_list = glob.glob(pjoin(TEST_DATA_DIR, f"{TEST_PREFIX}*"))
        return dir_list

    def get_png_dir_path(self):

        PNG_DIR = TEST_LOT_01
        png_pattern = pjoin(PNG_DIR, "*.png")
        os.makedirs(PNG_DIR, exist_ok=True)
        png_files = glob.glob(png_pattern)
        jpg_pattern = pjoin(TEST_DATA_DIR, "_jpg_templates", "*.jpg")
        jpg_files = glob.glob(jpg_pattern)
        if len(png_files) < len(jpg_files):
            # create png files from jpg files (needs to be run only once)
            for jpg_fpath in jpg_files:
                cmd = f"mogrify -monitor -format png -resize 3000 -path {PNG_DIR} {jpg_fpath}"
                # print(cmd)
                os.system(cmd)

        return PNG_DIR

    def test000__sqlite_db(self):
        from chimcla import stage_2a_bar_selection as bs

        db = bs.db

        # keys: CCI-fnames like "2023-06-26_23-20-31_C0.jpg"
        # values:  list of critical cell-imgs like ['2023-06-26_23-20-31_C0_a16.jpg', '2023-06-26_23-20-31_C0_a26.jpg', ...]

        # if this fails the file data/file-info.sqlite is missing
        self.assertIn("cell_mappings", db)
        cmp = db["cell_mappings"]

    def test010__preprocessing(self):
        from chimcla import stage_1a_preprocessing as s1a

        png_dir_path = self.get_png_dir_path()

        args = Container(img_dir=png_dir_path, prefix=TEST_PREFIX, no_parallel=self.no_parallel)

        # be sure that no leftover from last test is lurking around
        self.assertEqual(len(self.get_tmp_data_dirs()), 0)

        # get the preprocessor object
        ppo = s1a.main(args)

        jpg0_files = glob.glob(pjoin(ppo.jpg0_target_dir_path, "*.jpg"))
        self.assertEqual(len(jpg0_files), 5)

        err_report_dict = ppo.get_error_report()
        self.assertEqual(len(err_report_dict[None]), 3)
        self.assertEqual(len(err_report_dict["empty slot probable via correlation"]), 1)
        self.assertEqual(len(err_report_dict["error during cropping"]), 1)

        shading_corrected_files = glob.glob(pjoin(ppo.shading_corrected_target_dir_path, "*.jpg"))
        self.assertEqual(len(shading_corrected_files), 3)

    def test020__bboxes(self):
        from chimcla import stage_2a_bar_selection as bs

        tmp_path = pjoin(TEST_DATA_DIR, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        ccia = bs.CavityCarrierImageAnalyzer(tmp_path, bboxes=True)

    def test030__symloghist(self):
        from chimcla import stage_2a_bar_selection as bs

        tmp_path = pjoin(TEST_DATA_DIR, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        bs.get_symlog_hist(tmp_path, *"a 20".split(), dc=None)

    def test040__find_critical(self):
        from chimcla import stage_2a_bar_selection as bs

        tmp_path = pjoin(TEST_DATA_DIR, "stage1_completed", "2023-06-26_06-19-25_C50.jpg")
        he = bs.HistEvaluation(img_fpath=tmp_path)
        he.initialize_hist_cache()
        he.find_critical_cells_for_img(save_options={"save_plot": False, "push_db": False})
