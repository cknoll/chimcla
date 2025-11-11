import os
import unittest
import glob
import shutil
import tempfile
from datetime import datetime, timedelta

import pytest

# Note: for performance reasons some imports are moved inside the tests

from ipydex import IPS, Container, set_trace

from chimcla import util

# run eg with `pytest -s -k test01``

pjoin = os.path.join

dir_of_this_file = os.path.dirname(__file__)
TEST_DATA_DIR = pjoin(dir_of_this_file, "testdata")

TEST_PREFIX = "tmp_pp_"

TEST_LOT_01 = pjoin(TEST_DATA_DIR, "lots", "2024-09-17", "part000")
TEST_LOT_02 = pjoin(TEST_DATA_DIR, "lots", "2024-07-08_06-03-45__2d__56.7k", "part000")


class TestCases1(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def auto_setup(self, keep_data, no_parallel):
        """
        This method is called once for each test method (due to the autouse option).
        Argument values are taken from commandline with default False (see conftest.py).

        This method is not to be confused with `TestCase.setUp`.
        Note: After each test method `TestCase.tearDown` is called.
        """

        self.keep_data = keep_data
        self.no_parallel = no_parallel
        self.tmp_work_dir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmp_work_dir)

    def tearDown(self):
        if not self.keep_data:

            dir_list = self.get_tmp_data_dirs()
            dir_list.append(self.tmp_work_dir)
            for dir in dir_list:
                shutil.rmtree(dir, ignore_errors=True)

        os.chdir(self.orig_cwd)

    def get_tmp_data_dirs(self):
        dir_list = glob.glob(pjoin(TEST_DATA_DIR, f"{TEST_PREFIX}*"))
        return dir_list

    def ensure_raw_data_exists(self, raw_data_target_dir):
        """
        For testing we want to process realistic raw data (e.g. big png files).
        However, we do not want to include this big data in the repository.

        Solution: We include relatively small jpg templates and "inflate" them when needed.
        """

        # raw_data_target_dir = TEST_LOT_01
        os.makedirs(raw_data_target_dir, exist_ok=True)
        raw_data_target_files = util.get_png_or_jpg_list(raw_data_target_dir)

        # TODO: extract the relative lot-part path more elegantly by reusing respective code
        # from Stage1Preprocessor.get_data_base_dir
        # for now we assume certain structure like ".../lots//2024-09-17/part000"

        relevant_parts = raw_data_target_dir.split(os.path.sep)[-2:]

        template_pattern = pjoin(TEST_DATA_DIR, "_jpg_templates", *relevant_parts, "*.jpg")
        jpg_files = glob.glob(template_pattern)
        if len(raw_data_target_files) < len(jpg_files):
            # create png files from jpg files (needs to be run only once)
            for jpg_fpath in jpg_files:
                cmd = f"mogrify -monitor -format png -resize 3000 -path {raw_data_target_dir} {jpg_fpath}"
                # print(cmd)
                os.system(cmd)

        return raw_data_target_dir

    def test_i000__sqlite_db(self):
        from chimcla import stage_2a_bar_selection as bs

        db = bs.db

        # keys: CCI-fnames like "2023-06-26_23-20-31_C0.jpg"
        # values:  list of critical cell-imgs like ['2023-06-26_23-20-31_C0_a16.jpg', '2023-06-26_23-20-31_C0_a26.jpg', ...]

        # if this fails the file data/file-info.sqlite is missing
        self.assertIn("cell_mappings", db)
        cmp = db["cell_mappings"]

    def test_i010__preprocessing(self):
        from chimcla import stage_1a_preprocessing as s1a

        raw_data_dir = self.ensure_raw_data_exists(raw_data_target_dir=TEST_LOT_01)

        args = Container(img_dir=raw_data_dir, prefix=TEST_PREFIX, no_parallel=self.no_parallel)

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

    def test_i011__preprocessing(self):
        from chimcla import stage_1a_preprocessing as s1a

        raw_data_dir = self.ensure_raw_data_exists(raw_data_target_dir=TEST_LOT_02)
        args = Container(img_dir=raw_data_dir, prefix=TEST_PREFIX, no_parallel=self.no_parallel)

        # be sure that no leftover from last test is lurking around
        self.assertEqual(len(self.get_tmp_data_dirs()), 0)

        # get the preprocessor object
        ppo = s1a.main(args)

        jpg0_files = glob.glob(pjoin(ppo.jpg0_target_dir_path, "*.jpg"))
        self.assertEqual(len(jpg0_files), 5)
        err_report_dict = ppo.get_error_report()
        self.assertEqual(len(err_report_dict[None]), 4)
        self.assertEqual(len(err_report_dict["empty slot probable via correlation"]), 1)

        shading_corrected_files = glob.glob(pjoin(ppo.shading_corrected_target_dir_path, "*.jpg"))
        self.assertEqual(len(shading_corrected_files), 4)

    def test_i020__bboxes(self):
        from chimcla import stage_2a_bar_selection as bs

        tmp_path = pjoin(TEST_DATA_DIR, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        set_trace()
        ccia = bs.CavityCarrierImageAnalyzer(tmp_path, bboxes=True)

    def test_i025__bboxes_new(self):
        from chimcla import stage_2a1_bar_selection_new as bs

        tmp_path = pjoin(TEST_DATA_DIR, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        # if this passes 81 rows could be found
        ccia = bs.CavityCarrierImageAnalyzer(tmp_path, bboxes=True)

    def test_i030__symloghist(self):
        from chimcla import stage_2a_bar_selection as bs

        tmp_path = pjoin(TEST_DATA_DIR, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        bs.get_symlog_hist(tmp_path, *"a 20".split(), dc=None)

    def test_i040__find_critical(self):
        from chimcla import stage_2a_bar_selection as bs

        tmp_path = pjoin(TEST_DATA_DIR, "stage1_completed", "2023-06-26_06-19-25_C50.jpg")
        he = bs.HistEvaluation(img_fpath=tmp_path)
        he.initialize_hist_cache()
        he.find_critical_cells_for_img(save_options={"save_plot": False, "push_db": False})

    def test_u010__bgr_convert(self):

        jpg_wildcard_path = pjoin(
            TEST_DATA_DIR, "_jpg_templates/2024-07-08_06-03-45__2d__56.7k/part000/*.jpg"
        )
        jpg_files = glob.glob(jpg_wildcard_path)

        for jpg_file in jpg_files:
            filename = os.path.basename(jpg_file)
            shutil.copy2(jpg_file, pjoin(self.tmp_work_dir, filename))

        res = os.system("chimcla_main bgr-convert ./")

        # we just test that the program exited without error
        self.assertEqual(res, 0)

    def test_u020__split_into_lots(self):

        target_dir = "./raw_jpg"
        os.makedirs(target_dir, exist_ok=True)
        # Create 100 empty files with timestamp filenames

        self._create_time_stamp_files(target_dir=target_dir, start_time=datetime(2025, 10, 1, 6, 0, 0))
        self._create_time_stamp_files(
            target_dir=target_dir, start_time=datetime(2025, 11, 1, 12, 0, 0), N=130
        )

        cmd = "chimcla_main split-into-lots ./raw_jpg/path_list.txt 25"
        res = os.system(cmd)
        self.assertEqual(res, 0)

        expected_paths = [
            'raw_jpg/lots/2025-10-01_06-00-00__0d__100/part000',
            'raw_jpg/lots/2025-10-01_06-00-00__0d__100/part003',
            'raw_jpg/lots/2025-11-01_12-00-00__0d__130/part000',
            'raw_jpg/lots/2025-11-01_12-00-00__0d__130/part005',
        ]
        for path in expected_paths:
            self.assertTrue(os.path.isdir(path))

    def _create_time_stamp_files(self, target_dir, start_time: datetime, N=100):

        fname_list = []

        for i in range(N):
            timestamp = start_time + timedelta(seconds=i * 5)
            filename = timestamp.strftime(r"%Y-%m-%d_%H-%M-%S_C0.jpg")
            filepath = pjoin(target_dir, filename)
            fname_list.append(filename)
            # Create empty file
            open(filepath, 'a').close()

        with open(pjoin(target_dir, "path_list.txt"), "a") as fp:
            for fname in fname_list:
                fp.write(f"{fname}\n")
