import os
import unittest
import glob
import shutil

# Note: for performance reasons some imports are moved inside the tests

from ipydex import IPS, Container

# run eg with `pytest -s -k test01``

pjoin = os.path.join

dir_of_this_file = os.path.dirname(__file__)
TESTDATA = pjoin(dir_of_this_file, "testdata")

_RAW_PNG_DIR = pjoin(TESTDATA, "raw_png")


class TestCases1(unittest.TestCase):

    def setUp(self) -> None:
        self.jpg0_dir_path = pjoin(TESTDATA, "jpg0")

        return super().setUp()

    def tearDown(self):
        shutil.rmtree(self.jpg0_dir_path, ignore_errors=True)

    def get_png_dir_path(self):
        png_pattern = pjoin(_RAW_PNG_DIR, "*.png")
        png_files = glob.glob(png_pattern)
        jpg_pattern = pjoin(TESTDATA, "_jpg_templates", "*.jpg")
        jpg_files = glob.glob(jpg_pattern)
        if len(png_files) < len(jpg_files):
            # create png files from jpg files (needs to be run only once)
            for jpg_fpath in jpg_files:
                cmd = f"mogrify -monitor -format png -resize 3000 -path {_RAW_PNG_DIR} {jpg_fpath}"
                # print(cmd)
                os.system(cmd)

        return _RAW_PNG_DIR

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

        args = Container(img_dir=png_dir_path, target_rel_dir="jpg0", no_parallel=True)
        self.assertEqual(self.jpg0_dir_path, pjoin(TESTDATA, args.target_rel_dir))

        self.assertFalse(os.path.isdir(self.jpg0_dir_path))

        # get the preprocessor object
        ppo = s1a.main(args)

        jpg0_files = glob.glob(pjoin(self.jpg0_dir_path, "*.jpg"))
        self.assertEqual(len(jpg0_files), 5)

        err_report_dict = ppo.get_error_report()
        self.assertEqual(len(err_report_dict[None]), 4)
        self.assertEqual(len(err_report_dict["empty slot probable via correlation"]), 1)

    def test020__bboxes(self):
        from chimcla import stage_2a_bar_selection as bs
        tmp_path = pjoin(TESTDATA, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        ccia = bs.CavityCarrierImageAnalyzer(tmp_path, bboxes=True)

    def test030__symloghist(self):
        from chimcla import stage_2a_bar_selection as bs
        tmp_path = pjoin(TESTDATA, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        bs.get_symlog_hist(tmp_path, *"a 20".split(), dc=None)

    def test040__find_critical(self):
        from chimcla import stage_2a_bar_selection as bs
        tmp_path = pjoin(TESTDATA, "stage1_completed", "2023-06-26_06-19-25_C50.jpg")
        he = bs.HistEvaluation(img_fpath=tmp_path)
        he.initialize_hist_cache()
        he.find_critical_cells_for_img(save_options = {"save_plot": False, "push_db": False})
