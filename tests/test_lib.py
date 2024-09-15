import os
import unittest
from chimcla import stage_2a_bar_selection as bs
from chimcla import stage_2a_bar_selection

# run eg with `pytest -s -k test01``

pjoin = os.path.join

dir_of_this_file = os.path.dirname(__file__)
TESTDATA = pjoin(dir_of_this_file, "testdata")


class TestCases1(unittest.TestCase):

    def test00_ips(self):
        from ipydex import IPS
        db = bs.db

        # keys: CCI-fnames like "2023-06-26_23-20-31_C0.jpg"
        # values:  list of critical cell-imgs like ['2023-06-26_23-20-31_C0_a16.jpg', '2023-06-26_23-20-31_C0_a26.jpg', ...]
        # IPS()
        cmp = db["cell_mappings"]

    def test01_bboxes(self):
        # tmp_path = "/home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0/2023-06-26_06-19-25_C50.jpg"
        tmp_path = pjoin(TESTDATA, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        ccia = bs.CavityCarrierImageAnalyzer(tmp_path, bboxes=True)

    def test02_symloghist(self):

        tmp_path = pjoin(TESTDATA, "stage1_completed", "2023-06-26_06-19-58_C50.jpg")
        bs.get_symlog_hist(tmp_path, *"a 20".split(), dc=None)

    def test03_find_critical(self):
        tmp_path = pjoin(TESTDATA, "stage1_completed", "2023-06-26_06-19-25_C50.jpg")
        he = bs.HistEvaluation(img_fpath=tmp_path)
        he.initialize_hist_cache()
        he.find_critical_cells_for_img(save_options = {"save_plot": False, "push_db": False})
