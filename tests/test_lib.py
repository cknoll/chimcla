import unittest
from .. import stage_2a_bar_selection as bs

# run eg with `pytest -s -k test01``


class TestCases1(unittest.TestCase):

    def test01_bboxes(self):
        # tmp_path = "/home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0/2023-06-26_06-19-25_C50.jpg"
        tmp_path = "/home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0/2023-06-26_06-19-58_C50.jpg"
        ccia = bs.CavityCarrierImageAnalyzer(tmp_path, bboxes=True)

    def test02_symloghist(self):

        tmp_path = "/home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0/2023-06-26_06-19-25_C50.jpg"
        bs.get_symlog_hist(tmp_path, *"a 20".split(), dc=None)

    def test03_find_critical(self):
        he = bs.HistEvaluation()
        he.find_critical_cells()
        pass