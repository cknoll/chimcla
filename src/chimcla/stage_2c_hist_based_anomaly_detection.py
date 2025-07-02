"""
Very short wrapper for `HistEvaluation.find_critical_cells()`
"""

import os
import sys
import numpy as np
import glob
import itertools as it
import collections
import argparse

import stage_2a_bar_selection as bs


def main():

    he = bs.HistEvaluation()
    he.find_critical_cells()


if __name__ == "__main__":
    main()