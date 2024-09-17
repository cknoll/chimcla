"""
This module contains general utility functions
"""


import os
import glob

pjoin = os.path.join
# assuming that the package is installed with `pip install -e .`
CHIMCLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CHIMCLA_DATA = pjoin(CHIMCLA_ROOT, "data")

def get_png_or_jpg_list(dir_path):
    png_paths = glob.glob(pjoin(dir_path, "*.png"))
    jpg_paths = glob.glob(pjoin(dir_path, "*.jpg"))

    if png_paths and not jpg_paths:
        img_path_list = png_paths
    elif not png_paths and jpg_paths:
        img_path_list = jpg_paths
    elif not png_paths and not jpg_paths:
        img_path_list = []
    else:
        msg = f"Unexpected situation: {len(png_paths)=} and {len(jpg_paths)=}"
        raise ValueError(msg)

    return img_path_list
