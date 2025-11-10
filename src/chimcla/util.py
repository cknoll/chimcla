"""
Important:

This module contains general utility functions used in other chimcla modules.

- ImageInfoContainer

**TODO**: code from other (util) modules might be ported to here to improve code architecture
"""


import os
import glob
from functools import wraps

from ipydex import IPS

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



class ImageInfoContainer:
    """
    Class to track information about individual images
    """

    def __init__(self, original_fpath, data_base_dir):
        self.original_fpath = original_fpath
        self.latest_fpath = original_fpath
        self.original_dirpath, self.fname = os.path.split(original_fpath)
        self.basename, _ = os.path.splitext(self.fname)
        self.fname_jpg = f"{self.basename}.jpg"
        self.data_base_dir = data_base_dir

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

def handle_error(func):
    @wraps(func)
    def wrapper(self, iic):
        if iic.error is not None:
            return  # Skip execution if there's an error
        try:
            return func(self, iic)  # Call the original function
        except Exception as ex:
            err_msg = f"{func.__name__}: Exception ({ex})"
            iic.error = err_msg
            return
    return wrapper


def bgr_convert(dir_path):
    """
    Convert jpg files from BGR to RGB and vice versa.
    """

    # performing this import here (instead of at the top) speeds up cli usage for all other commands
    import cv2
    import sys

    fnames = os.listdir(dir_path)

    fnames.sort()

    for fname in fnames:
        if not fname.lower().endswith("jpg"):
            continue
        fpath = os.path.join(dir_path, fname)
        img  = cv2.imread(fpath)
        if img is None:
            print(f"could not read {fpath}")
            continue
        try:
            pass
            res = cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        except Exception as e:
            print(f"!! {e}")
            continue
        print(f"{fname} done")