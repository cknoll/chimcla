"""
Contains image related utility functions
"""
import os
import cv2

# shortcut to scale plots via `**vv``
vv = {"vmin": 0, "vmax": 255}

def load_img(fpath, rgb=False):

    assert os.path.isfile(fpath), f"FileNotFound: {fpath}"
    image1  = cv2.imread(fpath)

    if rgb:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    else:
        # use BGR, do not convert
        pass

    return image1

def rgb(img):
    # useful for imshow
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)