# based on: http://localhost:8888/notebooks/iee-ge/XAI-DIA/image_classification/stage1/c_determine_shading_correction.ipynb


import os
import cv2
import argparse
import itertools as it
import asyncio

import numpy as np
import matplotlib.pyplot as plt

from ipydex import IPS



parser = argparse.ArgumentParser(
    prog='stage_1c_shading_correction',
    description='This program corrects the shading of the images',
)



parser.add_argument('img_fpath', help="e.g. /home/ck/iee-ge/XAI-DIA/image_classification/stage2/single_bars/raw/2023-06-26_06-22-47_C50.jpg")
parser.add_argument('row', help="one of a, b or c")
parser.add_argument('col', help="integer in [1, 27]")

args = parser.parse_args()


# img_dir = "shading_correction_base_data"
# img_dir = "/home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000"

img_fpath = args.img_fpath


def load_img(fpath):

    image1  = cv2.imread(fpath)

    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # use BGR, do not convert

    return image1

def rgb(img):
    # useful for imshow
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def background(f):
    """
    decorator for paralelization
    """
    # source: https://stackoverflow.com/a/59385935
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def get_bbox_list(img, plot=False):

    if plot:
        img2 = img*1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(gray, thresh=70, maxval=255, type=cv2.THRESH_BINARY)


    inverted_thresh = 255 - thresh
    #plt.imshow(inverted_thresh)

    # Find the contours in the image
    cnts, _ = cv2.findContours(inverted_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and draw bounding rectangles around each one


    BBOX_EXPECTED_WITH = 26
    BBOX_EXPECTED_HEIGHT = 104
    BBOX_TOL = 6

    bbox_list = []

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if (abs(w - BBOX_EXPECTED_WITH) > BBOX_TOL) or (abs(h - BBOX_EXPECTED_HEIGHT) > BBOX_TOL):
            continue
        else:
            # print(w, h, "--", abs(w - BBOX_EXPECTED_WITH) > BBOX_TOL, abs(h - BBOX_EXPECTED_HEIGHT) > BBOX_TOL)
            pass

        # the last two values will be assigned later (row and column index)
        bbox_list.append(np.r_[x, y, w, h, -1, -1])

        if plot:
            cv2.rectangle(img2,(x,y),(x + w,y + h),(0, 255, 0), 2)

    # bboxes = np.array(bboxes)

    if plot:
        plt.imshow(rgb(img2))
        plt.show()

    return bbox_list


def assign_row_col(bbox_list):
    """
    problem: The list of bounding boxes is not sorted. it has to be calculated in which
    row an col every bb is. Also after this function the bbox_list is sorted (starting with 1st row)
    """
    # 3 rows, 27 cols
    bbox_arr = np.array(bbox_list)

    # this was generated from a reference image
    if 0:

        xx, yy, ww, hh = bbox_arr.T
        xx.sort()
        yy.sort()

        dx_mean = np.average(np.diff(xx.reshape(-1, 3), axis=0)) ##:
        dy_mean = np.average(np.diff(yy.reshape(3, -1), axis=0)) ##:
    else:
        # average distances between boxes
        dx_mean = 30.73076923076923
        dy_mean = 153.72222222222223

    xx, yy, ww, hh, _, _ = bbox_arr.T

    xmin = np.min(xx)
    xmax = np.max(xx)

    ymin = np.min(yy)
    ymax = np.max(yy)

    xx_index_candidates = (xx - xmin)/dx_mean
    yy_index_candidates = (yy - ymin)/dy_mean

    xx_idcs = np.int16(xx_index_candidates.round())
    yy_idcs = np.int16(yy_index_candidates.round())

    # ensure that it is clear which index each bb refers to
    assert np.max(np.abs(xx_index_candidates - xx_idcs)) < 0.45
    assert np.max(np.abs(yy_index_candidates - yy_idcs)) < 0.45


    for i, (xx_idx, yy_idx) in enumerate(zip(xx_idcs, yy_idcs)):
        bbox_list[i][4] = yy_idx  # row
        bbox_list[i][5] = xx_idx  # col


    # sort such that first row comes first
    bbox_list.sort(key=lambda seq: seq[4]*1e4 + seq[5])


def index_combinations():
    return list(it.product(range(3), range(27)))


def find_missing_boxes(bbox_list):

    idcs = index_combinations()
    for bbox in bbox_list:
        row_col = tuple(bbox[-2:])

        try:
            idcs.remove(row_col)
        except ValueError:
            msg = f"unexpected row_col index pair: {row_col}"
            raise ValueError(msg)

    # in the nominal case this list is now empty
    return idcs

def handle_missing_boxes(bbox_list):
    missing_boxes = find_missing_boxes(bbox_list)

    if len(missing_boxes) > 0:
        # IPS()

        print(f"!!  problems with {args.img_fpath}")
        exit()

        # raise NotImplementedError

    # TODO:
    # next steps:
    # - detect missing bboxes
    # - isolate bars
    # - do statistics for each C0-bar (over many sample images)
    # - for every unclassified image: compare each bar to the C0 - statistics
    # -> identify candidates for non-C0 images
    pass

def convert_to_dict(bbox_list, img):

    idcs = index_combinations()
    res = {}
    for idx_pair, bbox in zip(idcs, bbox_list):
        assert list(idx_pair) == list(bbox[-2:]), f"{list(idx_pair)=}  {bbox[-2:]=}"
        x, y, w, h = bbox[:4]
        res[idx_pair] = img[y:y+h, x:x + w, :]

    return res



def select_bar_from_file(fpath, hr_row, hr_col):

    img = load_img(fpath)
    bbox_list = get_bbox_list(img, plot=False)

    assign_row_col(bbox_list)
    handle_missing_boxes(bbox_list)

    # introduce caching of image data (row_col_dict)
    row_col_dict = convert_to_dict(bbox_list, img)

    assert hr_row in ("a", "b", "c")
    row_idx = {"a": 0, "b": 1, "c": 2}[hr_row]
    col_idx = int(hr_col) - 1
    assert 0 <= col_idx <= 26

    part_img = row_col_dict[(row_idx, col_idx)]

    return part_img

def main():


    part_img = select_bar_from_file(args.img_fpath, args.row, args.col)
    plt.imshow(rgb(part_img))

    prefix, fname = os.path.split(args.img_fpath)
    fname, ext = os.path.splitext(fname)
    col = int(args.col)
    new_fname = f"{fname}__{args.row}{col:02d}{ext}"
    new_fpath = os.path.join(prefix, "..", new_fname)

    res = cv2.imwrite(new_fpath, part_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    assert res, f"Something went wrong during the creation of {new_fpath}"
    print(f"file written: {new_fpath}")

    # plt.show()




if __name__ == "__main__":

    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)

    # asyncio.run(main())
    main()