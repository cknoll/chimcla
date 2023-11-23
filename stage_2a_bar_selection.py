# based on: http://localhost:8888/notebooks/iee-ge/XAI-DIA/image_classification/stage1/c_determine_shading_correction.ipynb


import os
import cv2
import argparse
import itertools as it
import asyncio
import copy
import collections
import re
import json
import time

import itertools as it
import random
import warnings
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy import optimize
from addict import Addict
import dill

from sqlitedict import SqliteDict

from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from skimage.io import imread

from ipydex import IPS, Container, activate_ips_on_exception

# activate_ips_on_exception()



BBOX_EXPECTED_WITH = 26
BBOX_EXPECTED_HEIGHT = 104
BBOX_TOL = 6

# this is used to decide wether outlier columns might be removed
BBOX_MIN_WITH = 23


vv = {"vmin": 0, "vmax": 255}


CELL_KEY_END = None
cell_keys = list(it.product("abc", np.array(range(1, 28), dtype=str)))[:CELL_KEY_END]


class ExtendedSqliteDict(SqliteDict):
    def put(self, main_key, sub_key, value, commit=False):
        data = self.get(main_key)
        if data is None:
            data = {}
        else:
            assert isinstance(data, dict)
        data.update({sub_key: value})
        self[main_key] = data

        if commit:
            self.commit()

    def put_container(self, main_key, cont, key_list, commit=False):
        """
        convenience function to write selected content of a container to the database
        """

        assert isinstance(cont, Container)
        for sub_key in key_list:
            value = cont.__dict__.get(sub_key)
            self.put(main_key, sub_key, value, commit)


db = ExtendedSqliteDict("file-info.sqlite")


def load_img(fpath):

    assert os.path.isfile(fpath), f"FileNotFound: {fpath}"
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

# ------------------


def vertical_detrend(img, y_start, y_end, dc=None):
    """
    Identify a linear trend in vertical direction and compensate it.

    Intended to be used for bbox detection only

    :param X1:       specify region of interest
    :param X2:       specify region of interest
    :param Y1:       specify region of interest
    :param Y2:       specify region of interest
    :param y_start:  start of trend detection
    :param y_end:    end of trend detection
    """

    H, W = img.shape

    # assume that the lightness changes in vertical direction -> detrend
    avg_trend = np.mean(img, axis=1)[y_start:y_end]
    yy = np.arange(y_start, y_end)

    coeffs = np.polyfit(yy, avg_trend, 1)
    poly = np.poly1d(coeffs)
    yy0 = np.arange(0, H)

    estimated_curve = np.clip(poly(yy0), np.min(avg_trend), np.max(avg_trend))

    avg_val = np.mean(avg_trend)

    # factor to compensate the linear trend such that the average value results
    factor = 1/estimated_curve * avg_val

    factor_bc = factor[:, np.newaxis].repeat(W, axis=1)
    multiplied_img = np.array(np.clip(img*factor_bc, 0, 255), dtype=np.uint8)

    # fill debug container
    if dc:
        assert isinstance(dc, Container)
        dc.fetch_locals()

    return multiplied_img



def get_bbox_list_robust(img, expected_number, plot=False, return_all=False, dc=None):

    bbox_list = get_bbox_list(img, plot, return_all, dc)

    if len(bbox_list) == expected_number:
        return bbox_list
    img2 = vertical_detrend(img, y_start=13, y_end=100, dc=dc)

    for thresh in CavityCarrierImageAnalyzier.THRESHOLDS:
        bbox_list = get_bbox_list(img2, plot, return_all, thresh=thresh, dc=dc)
        if len(bbox_list) == expected_number:
            return bbox_list

    msg = "could not find bbox, even with detrend and differnt threshold"
    raise MissingBoundingBoxes(msg)




def get_bbox_list(img, plot=False, return_all=False, thresh=75, dc=None):
    # notes for tresh: 70 resulted as too low, 80 as too high

    if plot:
        img2 = rgb(img*1)

    if len(img.shape) == 3 and img.shape[2] == 3:
        # we have an RGB image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Apply thresholding to binarize the image
    _, thresh_img = cv2.threshold(gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)

    inverted_thresh_img = 255 - thresh_img
    #plt.imshow(inverted_thresh)

    # Find the contours in the image
    cnts, _ = cv2.findContours(inverted_thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and draw bounding rectangles around each one

    bbox_list = []

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        if not return_all:
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
        plt.imshow(img2)
        plt.show()

    # fill debug container
    if dc:
        assert isinstance(dc, Container)
        dc.fetch_locals()

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
    """
    Iterate through a list of (extended) bounding boxes
    """

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

def handle_missing_boxes(bbox_list, fpath):
    missing_boxes = find_missing_boxes(bbox_list)

    if len(missing_boxes) > 0:
        # IPS()

        msg = f"missing bboxex for {fpath}: {missing_boxes}"
        raise NotImplementedError(msg)

    # TODO:
    # next steps:
    # - detect missing bboxes
    # - isolate bars
    # - do statistics for each C0-bar (over many sample images)
    # - for every unclassified image: compare each bar to the C0 - statistics
    # -> identify candidates for non-C0 images
    pass


img_bbox_cache = {}

def get_img_and_bbox_list(fpath):

    if res := img_bbox_cache.get(fpath):
        return res

    img = load_img(fpath)
    bbox_list = get_bbox_list(img, plot=False)

    res = (img, bbox_list)
    img_bbox_cache[fpath] = res

    return res


def get_raw_cell(fpath, hr_row, hr_col, e=0, f=0, plot=False):

    # hr_row, hr_col = "c", "8"

    img, bbox_list = get_img_and_bbox_list(fpath)

    assign_row_col(bbox_list)
    handle_missing_boxes(bbox_list, fpath)

    assert len(bbox_list) == 81

    assert hr_row in ("a", "b", "c")
    row_idx = {"a": 0, "b": 1, "c": 2}[hr_row]
    col_idx = int(hr_col) - 1

    idcs = bbox_list[row_idx*27 + col_idx]
    x, y, w, h = idcs[:4]
    part_img = img[y-e:y+h+e, x-f:x+w+f, :]

    # convert to Lightness A, B and then split to get lightness
    L, _, _ = cv2.split(cv2.cvtColor(part_img, cv2.COLOR_BGR2LAB)   )

    if plot:
        # plt.imshow(rgb(part_img))
        plt.imshow(L)

    return L



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# code for angle identification
# ######################################################################


class Attr_Array(np.ndarray):
    """
    Special layer on top of numpy arrays which accept custom attributes
    """
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

def rotate_img(img, angle, border_value=255, padding=3):

    padded_img = add_padding(img, padding)
    rotated_img0 = _rotate_img(padded_img, angle, border_value)

    res = rotated_img0[padding:-padding, padding:-padding]
    return res


def add_padding(img, padding=3):
    n1, n2 = img.shape
    p = padding
    padded_img = np.zeros((n1+2*padding, n2+2*padding))
    padded_img[p:-p, p:-p] = img

    for i in range(padding):
        # left
        padded_img[p:-p, i] = img[:, 0]
        # right
        padded_img[p:-p, n2 + padding + i] = img[:, -1]

        # up (row 0 is up)
        padded_img[i, p:-p] = img[0, :]
        # down
        padded_img[n1 + padding + i, p:-p] = img[-1, :]

    return padded_img


def _rotate_img(img, angle, border_value=255):
    height, width = img.shape[:2]

    # Calculate the rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height), borderValue=border_value)
    return rotated_image



def piecewise_linear2(x, x0, y0, k1, k2):
    # model of two linear functions meeting in x0, y0
    return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


def piecewise_linear3(x, x0, y0, x1, k1, k2, k3):
    # model of two linear functions meeting in x0, y0
    return np.piecewise(
        x,
        [x < x0, np.logical_and(x0 < x, x < x1), x1 <= x],
        [
            lambda x: k1*x + y0 - k1*x0,
            lambda x: k2*x + y0 - k2*x0,
            lambda x: k3*x + (k2*x1 + y0 -k2*x0) - k3*x1,
        ]
    )


class Fitter:
    def __init__(self, ii):
        self.ii = ii

        # reservoir for initial guesses
        # x0, y0, x1, k1, k2, k3

        res3 = [
            [20, 35, 50],
            [100, 250],
            [60, 75, 90],
            [-3, 0, 3],
            [-3, 0, 3],
            [-3, 0, 3],
        ]

        random.seed(1946)
        self.res3_p0 = list(it.product(*res3))
        random.shuffle(self.res3_p0)

    def fit_sequence2(self, seq, p0=None):
        if p0 is None:
            p0 = (40, 250, 2.5, 0)
        p , e = optimize.curve_fit(piecewise_linear2, self.ii, seq, p0=p0)
        # print(f"{e=}")
        return p


    def fit_sequence3(self, seq, p0=None):
        if p0 is None:
            p0 = (40, 250, 80, 2.5, 0, 0)
        p, err = optimize.curve_fit(piecewise_linear3, self.ii, seq, p0=p0)

        # err is full cov matrix
        perr = np.sqrt(np.diag(err))
        p = Attr_Array(p, err=err, perr=perr)

        # print(f"{e=}")
        return p


    def smart_fit_sequence3(self, seq):

        best_params = None
        best_errors = [np.inf, np.inf, np.inf]
        N = len(best_errors)
        bad_counter = 0

        for p0_guess in self.res3_p0:
            with warnings.catch_warnings():
                # see also
                # https://stackoverflow.com/questions/31301017/catch-optimizewarning-as-an-exception
                # warnings.simplefilter("error", optimize.OptimizeWarning)
                warnings.simplefilter("ignore", optimize.OptimizeWarning)

                try:
                    p = self.fit_sequence3(seq, p0=p0_guess)
                except optimize.OptimizeWarning:
                    # note: due to the ignore-policy this get not triggered anymore

                    # why this warning is triggered is unclear
                    # we use the result anyway

                    # try again with other initial values
                    # print("warning raised")
                    # continue
                    pass

            total_err = self.pw_err(piecewise_linear3, p, seq)[2]

            idx = np.searchsorted(best_errors, total_err)
            if idx >= N:
                bad_counter += 1
                if bad_counter >= 3:
                    # could not find better value
                    break

                # this initial guess was bad but we have some tries left
                continue

            # the result is quite good -> we have not yet explored sufficiently yet
            bad_counter = 0

            # print(idx, total_err)
            best_errors.insert(idx, total_err)
            worst = best_errors.pop()  # drop the worst value
            if worst == best_errors[0]:
                # we probably wont get better
                break

            if idx == 0:
                best_params = p

        return best_params


    def pw_err(self, model, params, seq):

        L = len(params)
        assert L % 2 == 0

        # extract x0, x1, etc assuming that y0 has index 1
        borders = list(params[:L // 2])
        borders.pop(1)

        # the last (right) boder is infinity (because we use `<`)
        borders.append(np.inf)

        # now compile a list of index-arrays which corresspond to the borders
        all_idcs = np.arange(self.ii.shape[0])
        L_all_idcs = self.ii.shape[0]

        idcs_list = []
        tmp_ii = self.ii*1 # working copy of independent array
        for b in borders:
            mask = tmp_ii < b
            idcs = all_idcs[mask]
            idcs_list.append(idcs)

            # remaining indices and values
            all_idcs = all_idcs[np.logical_not(mask)]
            tmp_ii = tmp_ii[np.logical_not(mask)]

        model_seq = model(self.ii, *params)
        diff = np.abs(model_seq - seq)

        section_diffs = []
        fractions = []

        for idcs in idcs_list:

            # for every section calculate the mean difference and its share of the total length
            if len(idcs) == 0:
                section_diffs.append(0)
            else:
                section_diffs.append(np.mean(diff[idcs]))
            fractions.append(len(idcs)/L_all_idcs)

        # only for debug printing
        # idcs_list2 = [idcs[[0, -1]] for idcs in idcs_list]
        # print(borders, idcs_list2)

        total_diff = np.mean(diff)

        return fractions, section_diffs, total_diff




def process_column(img, j, plot=False):
    fitter = Fitter(ii=np.arange(img.shape[0]))
    # p = fitter.fit_sequence3(img[:, j])
    p = fitter.smart_fit_sequence3(img[:, j])
    p.errors = fitter.pw_err(piecewise_linear3, p, img[:, j])
    if plot:
        color = colors[j]
        plt.plot(img[:, j], color=color, label=f"col {j}")
        plt.plot(fitter.ii, piecewise_linear3(fitter.ii, *p), "--", lw=2, color=color)

    p.scored_slopes = []  # will contain tuples
    fractions = p.errors[0]
    # assume pw3 model
    max_abs_slope = np.max(np.abs(p[3:]))
    for i, slope in enumerate(p[3:]):

        # score heuristic
        p.scored_slopes.append((
            slope,
            int(i == 1) + fractions[i]*3 + np.abs(slope)/5 + np.abs(slope) - max_abs_slope
        ))

    p.scored_slopes.sort(key = lambda tup: tup[1])
    p.estimated_slope = p.scored_slopes[-1][0]

    return p


BAR_TRESHOLD = 110

def get_test_column_idcs(img, plot=False):
    """
    Support for angle detection of a bar: find out which columns are relevant to test
    """

    n_rows, n_cols = img.shape
    dark_pixel_share = np.sum(img < BAR_TRESHOLD, axis=0)/n_rows

    jj = np.arange(n_cols)

    dark_pixel_indices = jj[dark_pixel_share > .8]

    j_first, j_last = dark_pixel_indices[[0, -1]]

    # actual test indices:
    j_left1 = np.max([j_first - 2, 0])
    j_left2 = j_left1 + 1

    # on the right side: calculate the corresponding negativ indices
    # i.e. right most col has index -1
    j_right1 = np.min([j_last + 2, n_cols - 1]) - n_cols
    j_right2 = j_right1 -1


    if plot:

        plt.plot(jj[dark_pixel_indices], dark_pixel_share[dark_pixel_indices], "o", label="chocolate bar columns")
        plt.plot(jj[[j_first, j_last]], dark_pixel_share[[j_first, j_last]], "o", ms=3, label="border columns")
        plt.plot(
            jj[[j_left1, j_left2, j_right1, j_right2]],
            dark_pixel_share[[j_left1, j_left2, j_right1, j_right2]],
            "x", color="tab:red", ms=8, label="test columns"
        )

        plt.plot(jj, dark_pixel_share, "o-", label="dark pixel share", color="tab:blue", alpha=0.3, zorder=0)

        plt.legend()


    return [j_left1, j_left2, j_right1, j_right2]


def get_angle(img, dc=None):

    # column_indices = [1, 2, -2, -1]
    column_indices = get_test_column_idcs(img)
    angles = []
    for j in column_indices:
        res = process_column(img, j)

        # res.estimated slope contains the slope of the lightness-change-rate
        # this is *not* the geometric slope of the desired line in pixel space
        # j is an integer index. sign(0) = 0 -> prevent this by adding 0.5
        a = res.estimated_slope / 3.31 * np.sign(j+.5)
        angles.append(a)

        # collect debug data
        if dc is not None:
            if not hasattr(dc, "angle_res"):
                dc.angle_res = []

            dc.angle_res.append(res)
            dc.get_angle_image = img

    angles.sort()
    # drop extreme values
    angles2 = angles[1:-1]

    # print(angles, np.var(angles2))
    if np.var(angles2) > 0.4:
        msg = f"could not determine consistent angles: {angles2}"
        raise InconsistentAngle(msg)

    return np.mean(angles2)


def correct_angle(img, y_offset=5, dc=None):
    roi = img[y_offset:-y_offset, :]
    a = get_angle(roi, dc=dc)
    res = rotate_img(img, -a)

    return res, a


# histogram creation

def gaussian_kernel(size, sigma):
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_smooth(data, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = np.convolve(data, kernel, mode='same')
    return smoothed

def symlog_transform(x, linthresh):
    eps = 1e-8  # correction for places where x is near 0
    # this prevents warnings, but changes nothing for the value (as long as linthresh > eps)


    return np.where(
        np.abs(x) > linthresh,
        np.sign(x) * np.log(np.abs(x)/linthresh + eps*(np.abs(x)/linthresh < eps)) + np.sign(x),
        x/linthresh
    )

class InconsistentAngle(ValueError):
    pass

class MissingBoundingBoxes(ValueError):
    pass


def get_symlog_hist(img_fpath, hr_row, hr_col, delta=None, return_cell=False, dc=None):
    """

    :param dc:  debug container
    """

    ccia = CavityCarrierImageAnalyzier(img_fpath)
    cell = ccia.get_corrected_cell(hr_row, hr_col)

    assert isinstance(cell, Attr_Array)
    angle = cell.angle

    hist = get_symlog_hist_from_cell(cell, delta=delta, dc=dc)

    # fill debug container
    if dc:
        assert isinstance(dc, Container)
        dc.fetch_locals()

    if return_cell:
        return hist, cell
    else:
        # this is only for compatibility with old code
        return hist


def get_symlog_hist_from_cell(cell, delta=None, dc=None):
    """
    :param delta: offset in pixels which will be ignored at each border
    """

    if delta is None or delta == 0:
        data = cell.flatten()
    else:
        assert delta > 0
        data = cell[delta:-delta, delta:-delta].flatten()


    hist = np.histogram(data, bins=np.arange(256))[0]
    hist2 = gaussian_smooth(hist, 20, sigma=5)

    sl_hist1 = symlog_transform(hist, linthresh=0.1)
    sl_hist2 = symlog_transform(hist2, linthresh=0.1)

    # fill debug container
    if dc:
        assert isinstance(dc, Container)
        dc.fetch_locals()

    return sl_hist1, sl_hist2


###############################################################


def convert_to_dict(bbox_list, img, desired_idx_pair=None):

    idcs = index_combinations()
    res = {}
    for idx_pair, bbox in zip(idcs, bbox_list):
        if desired_idx_pair is not None:
            if idx_pair != desired_idx_pair:
                continue

        assert list(idx_pair) == list(bbox[-2:]), f"{list(idx_pair)=}  {bbox[-2:]=}"

        part_img = adapt_rotation_and_margin(bbox, img, forced_angle=None)

        # without rotation:
        if 0:
            x, y, w, h = bbox[:4]
            part_img = img[y:y+h, x:x + w, :]
        res[idx_pair] = part_img

    # prevent non-match from passing silently (could happen for bad desired_idx_pair)
    assert res

    return res


def select_bar_from_file(fpath, hr_row, hr_col):

    img = load_img(fpath)
    bbox_list = get_bbox_list(img, plot=False)

    assign_row_col(bbox_list)
    handle_missing_boxes(bbox_list, fpath)

    assert hr_row in ("a", "b", "c")
    row_idx = {"a": 0, "b": 1, "c": 2}[hr_row]
    col_idx = int(hr_col) - 1
    assert 0 <= col_idx <= 26

    # TODO: introduce caching of image data (row_col_dict)
    row_col_dict = convert_to_dict(bbox_list, img, desired_idx_pair=(row_idx, col_idx))

    part_img = row_col_dict[(row_idx, col_idx)]

    return part_img


###############################################################


# this is old hough-transform based code

def get_angle_from_hough(cell):
    # Apply Canny edge detection to the grayscale image
    edges = cv2.Canny(cell, 50, 150, apertureSize=3)

    tested_angles = np.deg2rad(np.arange(-5.0, 5.0, step=0.1 ))
    h, theta, d = hough_line(edges, theta=tested_angles)

    # Retrieve the lines detected by Hough transform
    _, angles, _ = hough_line_peaks(h, theta, d)
    angle = np.rad2deg(np.mean(angles))

    return angle



def adapt_rotation_and_margin(bbox, img, forced_angle=None, plot=True):
    1/0

    x, y, w, h = bbox[:4]

    d = 5 # margin added before rotation
    e = d - 2 # margin subtracted after rotation
    part_img = img[y-d:y+h+d, x-d:x + w+d, :]

    gray = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)

    # generated by perplexity ai

    # Apply Canny edge detection to the grayscale image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    tested_angles = np.deg2rad(np.arange(-5.0, 5.0, step=0.1 ))
    h, theta, d = hough_line(edges, theta=tested_angles)

    # Retrieve the lines detected by Hough transform
    _, angles, _ = hough_line_peaks(h, theta, d)
    angle = np.rad2deg(np.mean(angles))

    if 0:

        # Find contours in the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if plot:
            pass
            # plt.imshow(edges)


        # Find the rectangle with the largest area
        max_area = 0
        max_rect = None
        max_cnt = None
        for cnt in contours:
            # this determines center, width, height and the angle
            rect = cv2.minAreaRect(cnt)
            area = rect[1][0] * rect[1][1]
            if area > max_area:
                max_area = area
                max_cnt = cnt
                max_rect = rect

        # Get the rotation angle of the rectangle
        angle = max_rect[2]

        if plot:
            cv2.drawContours(part_img, [max_cnt], 0, (0, 255, 0), 1)

            # now also draw the max_rect (which in general is rotated)
            box = cv2.boxPoints(max_rect)
            box = np.int0(box)

            # Draw the rotated rectangle on the original image
            cv2.drawContours(part_img, [box], 0, (200, 0, 0), 2)

            (xm, ym), (w, h), _ =  max_rect
            plt.imshow(part_img)
            plt.show()

    height, width = part_img.shape[:2]

    # adapt the rotation angle

    print(angle)

    if forced_angle is not None:
        angle = forced_angle
    else:
        if abs(angle) > 45:
            angle = angle - 90
        assert -5 <= angle <= 5

    # angle *= -1

    # Calculate the rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(part_img, rotation_matrix, (width, height))

    return rotated_image[e:-e, e:-e, :]


def main():


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

    # img_fpath = args.img_fpath


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


class CavityCarrierImageAnalyzier:
    BBOX_EXPECTED_WITH = 26
    BBOX_EXPECTED_HEIGHT = 104
    BBOX_EXPECTED_DX = 7  # horizontal space between boxes
    BBOX_EXPECTED_DY = 51  # vertical space between boxes

    BBOX_ROWS = 3
    BBOX_COLS = 27
    BBOX_NUMBER = BBOX_ROWS*BBOX_COLS

    # offsets for guessing ROI for
    ROI_DX = 5
    ROI_DY = 5

    THRESHOLDS = [75, 70, 80, 65, 85]

    def __init__(self, img_fpath, bboxes=True):
        self.img_fpath = img_fpath
        self.img = load_img(img_fpath)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_lght = cv2.split(cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB))[0]

        self.corners_dict = None

        # this will be a dict for handling situations, where finding
        # bboxes is difficult
        self.bbox_cache = None

        if bboxes:
            self.detrend_upper_row()
            self.make_sorted_bbox_list()

        if aa.available:
            self.angle_offset = aa.get_angle_offset_for_img(self)
        else:
            self.angle_offset = None

    def show(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.imshow(self.img, **vv)

    def detrend_upper_row(self):
        """
        Identify a linear trend in the upper rown and compensate it.

        Intended to be used for bbox detection only
        """

        X1 = 20
        X2 = self.img_lght.shape[1] - 20
        Y1 = 0
        Y2 = 150

        img_roi = self.img_lght[Y1:Y2, X1:X2]
        dc = Container()
        scaled_img_roi = vertical_detrend(img_roi, y_start=15, y_end=100, dc=dc)

        self.img_lght2 = self.img_lght*1
        self.img_lght2[Y1:Y2, X1:X2] = scaled_img_roi

        if 0:
            plt.plot(dc.yy, dc.avg_trend)
            plt.plot(dc.yy0, dc.estimated_curve)

            plt.figure()
            plt.imshow(self.img_lght2)
            plt.show()


    def make_sorted_bbox_list(self, plot=False):

        # use the cached version if possible
        if bbox_list := img_bbox_cache.get(self.img_fpath):
            self.bbox_list = bbox_list
            return

        last_excption = None

        self.bbox_cache = []

        for thresh in self.THRESHOLDS:
            self.bbox_list = get_bbox_list(self.img_lght2, plot=plot, thresh=thresh)
            assign_row_col(self.bbox_list)
            missing_boxes=find_missing_boxes(self.bbox_list)
            c = Container(thresh=thresh, bbox_list=self.bbox_list, missing_boxes=missing_boxes)
            self.bbox_cache.append(c)

            if not missing_boxes:
                img_bbox_cache[self.img_fpath] = self.bbox_list
                return

        # we have tried all meaningful threshold values but for none we found all boxes
        self.handle_missing_boxes()

    def handle_missing_boxes(self):

        # find the container with the least missing):
        self.bbox_cache.sort(key=lambda c: len(c.missing_boxes))

        c1 = self.bbox_cache[0]

        if len(c1.bbox_list) == 0:
            msg = f"For {self.img_fpath}: could not find any bounding box"
            raise MissingBoundingBoxes(msg)

        # at least one bbox exists

        # make bboxes easier accessible
        c1.store = {}
        for bbox_array in c1.bbox_list:
            key = tuple(bbox_array[-2:])
            c1.store[key] = bbox_array[:-2]


        while len(c1.store) < self.BBOX_NUMBER:
            # find a bbox which has missing direct neighbour
            row, col_border, col_missing = self.find_bbox_border(c1)

            hr_row = "abc"[row]
            hr_col = str(col_missing + 1)

            bbox_known = c1.store[(row, col_border)]

            # get the image and its bbox
            scaled_img_roi, roi_bbox = self.get_guessed_bbox_roi(bbox_known, col_border, col_missing)

            for thresh in self.THRESHOLDS:
                local_bbox_list = get_bbox_list(scaled_img_roi, thresh=thresh)
                if len(local_bbox_list) == 1:
                    break
            else:
                msg = f"{self.img_fpath} could not find a bbox for cell {hr_row}{hr_col} even in local region"
                raise MissingBoundingBoxes(msg)

            X, Y = roi_bbox[:2]  # corner coordinates of the partial image
            x, y, w, h = local_bbox_list[0][:4]

            new_bbox = np.r_[x + X, + y + Y, w, h]
            c1.store[(row, col_missing)] = new_bbox

            if 0:
                print(f"added cell {hr_row}{hr_col}")
                cv2.rectangle(scaled_img_roi,(x,y),(x + w,y + h),(0, 255, 0), 2)
                plt.imshow(scaled_img_roi)
                plt.show()

        items = list(c1.store.items())
        items.sort()

        self.bbox_list = []
        for (row, col), bbox in items:
            self.bbox_list.append(np.r_[bbox, row, col])

    def get_guessed_bbox_roi(self, bbox_known, col_known, col_missing):

        x, y, w, h = bbox_known

        s = np.sign(col_missing - col_known)  # positive for [known] left of [missing]

        x1 = x + s*(w + self.BBOX_EXPECTED_DX)
        roi = np.r_[x1-self.ROI_DX, y-self.ROI_DY, w + 2*self.ROI_DX, h + 2*self.ROI_DY]

        X, Y, W, H = roi

        # this occurred for some pictures
        assert Y > 0

        img_roi = self.img_lght[Y:Y+H, X:X+W]

        # assume that the lightness changes in vertical direction -> detrend
        y_start, y_end = 13, 100
        avg_trend = np.mean(img_roi, axis=1)[y_start:y_end]
        yy = np.arange(y_start, y_end)

        coeffs = np.polyfit(yy, avg_trend, 1)
        poly = np.poly1d(coeffs)
        yy0 = np.arange(0, H)

        avg_val = np.mean(avg_trend)


        # TODO: see detrend_upper_row for a better implementation
        # TODO: use with vertical_detrend (or remove this code here)

        # factor to compensate the linear trend such that the average value results
        factor = 1/poly(yy0) * avg_val

        factor_bc = factor[:, np.newaxis].repeat(W, axis=1)
        scaled_img_roi = np.array(np.clip(img_roi*factor_bc, 0, 255), dtype=np.uint8)

        if 0:
            plt.plot(yy, avg_trend)
            plt.plot(yy0, poly(yy0))
            plt.show()


        return scaled_img_roi, roi


    def find_bbox_border(self, bb_container):

        for (row, col), bbox in bb_container.store.items():

            if 0 < col < 26:
                possible_neigbours = [col -1, col + 1]
            elif col == 0:
                possible_neigbours = [col + 1]
            elif col == 26:
                possible_neigbours = [col - 1]
            else:
                msg = f"Unexpected column value: {col}"
                raise ValueError(msg)

            for col_test in possible_neigbours:
                if (row, col_test) not in bb_container.store:
                    return row, col, col_test


    def handle_missing_box(self, bb_container, row, col):
        msg = f"For {self.img_fpath}: could not find all bounding boxes"
        raise MissingBoundingBoxes(msg)

    def get_bbox_for_cell(self, hr_row, hr_col):
        # hr_row, hr_col = "c", "8"
        assert len(self.bbox_list) == 81

        assert hr_row in ("a", "b", "c")
        row_idx = {"a": 0, "b": 1, "c": 2}[hr_row]
        col_idx = int(hr_col) - 1

        bbox = self.bbox_list[row_idx*27 + col_idx]

        return bbox

    def get_raw_cell(self, hr_row, hr_col, e=0, f=0, rgb=False, plot=False):
        bbox = self.get_bbox_for_cell(hr_row, hr_col)
        x, y, w, h = bbox[:4]
        part_img = self.img[y-e:y+h+e, x-f:x+w+f, :]

        if rgb:
            return part_img

        # convert to Lightness A, B and then split to get lightness
        L, _, _ = cv2.split(cv2.cvtColor(part_img, cv2.COLOR_BGR2LAB))

        if plot:
            # plt.imshow(rgb(part_img))
            plt.imshow(L)

        return L

    def find_cell_corners(self, hr_row, hr_col, plot=False, dc=None):
        """
        return absolute coordinates of the upper left corner of a cell
        """
        bbox = self.get_bbox_for_cell(hr_row, hr_col)

        # additional margins (rows and cols)
        e = 3
        f = 3

        cell_img = self.get_raw_cell(hr_row, hr_col, e, f)

        bb_col, bb_row, bb_delta_col, bb_delta_row = bbox[:4]

        # get left and right col in the upper and lower part separately

        left1, right1 = get_border_columns(cell_img[:bb_delta_row//3, :])
        left2, right2 = get_border_columns(cell_img[-bb_delta_row//3:, :])
        up, down = get_border_columns(cell_img.T)

        res = Container()
        res.upper_left = (left1 + bb_col - e, up + bb_row - f)
        res.upper_right = (right1 + bb_col - e, up + bb_row - f)

        res.lower_left = (left2 + bb_col - e, down + bb_row - f)
        res.lower_right = (right2 + bb_col - e, down + bb_row - f)

        if plot:
            # plt.imshow(self.img[:200, :200])
            a = .99
            plt.plot([res.upper_left[0]], [res.upper_left[1]], ".", color=colors[4], alpha=a)
            plt.plot([res.lower_left[0]], [res.lower_left[1]], ".", color=colors[1], alpha=a)
            plt.plot([res.lower_right[0]], [res.lower_right[1]], ".", color=colors[2], alpha=a)
            plt.plot([res.upper_right[0]], [res.upper_right[1]], ".", color=colors[3], alpha=a)

        # fill debug container
        if dc:
            assert isinstance(dc, Container)
            dc.fetch_locals()

        return res

    def fill_corners_dict(self, plot=None, dc=None):

        if self.corners_dict:
            return self.corners_dict

        self.corners_dict = {}
        for row, col in cell_keys:

            corner_res = self.find_cell_corners(row, col, plot=plot, dc=dc)
            key = (row, str(col))
            self.corners_dict[key] = corner_res

        return self.corners_dict

    def get_horizontal_line(self, hr_row, corner_name="upper_left", plot=False):
        self.fill_corners_dict()
        assert hr_row in "abc"

        res = Container()
        points = []

        for row, col in cell_keys:
            if row != hr_row:
                continue

            corner_container = self.corners_dict[(row, col)]
            points.append(getattr(corner_container, corner_name))

        res.points = np.array(points)
        res.coeffs = np.polyfit(*res.points.T, 2)
        res.poly = np.poly1d(res.coeffs)
        res.slope = res.poly.deriv()

        if plot:
            xx = np.linspace(0, 1000, 500)
            line, = plt.plot(*res.points.T, ".")
            plt.plot(xx, res.poly(xx), color=line.get_color())

        return res

    def get_bbox_based_angle(self, hr_row, hr_col):
        bbox = self.get_bbox_for_cell(hr_row, hr_col)
        line_container1 = self.get_horizontal_line(hr_row, corner_name="upper_left")
        line_container2 = self.get_horizontal_line(hr_row, corner_name="lower_left")

        x_eval = bbox[0] + self.BBOX_EXPECTED_WITH/2

        avg_slope = (line_container1.slope(x_eval) + line_container2.slope(x_eval))/2

        angle = np.arctan(avg_slope) * 180 / np.pi
        return angle

    def get_angle_from_cell(self, hr_row, hr_col, e=3, f=3, dc=None):
        cell = self.get_raw_cell(hr_row, hr_col, e, f)

        # manually determined from artificial images (for 10px vertical cutoff)
        correction = 0.67
        angle = get_angle_from_moments(cell[10:-10, :])*correction

        return angle

    def estimate_angle_for_cell(self, hr_row, hr_col, e=3, f=3, dc=None):
        """
        Use angle analyser if possible. Evaluate image otherwise
        """

        if aa.available:
            scaling_correction = 0.8  # drastically improves the result
            # possible reason: compensate for lightness trends

            return self.angle_offset + aa.fitted_angles[(hr_row, hr_col)] * scaling_correction
        else:
            return self.get_angle_from_cell(hr_row, hr_col, e=e, f=f, dc=dc)


    def get_corrected_cell(self, hr_row, hr_col, e=3, f=3, cut_to_bb=True, plot=False, force_angle=None, dc=None):

        cell = self.get_raw_cell(hr_row, hr_col, e, f)

        if force_angle is None:
            angle = self.estimate_angle_for_cell(hr_row, hr_col, e=e, f=f, dc=dc)
        else:
            angle = force_angle
        new_cell = rotate_img(cell, -angle)

        # this operation changed the data type -> convert back to uint
        new_cell2 = np.array(new_cell, dtype=np.uint8)

        if cut_to_bb:
            # after rotation the bounding box might have changed:
            bbox_list = get_bbox_list_robust(new_cell2, expected_number=1, plot=plot)

            x, y, w, h = bbox_list[0][:4]
            new_cell2 = new_cell2[y:y+h, x:x+w]

            for i in range(1):
                if new_cell2.shape[1] <=  BBOX_MIN_WITH:
                    break
                # we cut off a border column if it deviates too buch from the rest (inner area)
                delta = 1
                avg = np.average(new_cell2[delta:-delta, delta:-delta])
                std = np.std(new_cell2)
                avg_left = np.average(new_cell2[:, 0])
                avg_right = np.average(new_cell2[:, -1])

                # print(f"{avg=} {std=} {avg_left=} {avg_right=}")
                if np.abs(avg_left - avg) > 2*std:
                    # cut off left column
                    new_cell2 = new_cell2[:, 1:]
                elif np.abs(avg_right - avg) > 2*std:
                    # cut off right column
                    new_cell2 = new_cell2[:, :-1]
                else:
                    break

        # fill debug container
        if dc:
            assert isinstance(dc, Container)

            # this is for historical reasons
            dc.fetch_locals()

        new_cell3 = Attr_Array(new_cell2)
        new_cell3.angle = angle

        return new_cell3


class AngleAnalyzer:
    def __init__(self):

        self.pp_data = Addict()
        self.polys = Addict()
        self.avg = Addict()
        self.ii = np.arange(27)
        for i, k in enumerate("abc"):
            self.pp_data[k] = []

        self.hist_dict_list = None
        self.available = False
        self.fitted_angles = Addict()

        self.load_data()

    def load_data(self):
        # this assumes that stage_02b has already been done

        # TODO: extract the relevant data and store in one separate file
        hist_dict_path = "dicts_stable"
        self.hist_dict_list = glob.glob(f"{hist_dict_path}/hist_*.dill")
        self.hist_dict_list.sort()

        if len(self.hist_dict_list) > 0:
            self.available = True

        self.process_all_dicts()
        self.fit_curves()


    def process_dict(self, hist_dict, alpha=0.5, plot=False):
        phi = Addict()

        for i, k in enumerate("abc"):
            phi[k] = []
            for j in range(27):
                phi[k].append(hist_dict["angles"][(k, str(j+1))])
            self.pp_data[k].append(np.array(phi[k]))
            if plot:
                plt.plot(phi[k], color=colors[i], alpha=alpha)

    def process_all_dicts(self, alpha=0.5, plot=False):

        for hist_dict_path in self.hist_dict_list:
            with open(hist_dict_path, "rb") as fp:
                hist_dict = dill.load(fp)
                self.process_dict(hist_dict, alpha=alpha, plot=plot)

    def fit_curves(self, plot=False):
        for i, k in enumerate("abc"):
            self.avg[k] = np.average(self.pp_data[k], axis=0)
            coeffs = np.polyfit(self.ii, self.avg[k], 6)
            poly = np.poly1d(coeffs)
            self.polys[k] = poly
            values = poly(self.ii)

            N = 27
            for i in range(N):
                self.fitted_angles[(k, str(i + 1))] = values[i]

            if plot:
                plt.plot(self.avg[k], "--", color=colors[i])
                plt.plot(self.ii, values, color=colors[i])

    def get_angle_offset_for_img(self, ccia: CavityCarrierImageAnalyzier):

        estimated_angles = []
        expected_angles = []
        for key in ["b1", "b9", "b18", "b27"]:
            tup = key[0], key[1:]
            angle = ccia.get_angle_from_cell(*tup)
            estimated_angles.append(angle)
            expected_angles.append(self.fitted_angles[tup])

        diff = np.array(estimated_angles) - np.array(expected_angles)
        return np.average(diff)


    def get_angle_offset_for_img2(self, img_angles):

        offset_list = []

        for i, k in enumerate("abc"):
            # iterate over angles for every hist
            self.diff0_data[k] = []
            row_angles = img_angles[k]
            diff = row_angles - self.avg[k]
            angle_offset = np.average(diff)
            offset_list.append(angle_offset)
        return np.average(offset_list)

    # only for development and debugging
    def check_offset(self):
        self.diff0_data = Addict()
        for i, k in enumerate("abc"):
            # iterate over angles for every hist
            self.diff0_data[k] = []
            for angles in self.pp_data[k]:
                diff = angles - self.avg[k]
                diff0 = np.average(diff)
                plt.plot(self.ii, diff - diff0, color=colors[i], alpha=0.1)
                self.diff0_data[k].append(diff0)

        plt.figure()

        for i, k in enumerate("abc"):
            plt.plot(self.diff0_data[k], "-o")

        plt.title("show correlation 1")

        plt.figure()
        plt.plot(self.diff0_data.a, self.diff0_data.b, ".")
        plt.plot(self.diff0_data.a, self.diff0_data.c, ".")
        plt.title("show correlation 2")


# create a global Analyzer object
aa = AngleAnalyzer()


def get_angle_from_moments(img):

    # this is based on http://raphael.candelier.fr/?blog=Image%20Moments
    # which cites: http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf
    # (Image Moments-based Structuring and Tracking of Objects by L. Rocha, L. Velho and P.C.P. Calvalho (2002))
    # and claims that they have typos ("6" instead of "8") in their formulas for l and w (not relevant here)

    M = Addict(cv2.moments(img))

    X = M.m10 / M.m00
    Y = M.m01 / M.m00

    MU20 = M.m20 / M.m00 - X**2
    MU11 = M.m11 / M.m00 - X*Y
    MU02 = M.m02 / M.m00 - Y**2

    theta = 0.5 * np.arctan(2*MU11 / (MU20 - MU02)) * 180/np.pi
    return theta


def get_border_columns(cell_img, dark_value_tresh=100, dark_share_tresh=0.7, dc=None):
    """
    For a given cell image return the indices of columns, that are the border of the chocolate bar
    """
    n_rows, n_cols = cell_img.shape
    dark_pixel_share = np.sum(cell_img < dark_value_tresh, axis=0)/n_rows

    jj = np.arange(n_cols)

    dark_pixel_indices = jj[dark_pixel_share > dark_share_tresh]

    j_first, j_last = dark_pixel_indices[[0, -1]]

    # fill debug container
    if dc:
        assert isinstance(dc, Container)
        dc.fetch_locals()

    return j_first, j_last


pfname = collections.namedtuple(typename="pfname", field_names="dir date time klass cell ext".split())
def analyze_img_fpath(fpath):

    path, fname = os.path.split(fpath)
    assert "#" not in fname
    fname2 = fname.replace("_", "#") # because # is contained in \w
    regex = re.compile("^([\d-]*?)#([\d-]*?)#(\w+)#??(\w*?)\.(.*?)$")
    re_res = regex.match(fname2)
    return pfname(path, *re_res.groups())


def get_cell_key_from_fpath(fpath):
    pfn = analyze_img_fpath(fpath)

    cell_key = (pfn.cell[0], pfn.cell[1:])
    return cell_key


def get_img_list(img_dir):

    img_path_list = glob.glob(f"{img_dir}/*.jpg")
    img_path_list.sort()

    # omit C100 images

    img_path_list2 = []
    C100_list = []
    for img_fpath in img_path_list:
        # find out if C100 with same base name is in list
        first_parts = img_fpath.split("_")[:-1]
        checkpath = f"{'_'.join(first_parts)}_C100.jpg"
        if checkpath in img_path_list:
            C100_list.append(img_fpath)
        else:
            img_path_list2.append(img_fpath)

    return img_path_list2


def get_original_image_fpath(img_fpath, cropped=True, resized=True) -> str:

    if not cropped or not resized:
        raise NotImplementedError()

    import pathlib

    p = pathlib.Path(img_fpath)
    fname = p.parts[-1]
    direct_parent = p.parts[-2]
    new_direct_parent  = direct_parent.replace("_shading_corrected", "")
    new_dir = p.parents[1]/new_direct_parent
    assert new_dir.is_dir()
    u = new_dir/fname
    assert u.is_file()

    return u.as_posix()


# ####################################################################################
# histogram evaluation
# ####################################################################################



def get_hist_for_cell_pict(fpath):
    pfn = analyze_img_fpath(fpath)

    the_dict_path = f"{HistEvaluation.hist_dict_path}/hist_{pfn.date}_{pfn.time}_{pfn.klass}.dill"

    with open(the_dict_path, "rb") as fp:
        hist_dict = dill.load(fp)

    cell_key = (pfn.cell[0], pfn.cell[1:])
    return hist_dict[cell_key][0]



class HistEvaluation:

    """
    HistogramEvaluation for one complete (81-cell) image
    """

    # img_dir = "/home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk000_stage1_completed/C0"
    hist_dict_path = "dicts"
    total_res_fpath = f"{hist_dict_path}/_total_res.dill"

    # limit for which criticality score (cs) a histogram is considered an anomaly
    CS_LIMIT = 20

    def __init__(self, img_fpath: str, suffix="", ev_crit_pix=False, training_data_flag=False):
        """
        :param ev_crit_pix:         bool; default False; evaluate critical pixels
                                    if true additional information about the critical pixels is collected and stored
        :param training_data_flag:  bool; save plots also for non-critical cells
                                    + store plots for raw cell images
        """

        self.img_fpath = img_fpath
        self.ev_crit_pix = ev_crit_pix
        self.training_data_flag=training_data_flag
        self.corrected_cell_cache = {}
        self.critical_hist_dir = self._get_result_dir_from_suffix(suffix)
        self.hist_cache = self.initialize_hist_cache()

        self.img_dir, self.img_fname = os.path.split(img_fpath)
        self.img_basename, self.img_ext = os.path.splitext(self.img_fname)

        self.corrected_cell_dir = os.path.join(self.critical_hist_dir, "_corrected_cells")

        with open(self.total_res_fpath, "rb") as fp:
            self.total_res = dill.load(fp)
        self.hist_dict_list = glob.glob(f"{self.hist_dict_path}/hist_*.dill")
        self.hist_dict_list.sort()

        # this is used for false positive correction
        self.total_res_adapted = copy.deepcopy(self.total_res)

        # helpful for debugging
        self.current_cell_key = None

    def initialize_hist_cache(self):
        # use the debug-container mechanism to extract the angle from the function
        # without changing the interface
        dc = Container()

        self.hist_cache = collections.defaultdict(list)
        self.hist_cache["bad_cells"] = collections.defaultdict(list)

        # this will map the cell tup to the identified angle
        self.hist_cache["angles"] = {}

        for cell_key in cell_keys:
            try:
                (hist_raw, hist_smooth), cell = get_symlog_hist(
                    self.img_fpath, *cell_key, delta=1, dc=dc, return_cell=True
                )
            except Exception as ex:
                self.hist_cache["bad_cells"][self.img_fpath].append(cell_key)
                print(f"{type(ex)}: bad cell {self.img_fpath.split('/')[-1]}: {cell_key}")
                hist_smooth = None
                dc.angle = None
                cell = None
            self.hist_cache[cell_key].append(hist_smooth)
            self.hist_cache["angles"][cell_key] = dc.angle
            self.corrected_cell_cache[cell_key] = cell

        return self.hist_cache

    @staticmethod
    def get_result_files(result_dir):
        BAD_IMGS = os.path.join(result_dir, "_bad_imgs.txt")
        GOOD_IMGS = os.path.join(result_dir, "_good_imgs.txt")
        ERR_IMGS = os.path.join(result_dir, "_err_imgs.txt")

        return BAD_IMGS, GOOD_IMGS, ERR_IMGS

    @staticmethod
    def _get_result_dir_from_suffix(suffix: str) -> str:
        return f"critical_hist{suffix}"

    @staticmethod
    def reset_result_files(suffix: str):
        result_dir =HistEvaluation._get_result_dir_from_suffix(suffix)
        fpaths = HistEvaluation.get_result_files(result_dir)
        for fpath in fpaths:
            if os.path.isfile(fpath):
                os.unlink(fpath)

    def save_eval_res(self, img_fpath, crit_cell_list, err_list):
        os.makedirs(self.critical_hist_dir, exist_ok=True)
        fname = os.path.split(img_fpath)[1]

        BAD_IMGS, GOOD_IMGS, ERR_IMGS = HistEvaluation.get_result_files(self.critical_hist_dir)

        if err_list:
            with open(ERR_IMGS, "a") as fp:
                json.dump(err_list, fp)
                fp.write("\n")

        elif crit_cell_list:
            with open(BAD_IMGS, "a") as fp:
                json.dump({fname: crit_cell_list}, fp)
                fp.write("\n")
        else:
            with open(GOOD_IMGS, "a") as fp:
                fp.write(f"{fname}\n")
                fp.write("\n")

    def get_quantiles(self, tup):
        q = Container()

        q.lower = self.total_res[tup]["q_lower"]
        q.mid = self.total_res[tup]["q_mid"]
        q.upper = self.total_res[tup]["q_upper"]
        q.ii = np.arange(len(q.lower))

        return q

    def get_criticality_score(self, cell_hist, cell, q_lower, q_upper, ev_crit_pix=None, dc=None):
        """
        for a given histogram and lower and upper bounds, calculate a score
        which reflects how critcal a given histogram is
        """
        # estimate the index which marks the border between dark side and bright side
        midpoint = int(np.average((np.argmax(q_lower), np.argmax(q_upper))))

        idcs = np.arange(len(cell_hist))
        dark_mask = idcs <= midpoint
        bright_mask = idcs > midpoint

        mask1 = cell_hist < q_lower
        mask2 = cell_hist > q_upper

        # diff1: our curve is below the lower border
        diff1_dark = q_lower[mask1*dark_mask] - cell_hist[mask1*dark_mask]
        diff1_bright = q_lower[mask1*bright_mask] - cell_hist[mask1*bright_mask]

        # diff2: our curve is above the upper border
        diff2_dark = cell_hist[mask2*dark_mask] - q_upper[mask2*dark_mask]
        diff2_bright = cell_hist[mask2*bright_mask] - q_upper[mask2*bright_mask]

        # add up critical areas (deviations on the dark side is not so important )

        dark_discount = 0.5
        area1 = dark_discount*np.sum(diff1_dark) + np.sum(diff1_bright)
        area2 = dark_discount*np.sum(diff2_dark) + np.sum(diff2_bright)

        res = Container()
        # res.area_str = f"a1={area1:03.1f}, a2={area2:03.1f}"
        res.area_str = f"a2={area2:03.1f}"
        res.a1 = area1
        res.a2 = area2

        # it turned out that diff1 (and thus area1) is not useful
        res.score = area2
        res.score_str = f"{int(res.score):04d}"


        if ev_crit_pix or (ev_crit_pix is None and self.ev_crit_pix):
            tmp_res = self.get_critical_pixel_info(cell_hist, cell, q_upper, dc=dc)

            ## add all pixel-results
            res.__dict__.update(tmp_res.item_list())

        # fill debug container
        if dc:
            assert isinstance(dc, Container)
            dc.fetch_locals()

        return res

    def get_critical_pixel_info(self, cell_hist, cell, q_upper, dc=None):
        """

        """
        assert isinstance(cell, np.ndarray)
        res = Container()

        # find lowest lightness value (index of q_upper) for which q_upper is zero and stays zero until the end
        res.crit_lightness = np.diff((q_upper==0)).nonzero()[0][-1] + 1

        res.crit_pix_mask = cell*0
        res.crit_pix_mask[cell > res.crit_lightness] = 1
        res.crit_pix_vals = cell[cell > res.crit_lightness].flatten()

        # number of critical pixels -> area
        res.crit_pix_nbr = res.crit_pix_vals.shape[0]

        if res.crit_pix_nbr:
            res.crit_pix_mean = np.mean(res.crit_pix_vals)
            res.crit_pix_median = np.median(res.crit_pix_vals)
            res.crit_pix_q95 = np.quantile(res.crit_pix_vals, 0.95)

        if dc:
            assert isinstance(dc, Container)
            dc.fetch_locals()

        return res



    def find_critical_cells(self):
        raise NotImplementedError("Outdated due to changed interface, use git blame if necessary")
        for hist_dict_path in self.hist_dict_list:
            self.find_critical_cells_for_hist_dict_path(hist_dict_path)

    def find_critical_cells_for_hist_dict_path(self, hist_dict_path):
        raise NotImplementedError("Outdated due to changed interface, use git blame if necessary")

        img_fpath = hist_dict_path.replace("dicts/hist_", f"{self.img_dir}/").replace(".dill", ".jpg")
        path, img_fname = os.path.split(img_fpath)

        # hist_dict_path = img_fpath.replace(f"{img_dir}/", "dicts/hist_").replace(".jpg", ".dill")
        with open(hist_dict_path, "rb") as fp:
            hist_dict = dill.load(fp)

        self.find_critical_cells_for_hist_dict(hist_dict, img_fpath)

    def find_critical_cells_for_hist_dict(
            self, hist_dict, img_fpath, exclude_cell_keys=None, training_data_flag=False
        ):
        raise NotImplementedError("Outdated due to changed interface, use git blame if necessary")

    def find_critical_cells_for_img(self, exclude_cell_keys=None, save_options=None):
        """
        """

        if save_options is None:
            # default values
            save_options = {"save_plot": True, "push_db": True}

        if exclude_cell_keys is None:
            exclude_cell_keys = []
        crit_cell_list = []
        for cell_key in cell_keys:
            if cell_key in exclude_cell_keys:
                continue
            res = self.evaluate_cell(cell_key, save_options=save_options)
            if res.is_critical:
                crit_cell_list.append(cell_key + (res.criticality_score,))
        return crit_cell_list


    def evaluate_cell(self, cell_key, dc=None, save_options=None, force_plot=False, recalc_hist=False,):
        """
        returns 0 for an uncritical cell, 1 for a critical cell
        Also saves an evaluation plot for every critical cell
        """

        assert save_options is not None

        # test error handling
        # if "2023-06-26_23-23-21_C0" in img_fpath:
        #     raise NotImplementedError

        q = self.get_quantiles(cell_key)
        res = Container(is_critical=False)
        self.current_cell_key = cell_key

        if recalc_hist:
            cell_hist, cell = get_symlog_hist(self.img_fpath, *cell_key, delta=1, return_cell=True, dc=dc)[1]
        else:
            cell_hist = self.hist_cache[cell_key][0]
            cell = self.get_corrected_cell(cell_key)

        if self.training_data_flag:
            self.save_corrected_cell(cell_key)

        criticality_container = self.get_criticality_score(cell_hist, cell, q.lower, q.upper, dc=dc)
        res.criticality_score = criticality_container.score
        if self.CS_LIMIT < criticality_container.score:
            res.is_critical = True
        if res.is_critical or force_plot:
            print(self.img_fpath, cell_key, criticality_container.score)
            try:
                self.save_and_plot_critical_cell(self.img_fpath, *cell_key, cell_hist, q, criticality_container, save_options=save_options)
            except RuntimeError as err:
                print(err)

        # fill debug container
        if dc:
            assert isinstance(dc, Container)
            dc.fetch_locals()

        return res

    def get_corrected_cell(self, cell_key):
        # assume the cache has been filled by self.initialize_hist_cache()
        cell = self.corrected_cell_cache[cell_key]
        assert cell is not None
        return cell

    def save_corrected_cell(self, cell_key):
        corrected_cell = self.get_corrected_cell(cell_key)
        os.makedirs(self.corrected_cell_dir, exist_ok=True)

        fname = f"{self.img_basename}_{''.join(cell_key)}_raw{self.img_ext}"
        fpath = os.path.join(self.corrected_cell_dir, fname)

        res = cv2.imwrite(fpath, corrected_cell, [cv2.IMWRITE_JPEG_QUALITY, 98])
        assert res, f"Something went wrong during the creation of {fpath}"

    def false_positive_correction(self, false_positive_dir):
        """
        This iterates over false positives and adapts the affected historgrams such that they are not
        recognized as annomaly anymore.
        """

        false_positive_fpath_list = glob.glob(f"{false_positive_dir}/*.jpg")
        false_positive_fpath_list.sort()

        for cell_pict_fpath in false_positive_fpath_list:
            self.fp_correct_for_cell(cell_pict_fpath)

        N = len(false_positive_fpath_list)
        import time
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{time_str}: False Positive correction for {N} histograms"

        meta_key = "__meta"
        if meta_key not in self.total_res_adapted:
            self.total_res_adapted[meta_key] = [msg]
        else:
            self.total_res_adapted[meta_key].append(msg)

        time_str2 = time.strftime("%Y-%m-%d__%H-%M-%S")
        backup_fpath = self.total_res_fpath.replace(".dill", f"_backup_{time_str2}.dill")
        os.rename(self.total_res_fpath, backup_fpath)
        print(f"Backup written: {backup_fpath}")

        with open(self.total_res_fpath, "wb") as fp:
            dill.dump(self.total_res_adapted, fp)

        print(f"File written: {self.total_res_fpath}")






    def fp_correct_for_cell(self, cell_pict_fpath):
        """
        False Positive Correction
        """
        # cell_pict_fpath

        cell_key = get_cell_key_from_fpath(cell_pict_fpath)
        cell_hist = get_hist_for_cell_pict(cell_pict_fpath)

        dc = None
        fake_cell = None  # not needed here

        q = self.get_quantiles(cell_key)
        criticality_container = self.get_criticality_score(cell_hist, fake_cell, q.lower, q.upper, dc=dc)

        WEIGHT = 100
        TRIES = 2*WEIGHT

        # assume a2 (related to upper bound!) is the critical part
        assert criticality_container.a2 > self.CS_LIMIT

        joined = np.where(cell_hist > q.upper, cell_hist, q.upper)
        q.new_upper = q.upper*1  # copy array
        for i in range(TRIES):
            q.new_upper = (q.new_upper*WEIGHT + joined)/(WEIGHT + 1)
            cc2 = self.get_criticality_score(cell_hist, fake_cell, q.lower, q.new_upper, dc=dc)
            # print(f"{cc2.score=}")

            if cc2.score < self.CS_LIMIT:
                break
        else:
            msg = f"False Positive Adaption: Could not adapt curve within {TRIES} tries."
            raise ValueError(msg)

        self.total_res_adapted[cell_key]["q_upper"] = q.new_upper



    def save_and_plot_critical_cell(self, img_fpath, hr_row, hr_col, cell_hist, q, cc, save_options):
        """
        :param cc: criticality_container
        :param q:  quantile container
        """

        path, fname = os.path.split(img_fpath)
        basename, ext = os.path.splitext(fname)
        new_fname = f"{basename}_{hr_row}{hr_col}{ext}"
        new_fpath = f"{self.critical_hist_dir}/{new_fname}"

        if save_options["push_db"]:
            # anchor::db_keys
            keys = ["crit_pix_nbr", "crit_pix_mean", "crit_pix_median", "crit_pix_q95", "score_str"]
            db.put_container(new_fname, cc, keys)
            db.commit()

        if not any( (save_options.get("create_plot"), save_options.get("save_plot")) ):
            # no reason to create image
            return

        ccia = CavityCarrierImageAnalyzier(img_fpath)
        cell_key =  (hr_row, hr_col)
        cell_mono = ccia.get_raw_cell(*cell_key)
        corrected_cell = self.get_corrected_cell(cell_key)

        fig = plt.figure(figsize=(9, 10))
        gs = GridSpec(2, 5, figure=fig)
        ax0 = fig.add_subplot(gs[0, :])

        # trim border (which was increased before rotation)
        x, y, w, h = ccia.get_bbox_for_cell(*cell_key)[:4]
        new_img = ccia.img.copy()
        dx = dy = 3
        lw = 2  # linewidth
        cv2.rectangle(new_img,(x - dx - 1, y - dy - 1),(x + w + dx,y + h + dy),(255, 0, 50), lw)
        ax0.imshow(new_img)
        ax0.axis("off")

        cell_rgb = ccia.get_raw_cell(*cell_key, rgb=True)

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.imshow(cell_rgb)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.imshow(cell_mono, **vv)
        ax2.axis("off")
        ax2.set_title(str(cell_mono.shape))

        corrected_cell = ccia.get_corrected_cell(*cell_key)

        ax3 = fig.add_subplot(gs[1, 2])
        ax3.imshow(corrected_cell, **vv)
        ax3.axis("off")
        ax3.set_title(f"{corrected_cell.shape}")

        ax4 = fig.add_subplot(gs[1, 3:])
        plt.sca(ax4)  # set current axis

        plt.plot(q.ii, q.mid)
        plt.plot(q.ii, q.lower)
        plt.plot(q.ii, q.upper)
        plt.plot(q.ii, cell_hist, alpha=0.9, lw=3, ls="--")
        x_offset = 40
        y_offset = 8.7
        plt.axis([-5, 260, 0, 9.5])

        ff = {"fontfamily": "monospace"}

        plt.text(x_offset, y_offset, f"{cc.area_str}", **ff)

        plt.subplots_adjust(
            left=0.01,
            bottom=0.03,
            right=0.99,
            top=0.999,
            wspace=0,
            hspace=0.05
        )

        plt.title(f"{corrected_cell.angle:01.2f}° A={cc.score:04.2f}")

        if self.ev_crit_pix:
            # visualize information about critical pixels (see self.get_critical_pixel_info())
            plt.sca(ax4)

            # vertical line, where critical pixels begin
            plt.plot([cc.crit_lightness]*2, [0, 8], "k--")

            # more text information
            if cc.crit_pix_nbr >= 5:

                x, y = cc.crit_pix_mean, 4
                plt.plot([x]*2, [0, y], ":", color="0.6")
                plt.text(x, y, "avg", **ff)

                x, y = cc.crit_pix_median, 5
                plt.plot([x]*2, [0, y], ":", color="0.6")
                plt.text(x, y, "med", **ff)

                x, y = cc.crit_pix_q95, 6
                plt.plot([x]*2, [0, y], ":", color="0.6")
                plt.text(x, y, "qnt", **ff)


                x_offset = 122
                dy = 0.5
                plt.text(x_offset, y_offset, f"#crit-pixels = {cc.crit_pix_nbr}", **ff)
                y_offset -= dy
                plt.text(x_offset, y_offset, f"        mean = {cc.crit_pix_mean:.1f}", **ff)
                y_offset -= dy
                plt.text(x_offset, y_offset, f"      median = {cc.crit_pix_median:.1f}", **ff)
                y_offset -= dy
                plt.text(x_offset, y_offset, f"         q95 = {cc.crit_pix_q95:.1f}", **ff)

                # contour of critical pixels
                plt.sca(ax3)
                plt.contour(cc.crit_pix_mask, levels=[0.5], colors='red', linewidths=1)

        if save_options["save_plot"]:
            os.makedirs(self.critical_hist_dir, exist_ok=True)
            plt.savefig(new_fpath)
            plt.close()
            # self.create_symlink(new_fpath, cc.score)

    def create_symlink(self, existing_fpath, crit_score):

        basepath, fname = os.path.split(existing_fpath)

        limits = (25, 35, 60, 100, 200, float("inf"))

        for limit in limits:
            if crit_score < limit:
                new_dir = f"up_to_{limit:03.0f}"
                break

        dst_dir = os.path.join(basepath, new_dir)
        os.makedirs(dst_dir, exist_ok=True)
        try:
            os.symlink(os.path.join("..", fname), os.path.join(dst_dir, fname))
        except FileExistsError:
            pass


if __name__ == "__main__":

    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)

    # asyncio.run(main())
    main()