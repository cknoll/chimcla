# based on: http://localhost:8888/notebooks/iee-ge/XAI-DIA/image_classification/stage1/c_determine_shading_correction.ipynb


import os
import cv2
import argparse
import itertools as it
import asyncio

import itertools as it
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from addict import Addict



from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from skimage.io import imread

from ipydex import IPS, Container


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


    BBOX_EXPECTED_WITH = 26
    BBOX_EXPECTED_HEIGHT = 104
    BBOX_TOL = 6

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


cell_tups = list(it.product("abc", np.array(range(1, 28), dtype=str)))


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


def get_symlog_hist(img_fpath, hr_row, hr_col, dc=None):
    """

    :param dc:  debug container
    """

    ccia = CavityCarrierImageAnalyzier(img_fpath)
    cell = ccia.get_corrected_cell(hr_row, hr_col)

    assert isinstance(cell, Attr_Array)
    angle = cell.angle

    # fill debug container
    if dc:
        assert isinstance(dc, Container)
        dc.fetch_locals()

    return get_symlog_hist_from_cell(cell, dc=dc)


def get_symlog_hist_from_cell(cell, delta=None, dc=None):
    """
    :param delta: offset in pixels which will be ignored at each border
    """

    if delta is None:
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

    def show(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.imshow(self.img, vmin=0, vmax=255)

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

    def get_raw_cell(self, hr_row, hr_col, e=0, f=0, plot=False):
        bbox = self.get_bbox_for_cell(hr_row, hr_col)
        x, y, w, h = bbox[:4]
        part_img = self.img[y-e:y+h+e, x-f:x+w+f, :]

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
        for row, col in cell_tups:

            corner_res = self.find_cell_corners(row, col, plot=plot, dc=dc)
            key = (row, str(col))
            self.corners_dict[key] = corner_res

        return self.corners_dict

    def get_horizontal_line(self, hr_row, corner_name="upper_left", plot=False):
        self.fill_corners_dict()
        assert hr_row in "abc"

        res = Container()
        points = []

        for row, col in cell_tups:
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

    def get_corrected_cell(self, hr_row, hr_col, e=3, f=3, cut_to_bb=True, plot=False, dc=None):
        # angle = self.get_bbox_based_angle(hr_row, hr_col)
        cell = self.get_raw_cell(hr_row, hr_col, e, f)

        # manually determined from artificial images (for 10px vertical cutoff)
        correction = 0.67
        angle = get_angle_from_moments(cell[10:-10, :])*correction

        new_cell = rotate_img(cell, -angle)

        # this operation changed the data type -> convert back to uint
        new_cell2 = np.array(new_cell, dtype=np.uint8)

        if cut_to_bb:
            # after rotation the bounding box might have changed:
            bbox_list = get_bbox_list_robust(new_cell2, expected_number=1, plot=plot)

            x, y, w, h = bbox_list[0][:4]
            new_cell2 = new_cell2[y:y+h, x:x+w]

        # fill debug container
        if dc:
            assert isinstance(dc, Container)
            dc.fetch_locals()

        new_cell3 = Attr_Array(new_cell2)
        new_cell3.angle = angle

        return new_cell3


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


if __name__ == "__main__":

    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)

    # asyncio.run(main())
    main()