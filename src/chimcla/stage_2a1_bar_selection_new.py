"""
This Module is a (yet incomplete) rewrite of the historically grown module stage_2a_bar_selection.
It should eventually replace it. Currently it is only used in the tests.
"""

import os
from typing import Dict
import itertools as it

import numpy as np
import matplotlib.pyplot as plt
import cv2

from ipydex import IPS, activate_ips_on_exception, Container

from .util import ImageInfoContainer, handle_error
from .util_img import load_img, rgb, vv

pjoin = os.path.join


# TODO: this should not be a global variable
img_bbox_cache = {}

class CavityCarrierImageAnalyzer:
    BBOX_EXPECTED_WITH = 26
    BBOX_EXPECTED_HEIGHT = 104
    BBOX_TOL = 6  # tolerance for expected dimensions

    BBOX_EXPECTED_DX = 7  # horizontal space between boxes
    BBOX_EXPECTED_DY = 51  # vertical space between boxes

    BBOX_ROWS = 3
    BBOX_COLS = 27
    BBOX_NUMBER = BBOX_ROWS*BBOX_COLS

    # approx. y indices for middle row (b cells)
    # these are not the actual bounds but values which are safe inside
    # used to determine the threshold level
    MID_ROW_Y_UPPER = 190
    MID_ROW_Y_LOWER = 250

    # offsets for guessing ROI for
    ROI_DX = 5
    ROI_DY = 5

    DELTA_THRESHOLDS = [10, 5, 15, 0, -5, -10]

    def __init__(self, img_fpath, bboxes=True):
        self.img_fpath = img_fpath
        self.img = load_img(img_fpath)
        # self.img_fpath_uncorrected = get_original_image_fpath(img_fpath)
        # self.img_uncorrected = load_img(self.img_fpath_uncorrected, rgb=True)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_lght = cv2.split(cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB))[0]

        self.corners_dict = None

        # this will be a dict for handling situations, where finding
        # bboxes is difficult
        self.bbox_cache = None
        self.corrected_cell_cache = {}

        if bboxes:
            self.detrend_upper_row()
            self.make_sorted_bbox_list()

        # if aa.available:
        #     self.angle_offset = aa.get_angle_offset_for_img(self)
        # else:
        #     self.angle_offset = None

    def get_bbox_list(self, img, plot=False, return_all=False, thresh=75, dc=None) -> list:
        """
        This strongly depends on suitable threshold-value.

        """

        # plot = True  # !! ony for debugging
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
                if (
                    (abs(w - self.BBOX_EXPECTED_WITH) > self.BBOX_TOL)
                    or
                    (abs(h - self.BBOX_EXPECTED_HEIGHT) > self.BBOX_TOL)
                ):
                    # this bbox is too big or too small -> ignore it
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
            plt.figure()
            plt.imshow(inverted_thresh_img)
            plt.show()

        # fill debug container
        if dc:
            assert isinstance(dc, Container)
            dc.fetch_locals()

        return bbox_list

    def get_lightness_distribution_ido_y(self, y=None):
        if y is None:
            gray_of_y = np.median(self.img_lght2, axis=1)
        else:
            # only select a limited range
            assert isinstance(y, tuple) and len(y) == 2
            gray_of_y = np.median(self.img_lght2[y[0]:y[1], :], axis=1)
        return gray_of_y

    def show(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.imshow(self.img, **vv)

    @staticmethod
    def replace_cell(img, bbox, cell_img):

        x, y, w, h = bbox
        if isinstance(cell_img, np.ndarray):

            diff_h  = cell_img.shape[0] - h
            diff_w  = cell_img.shape[1] - w

            # diff < 0 means: cell_img is smaller → delta < 0
            # h is decreased, y is increased

            for i in range(abs(diff_h)):
                delta = np.sign(diff_h)
                h += delta
                if i % 2 == 1:
                    y -= delta

            for i in range(abs(diff_w)):
                delta = np.sign(diff_w)
                w += delta
                if i % 2 == 1:
                    x -= delta

            assert cell_img.shape[:2] == (h, w)

        img[y:y+h, x:x+w] = cell_img

    def detrend_upper_row(self):
        """
        Identify a linear trend in the upper row and compensate it.

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

        last_exception = None

        self.bbox_cache = []

        mid_row_idcs = (self.MID_ROW_Y_UPPER, self.MID_ROW_Y_LOWER)
        thresh_value0 = np.mean(self.get_lightness_distribution_ido_y(y=mid_row_idcs))

        for dt in self.DELTA_THRESHOLDS:
            thresh_value = thresh_value0 + dt
            self.bbox_list = self.get_bbox_list(self.img_lght2, plot=plot, thresh=thresh_value)
            assign_row_col(self.bbox_list)
            missing_boxes=find_missing_boxes(self.bbox_list)
            c = Container(thresh=thresh_value, bbox_list=self.bbox_list, missing_boxes=missing_boxes)
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
            # find a bbox which has missing direct neighbor
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
                possible_neighbors = [col -1, col + 1]
            elif col == 0:
                possible_neighbors = [col + 1]
            elif col == 26:
                possible_neighbors = [col - 1]
            else:
                msg = f"Unexpected column value: {col}"
                raise ValueError(msg)

            for col_test in possible_neighbors:
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

    def get_raw_cell(self, hr_row, hr_col, e=0, f=0, rgb=False, plot=False, uncorrected=False):
        bbox = self.get_bbox_for_cell(hr_row, hr_col)
        x, y, w, h = bbox[:4]

        if uncorrected:
            img = self.img_uncorrected
        else:
            img = self.img
        part_img = img[y-e:y+h+e, x-f:x+w+f, :]

        if rgb:
            return part_img

        # convert to Lightness A, B and then split to get lightness
        L, _, _ = cv2.split(cv2.cvtColor(part_img, cv2.COLOR_BGR2LAB))

        if plot:
            # plt.imshow(rgb(part_img))
            plt.imshow(L)

        L2 = Attr_Array(L)
        L2.bbox = bbox[:4]
        return L2

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
        Use angle analyzer if possible. Evaluate image otherwise
        """

        if aa.available:
            scaling_correction = 0.8  # drastically improves the result
            # possible reason: compensate for lightness trends

            return self.angle_offset + aa.fitted_angles[(hr_row, hr_col)] * scaling_correction
        else:
            return self.get_angle_from_cell(hr_row, hr_col, e=e, f=f, dc=dc)


    def get_corrected_cell(self, hr_row, hr_col, e=3, f=3, cut_to_bb=True, plot=False, force_angle=None, dc=None):

        key = (hr_row, hr_col, e, f, cut_to_bb)
        cached_res = self.corrected_cell_cache.get(key)
        if cached_res is not None:
            return cached_res

        res = self._get_corrected_cell(hr_row, hr_col, e, f, cut_to_bb, plot, force_angle, dc)
        self.corrected_cell_cache[key] = res
        return res

    def _get_corrected_cell(self, hr_row, hr_col, e=3, f=3, cut_to_bb=True, plot=False, force_angle=None, dc=None):

        raw_cell = self.get_raw_cell(hr_row, hr_col, e, f)

        if force_angle is None:
            angle = self.estimate_angle_for_cell(hr_row, hr_col, e=e, f=f, dc=dc)
        else:
            angle = force_angle
        new_cell = rotate_img(raw_cell, -angle)

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
                # we cut off a border column if it deviates too much from the rest (inner area)
                delta = 1
                avg = np.average(new_cell2[delta:-delta, delta:-delta])
                std = np.std(new_cell2)
                avg_left = np.average(new_cell2[:, 0])
                avg_right = np.average(new_cell2[:, -1])

                # print(f"{avg=} {std=} {avg_left=} {avg_right=}")
                if np.abs(avg_left - avg) > 2*std:
                    # cut off left column
                    new_cell2 = new_cell2[:, 1:]
                    x += 1
                elif np.abs(avg_right - avg) > 2*std:
                    # cut off right column
                    new_cell2 = new_cell2[:, :-1]
                    w -= 1
                else:
                    break

        # fill debug container
        if dc:
            assert isinstance(dc, Container)

            # this is for historical reasons
            dc.fetch_locals()

        new_cell3 = Attr_Array(new_cell2)
        new_cell3.angle = angle
        new_cell3.raw_cell_bbox = raw_cell.bbox
        new_cell3.corrected_bbox = x, y, w, h

        return new_cell3
# end of class CavityCarrierImageAnalyzer


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


# BBox-specific functions


def assign_row_col(bbox_list):
    """
    problem: The list of bounding boxes is not sorted. it has to be calculated in which
    row an col every bb is. Also after this function the bbox_list is sorted (starting with 1st row)
    """

    # 3 rows, 27 cols -> 81 cells
    if len(bbox_list) != 81:
        msg = f"unexpected number (not 81) of bounding boxes: {len(bbox_list)}"
        raise ValueError(msg)

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


## End of BBox-specific functions


class Stage2Preprocessor:
    def __init__(self, args):
        # general preparations
        # see cli.py for arg-definitions
        self.args = args
        self.iic_map: Dict[str, ImageInfoContainer] = {}


    def pipeline(self):
        pass
