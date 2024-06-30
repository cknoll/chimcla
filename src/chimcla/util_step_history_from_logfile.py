"""
This script is used to evaluate logfile

example call:
python step_history_from_logfile.py -l ~/mnt/XAI-DIA-gl/Carsten/logs/classifier-2023-07-10_since_2023-06-26.log

"""

import os
import re
from datetime import datetime as dtm
from sortedcontainers import SortedDict
import argparse
import glob

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


# this is only for debugging and can be removed in production
from ipydex import activate_ips_on_exception, IPS
activate_ips_on_exception()



# LOG_FILE_PATH = f"{os.environ.get('HOME')}/mnt/XAI-DIA-gl/Carsten/classifier.log"

# fname = "classifier-2023-07-10_since_2023-06-26.log"
# LOG_FILE_PATH = f"{os.environ.get('HOME')}/mnt/XAI-DIA-gl/Carsten/logs/{fname}"

def load_lines(logfile_path):
    with open(logfile_path) as fp:
        lines = fp.readlines()
    return lines


line_cache = {}


def get_relevant_lines(raw_lines, regex=None, return_indices=False):


    if not regex:
        # use https://pythex.org/ to test better regexes
        regex = "^2023-0.*$"


    cache_result = line_cache.get(regex)
    if cache_result is not None:
        return cache_result

    rec = re.compile(regex)
    res = []
    relevant_idcs = []

    # only for debugging
    bad = []

    for i, line in enumerate(raw_lines):
        if rec.match(line):
            res.append(line)
            relevant_idcs.append(i)
        else:
            bad.append(line)

    line_cache[regex] = res

    if return_indices:
        return relevant_idcs
    else:
        return res

class TimeDeltaManager:
    """
    This class processes and models the timing information for the conveyor belt.

    Its main purpose is the method `get_position_time_vector`.
    """

    def __init__(self, relevant_lines):
        """
        :params relevant_lines:    sequence of string object containing relevant lines from log file

        """

        # contains entries like
        # "2023-06-26 01:34:54,885 - XAI-Server - INFO - Takt Handler activated: Value = stop (False)"
        self.relevant_lines = relevant_lines

        tuple_list = []
        for i, ts in enumerate(relevant_lines):
            tmp_tuple = (dtm.fromisoformat(ts.split(" - XAI")[0].replace(",", ".")), i)
            tuple_list.append(tmp_tuple)

        # create a SortedDict from that key-value-list
        # SortedDict always maintains strict order of keys (regardless when they are inserted)
        time_steps0 = SortedDict(tuple_list)

        self.datetime_objects = np.array(time_steps0.keys())

        # store the datetime when the of the first relevant log line
        self.step0 = self.datetime_objects[0]

        # time difference in seconds w.r.t self.step0
        self.time_deltas_to_step0 = np.array([q.total_seconds() for q in self.datetime_objects[:]-self.step0])
        assert self.time_deltas_to_step0[0] == 0  # check consistency

        # store the time difference in seconds for each log line
        self.time_deltas = np.diff(self.time_deltas_to_step0)

    def get_position_time_vector(self, end_time:str, N: int = 1400, return_abs_times=False):
        """
        :param end_time:    datetime or str like "2023-06-27 12:59:58,750"
                            (The comma might be there for historical reasons)
        :param N:           int; Number of steps of the conveyor belt
        """
        if isinstance(end_time, str):
            end_time = dtm.fromisoformat(end_time.replace(",", "."))

        # find the end_index (w.r.t. events represented by relevant lines)
        # -> index of first logged event which happened after end_time
        end_idx = np.where(end_time < self.datetime_objects)[0][0]

        first_idx = end_idx - N
        # index of the event which happened N steps before the end_index-event

        # if end_time is too short after the beginning of the log file, there are less
        # than N event logged. However, the result should always have length N.
        # -> we insert nan-values at the missing places
        if first_idx >= 0:
            start_idx = first_idx
            patch = 0
        else:
            start_idx = 0
            patch = -first_idx
            assert patch > 0

        station_time_vector0 = self.time_deltas[start_idx:end_idx]
        station_time_vector = np.concatenate(([np.nan]*patch, station_time_vector0))

        if return_abs_times:
            # +1 because index refers to deltas
            abs_times0 = list(self.datetime_objects[start_idx+1:end_idx+1])
            abs_times_str0 = [x.isoformat() for x in abs_times0]
            abs_times_str = ["NaT"]*patch + abs_times_str0
            return station_time_vector, abs_times_str

        return station_time_vector

class Container:
    pass


def get_img_filenames_from_dir(image_dir):
    assert os.path.isdir(image_dir)
    png_files = glob.glob(f"{image_dir}/*.png")
    jpg_files = glob.glob(f"{image_dir}/*.jpg")

    path_list = [*png_files, *jpg_files]
    path_list.sort()
    return _get_fpath_container_from_path_list(path_list)


def get_img_filenames_from_file(fpaths_file):

    with open(fpaths_file) as fp:
        txt = fp.read()
    path_list = txt.split("\n")
    return _get_fpath_container_from_path_list(path_list)


def _get_fpath_container_from_path_list(path_list) -> Container:
    res = Container()
    # extract timestamps from logfile
    res.time_stamps = []
    res.fpaths = []

    for fpath in path_list:
        if not fpath:
            continue

        res.fpaths.append(fpath)
        fname = os.path.splitext(os.path.split(fpath)[1])[0]  # something like '2023-06-26_06-16-09_C50'

        # convert file name into iso date time format
        # assume filename starting like 2023- etc or some prefix like S000056_2023-...
        if not fname.startswith("202"):
            idx = fname.index("_")
            fname = fname[idx+1:]
        assert fname.startswith("202")

        p0, p1, _ = fname.split("_")
        iso_str = f"{p0} {p1.replace('-', ':')}"

        res.time_stamps.append((fname, iso_str))

    return res


def get_img_filenames_from_logfile(all_lines):
    # DEBUG - Image: /home/sascha/Devel/xaidia-server/Classifier/images/2023-06-27_12-59-41_C50.png,
    regex_str = ".*DEBUG - Image: /home/sascha/Devel/xaidia-server/Classifier/images.*"
    img_lines0 = get_relevant_lines(all_lines, regex=regex_str, return_indices=False)

    res = Container()
    # extract timestamps from logfile
    res.time_stamps = []
    res.fpaths = []
    for line in img_lines0:
        fpath = line.split("Image:")[1].split(",")[0].strip()

        res.fpaths.append(fpath)
        fname = os.path.splitext(os.path.split(fpath)[1])[0]  # something like '2023-06-26_06-16-09_C50'

        # convert file name into iso date time format
        p0, p1, _ = fname.split("_")
        iso_str = f"{p0} {p1.replace('-', ':')}"

        res.time_stamps.append((fname, iso_str))

    if 0:
        # write all filenames to textfile
        dirname = "output"
        with open(os.path.join(dirname, "_fpaths.txt"), "w") as fp:
            fp.write("\n".join(res.fpaths))

    return res

def plot_histogram_of_time_deltas(time_deltas):
    """
    This function is useful for debugging.
    """
    # make a copy of input data
    time_deltas10 = time_deltas*1

    # collapse all values >10 to 10 (for better visualization)
    time_deltas10[time_deltas >= 10] = 10
    plt.hist(time_deltas10, bins=[*np.arange(0, 1, .1), *np.arange(1, 9, .1), 10.1], rwidth=0.8, log=not True)
    plt.show()



class MainManager:
    def __init__(self):
        self.parse_args()
        self.load_logfile()

    def parse_args(self):

        parser = argparse.ArgumentParser(
            prog='conveyor_belt_history',
            description='evaluate the conveyor belt history of chocolate images',
        )

        parser.add_argument('--logfile', "-l", help="the logfile to be evaluated", required=True)
        parser.add_argument("--image-dir", "-i", help="the image dir to be evaluated")
        parser.add_argument("--fpaths", "-p", help="text file containing paths")

        # his is intended to store the all relevant information in a Database to be easily accessible from other scripts
        parser.add_argument("--db-mode", "-dm", help="start in database-mode", action="store_true")

        # this modes iterates over directories given by PATTERN and processes csv files (ignoring entries based on CRIT_SCORE_LIMIT)
        parser.add_argument("--csv-mode", "-cm", help="start in csv-mode", nargs=2, metavar=("PATTERN", "CRIT_SCORE_LIMIT"))



        self.args = parser.parse_args()

    def load_logfile(self):

        # get lines of log file
        self.all_lines = load_lines(self.args.logfile)

        # regex_str = r".*Value = moving \(True\).*"
        regex_str = r".*Value = stop \(False\).*"

        # get indices of relevant lines
        relevant_idcs0 = get_relevant_lines(self.all_lines, regex=regex_str, return_indices=True)

        # get the actual lines (corresponding to these indices)
        self.relevant_lines0 = np.array(self.all_lines)[relevant_idcs0]

        # create first auxiliary manager instance
        self.tdm0 = TimeDeltaManager(self.relevant_lines0)

        # here ↑ we interpret *every* 'Value = moving (True)'-line as relevant line
        # However, some of these lines come with unrealistic little delay after each other
        # the next step is to determine the limit time. i.e. the minimal time between two events which is
        # considered realistic. This is done by looking at the histogram: the working hypothesis thereby:
        # unrealistic short intervals should occur only seldom. Also, we know that the usual interval is about 3s

        if 0:
            # histogram for decision where to set the limit value
            # (this block normally should not be executed)
            plot_histogram_of_time_deltas(self.tdm0.time_deltas)
            exit()

        # by looking at the histogram this value was chosen:
        delta_limit = 2.6

        # sort out those lines which are too short after the previous step
        self.relevant_lines1 = [self.relevant_lines0[0]]
        self.relevant_idcs1 = [relevant_idcs0[0]]  # this is to save the indices of the original log file
        dt_saved = 0

        # iterate over the time deltas and discard a log line if it comes too short after the last one
        # TODO: check if i should start at 1? (Because time_deltas refer to those lines starting with index 1)
        for i, dt in enumerate(self.tdm0.time_deltas, start=1):
            if dt_saved + dt >= delta_limit:
                self.relevant_lines1.append(self.relevant_lines0[i])
                self.relevant_idcs1.append(relevant_idcs0[i])
                dt_saved = 0
            else:
                dt_saved += dt

        self.tdm1 = TimeDeltaManager(self.relevant_lines1)

        if 0:
            plot_histogram_of_time_deltas(self.tdm1.time_deltas)
            exit()
        assert min(self.tdm1.time_deltas) > delta_limit

    def main(self):
        if self.args.csv_mode:
            self.handle_csv_mode()
        elif self.args.db_mode:
            self.create_db_with_filenames()
        else:
            self.create_position_time_images()

    def handle_csv_mode(self):
        CSV_FNAME = "_criticality_list.csv"
        pattern, crit_score_limit = self.args.csv_mode

        IPS()


    def create_db_with_filenames(self):

        c: Container = get_img_filenames_from_logfile(all_lines=self.all_lines)
        # IPS()
        msg = "It is not trivial to efficiently store a map from filename to position time vectors"
        # currently we can stick with the get_img_filenames_from_file option
        raise NotImplementedError(msg)

    def create_position_time_images(self):

        # for debugging
        # i_test = self.tdm1.get_position_time_vector("2023-06-27 12:59:58,750")

        dirname = "output"
        os.makedirs(dirname, exist_ok=True)

        # now get filenames of images of interest

        if self.args.fpaths:
            c: Container = get_img_filenames_from_file(self.args.fpaths)
        elif self.args.image_dir:
            c: Container = get_img_filenames_from_dir(self.args.image_dir)

        else:
            # TODO handle
            # those images which are present in the logfile

            c: Container = get_img_filenames_from_logfile(all_lines=self.all_lines)

        # create final results (visualization of position time vector)
        for i, (basename, ts) in enumerate(c.time_stamps):

            # quick hack to ignore first 100 boring images
            if 0 and i <= 10:
                continue

            position_time_vector, abs_times_str = self.tdm1.get_position_time_vector(ts, return_abs_times=True)
            plt.plot(position_time_vector)
            plt.title(basename)
            img_fpath = os.path.join(dirname, f"{basename}_ptv.png")
            tab_fpath = os.path.join(dirname, f"{basename}_tab.csv")


            # np.savetxt(tab_fpath, position_time_vector, delimiter=",")

            df1 = pd.DataFrame({"duration": np.round(position_time_vector, 3), "timestamp": abs_times_str})
            df1.to_csv(tab_fpath)

            plt.savefig(img_fpath)
            print(f"{img_fpath} written")
            plt.close()

            if 0 and i >= 3:
                # stop the script (useful during development)
                break


# this is executed by the cli script (see pyproject.toml)
def main():
    mm = MainManager()
    mm.main()

# obsolete but does not harm
if __name__ == "__main__":
    main()