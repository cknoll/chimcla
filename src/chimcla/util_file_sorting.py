import os
import sys
import argparse

import numpy as np
import addict
from tqdm import tqdm
import yaml

from ipydex import IPS, activate_ips_on_exception


def diff_to_days(diff):

    return float(np.round(diff.total_seconds() / (24*3600), 3))


def fname_to_date_str(fname):
    return "_".join(fname.split("_")[:2])

class Lot(addict.Addict):
    """
    A Lot-instance models a production lot (typically spanning several days)
    """
    def __init__(self, start_index):
        super().__init__(self)
        self.start_index = start_index
        self.end_index = None
        self.first_file = None
        self.last_file = None
        self.pause_days = None
        self.duration_days = None
        self.duration_hours = None
        self.number_of_images = None
        self.dirname = None

    def set_var_values(self, dt_objs: np.ndarray, fnamelist: list[str]):
        assert self.end_index is not None
        duration = dt_objs[self.end_index] - dt_objs[self.start_index]
        self.duration_days = diff_to_days(duration)
        self.duration_hours = int(np.round(duration.total_seconds()/3600))
        self.number_of_images = self.end_index - self.start_index + 1

        self.first_file = fnamelist[self.start_index]
        self.last_file = fnamelist[self.end_index]
        date_str = fname_to_date_str(self.first_file)

        if self.number_of_images < 1e4:
            number_of_images_str = str(self.number_of_images)
        else:
            number_of_images_str = f"{(self.number_of_images/1000):3.1f}k"
        self.dirname = f"{date_str}__{int(self.duration_days)}d__{number_of_images_str}"


def split_into_lots():
    """
    Distribute a big list of files (with time stamp names) into subdirectories
    """
    import datetime as dt
    import numpy as np
    import collections

    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
    )

    parser.add_argument(
        'pathlist',
        help="txt file containing the paths",
    )
    args = parser.parse_args()

    basedir = os.path.split(args.pathlist)[0]

    print(f"reading {args.pathlist} ...")

    with open(args.pathlist, "r") as fp:
        pathlist = fp.readlines()

    def get_fname(path):
        return os.path.split(path.strip())[1]

    pathlist.sort(key=get_fname)


    fnamelist = [get_fname(path) for path in pathlist]

    cnt = collections.Counter(fnamelist)

    dupes = []
    for i, path in enumerate(pathlist):
        if cnt[get_fname(path)] > 1:
            dupes.append(path)

    if dupes:
        msg = (
            f"Unexpectedly {len(dupes)} duplicated files have been found:\n {dupes[:10]}. "
            "Please resolve that manually."
        )
        raise ValueError(msg)

    date_strings = [fname_to_date_str(fname) for fname in fnamelist]

    date_format = r"%Y-%m-%d_%H-%M-%S"
    activate_ips_on_exception()

    dt_objs = np.array([dt.datetime.strptime(date_str, date_format) for date_str in date_strings])
    diffs = np.diff(dt_objs)

    metadata = addict.Addict()
    metadata.pauses = []

    metadata.lots = [Lot(start_index=0)]

    for i, diff in enumerate(diffs):
        if diff >=  dt.timedelta(days=1):
            metadata.lots.append(Lot(start_index=i + 1))

            metadata.lots[-1].start_index = i + 1
            metadata.lots[-1].pause_days = diff_to_days(diff)

            # index of the last file for this
            metadata.lots[-2].end_index = i
            metadata.lots[-2].set_var_values(dt_objs, fnamelist)

    # note: the last lot might not yet be finished, but we assume it anyway
    metadata.lots[-1].end_index = i + 1
    metadata.lots[-1].set_var_values(dt_objs, fnamelist)

    meta_data_fname = os.path.join(basedir, "metadata.yaml")
    with open(meta_data_fname, "w") as fp:
        yaml.safe_dump(metadata.to_dict(), fp)

    for lot in metadata.lots:

        # this is to prevent double work
        # TODO: explicitly handle incomplete lots from the last run
        dir_path = os.path.join(basedir, "lots", lot.dirname)
        if os.path.exists(dir_path):
            continue
        for i in tqdm(range(lot.start_index, lot.end_index + 1)):
            counter = i - lot.start_index

            part_dir = f"part{(counter//1000):03d}"

            if counter % 1000 == 0:
                part_dir_path = os.path.join(basedir, "lots", lot.dirname, part_dir)
                os.makedirs(part_dir_path, exist_ok=True)

            original_path = os.path.join(basedir, pathlist[i].strip())
            cmd = f"mv {original_path} {part_dir_path}"
            if os.path.exists(original_path):
                os.system(cmd)

        print(f"processed: {dir_path}")
    print("all done")
