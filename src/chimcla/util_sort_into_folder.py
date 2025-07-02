"""
Helper script to copy images into different class directories based on
a hardcoded csv file (e.g. containing the result of manual labeling).

Not yet included in cli.py.
"""

import os
import pandas as pd
import numpy as np
import glob

# assumed working dir: where the images are, and where the dir `output` is

def main():

    data_path = "output/assigned_classes.csv"

    assert os.path.isfile(data_path)


    df = pd.read_csv(data_path)

    from ipydex import IPS

    dirs = np.array(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C0"])

    for d in dirs:
        os.makedirs(d, exist_ok=True)



    processed_files = {}

    for index, row in df.iterrows():

        mask = row[1:].to_numpy(dtype=bool)
        cdirs = dirs[mask]
        fname = row[0]
        for cdir in cdirs:
            cmd = f"cp {fname} {cdir}/{fname}"
            os.system(cmd)

        processed_files[fname] = 1

        # break

    all_files = glob.glob("*.jpg")

    for fname in all_files:
        if fname in processed_files:
            continue
        else:
            cmd = f"cp {fname} C0/{fname}"
            os.system(cmd)


if __name__ == "__main__":
    main()
