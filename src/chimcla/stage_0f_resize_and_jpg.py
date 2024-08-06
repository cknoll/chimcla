# based on: http://localhost:8888/notebooks/iee-ge/XAI-DIA/image_classification/stage1/c_determine_shading_correction.ipynb


import os
import argparse
import numpy as np

import glob
import asyncio

class OmnicientManager:
    pass

# (omniscient (aka all-knowing) manager object)
omo = OmnicientManager()


def background(f):
    """
    decorator for parallelization
    """
    # source: https://stackoverflow.com/a/59385935
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def do_work(fpath):


    prefix, fname = os.path.split(fpath)
    new_path = omo.target_dir

    cmd = f"mogrify -monitor -format jpg -resize 1000 -path {new_path} {fpath}"
    # print(cmd, "\n")
    os.system(cmd)
    # print(f"{new_path:<45} written, ({dt:05.3f})s")

    # return img_new


async def mainloop():
    tasks = []
    for fpath in omo.img_path_list:
        tasks.append(
            do_work(fpath)
        )
    await asyncio.gather(*tasks)

    # this line should be executed only if the loop is finished
    print("asyncio loop finished")


def main():

    parser = argparse.ArgumentParser(
        prog='stage_0f_resize_and_jpg',
        description='This program corrects resizes the original png files and converts to jpg',
    )


    parser.add_argument('img_dir', help="e.g. /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_roh_aus_peine_ab_2023-07-31")
    parser.add_argument('target_rel_dir', help="target directory (relative to img_dir/..)", nargs="?", default="bilder_jpg0")

    args = parser.parse_args()


    img_dir = args.img_dir.rstrip("/")
    omo.target_dir = os.path.join(img_dir, "..", args.target_rel_dir)
    assert os.path.exists(img_dir)

    os.makedirs(omo.target_dir, exist_ok=True)

    omo.img_path_list = glob.glob(f"{img_dir}/*.png")



    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    asyncio.run(mainloop())
