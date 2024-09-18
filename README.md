# `chimcla` Chocolate Image Classification


## Installation

### Preparation

- `pip install --upgrade pip setuptools wheel`
- `pip install -r requirements.txt`

### Installation in Development Mode

- `pip install -e .`

## Important commands

**Note:** Only parts of the chimcla functionality have been ported to the "professional" command line interface (see section `[project.scripts]` in `pyproject.toml`). Other parts are only available as ordinary python scripts. See also `docs/README_old.md`.


### Split into Lots

#### Background

There are many (\>\>100K) raw images. To simplify their handling they are split into "lots". Each lot corresponds to one cycle of production (e.g. 7 days).
Each lot is subdivided into chunks of (ca.) 1000 raw images.


#### Directory Structure

```
data_images
├── __chimcla_data__.txt    → the presence of this file indicates that its parent directory
│                             is the root of all relevant chimcla-data
├── jpg1000                 → rescaled work images (jpg), result of manual preprocessing
├── raw                     → original images in png format (before saving every image)
├── raw_jpg                 → original images in jpg format (necessary to reduce transfer load)
│   ├──
│   └──
│
├── pp_result               → host to the output dirs of automated preprocessing
│   ├── ...
│   └── <lotdir>
│       ├── part000
│       │   └── shading_corrected
│       │                   ↳ main output of preprocessing
│       └── ...
│
└── png_paths.txt           → file created by preparation command
```



#### Usage

- preparation: create a list of paths:
    - manually move all images from `~/mnt/XAI-DIA-gl/Sascha/Images_from_Peine` to `~/mnt/XAI-DIA-gl/Carsten/data_images/raw_jpg` (speed: 10K/min)
    - workdir: `~/mnt/XAI-DIA-gl/Carsten/data_images/raw_jpg`
    - command: `find . -type f -name '*.jpg' > jpg_paths.txt` (takes approx. 40s for 200K image)
- usage: `chimcla_split_into_lots ~/mnt/XAI-DIA-gl/Carsten/data_images/raw_jpg/jpg_paths.txt` (takes 20m for 200K images)
- manual post processing
    - move subdirectories of `data_images/raw_jpg/lots/` into `data_images/lots` (make sure that nothing is overwritten)

### Create Work Images

- Example call: `chimcla_create_work_images ~/mnt/XAI-DIA-gl/Carsten/data_images/lots/2024-07-08_06-03-45__2d__56.7k/part000`
    - Results: within `~/mnt/XAI-DIA-gl/Carsten/data_images/pp_result/2024-07-08_06-03-45__2d__56.7k/part000` the following subdirectories are created:
    - `jpg0` → intermediate results (can be deleted)
    - `cropped` → intermediate results (can be deleted)
    - `shading_corrected` → starting point for cell based evaluation



### Step History Evaluation

- overview:
    - `chimcla_step_history_eval --help`
- usage example
    - `chimcla_step_history_eval -l ~/mnt/XAI-DIA-gl/Carsten/logs/classifier-2023-07-10_since_2023-06-26.log --csv-mode stage3_results__history_test_y\* 300`
    - explanation:
        - `-l <path_to_log_file>`
        - `--csv-mode <pattern> <critical-score-limit>`
        - `<pattern>`: A pattern for preprocessed image files like `directory/*.jpg`
        - Note that in order to pass an asterisk character (`*`) as part of an argument to the python script it has to be escaped (prepended by a backslash). Otherwise the shell (e.g. bash) tries to expand it before passing the arguments to the script.
        - `<critical-score-limit>`: Lower limit for "cumulated criticality score" for images which should be considered in the step history creation.
        The criticality score specifies how much a cell image deviates from the expectation (e.g. homogenous brown). Values below 20 are considered unproblematic, values above 100 typically show bright regions or other obvious problems. "Cumulated" means that the criticality scores of all 81 cells of the image are added.


### Create Experimental Data (ced)

- overview:
    - `chimcla_ced --help`
- usage example:
    - `chimcla_ced --img_dir /home/ck/mnt/XAI-DIA-gl/Carsten/bilder_jpg2a/cropped/chunk001_shading_corrected/ --suffix _history_test_y -H`
    - explanation:
        - `--img_dir <path>`: specify source directory
        - `--suffix <path>`: specify target directory
        - `-H`: activate history-evaluation mode
