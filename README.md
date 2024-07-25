# `chimcla` Chocolate Image Classification


## Installation

### Preparation

- `pip install --upgrade pip setuptools wheel`
- `pip install -r requirements.txt`

### Installation in Development Mode

- `pip install -e .`

## Important commands

**Note:** Only parts of the chimcla functionality have been ported to the "professional" command line interface (see section `[project.scripts]` in `pyproject.toml`). Other parts are only available as ordinary python scripts. See also `docs/README_old.md`.


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
