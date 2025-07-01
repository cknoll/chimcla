# Module Documentation

This documentation covers the chimcla package, which provides tools for image analysis and classification of cavity carrier images.

## Main Package

```{py:module} chimcla
```

```{autodoc2-docstring} chimcla
:allowtitles:
```

## Submodules

The package consists of several specialized modules:

- **util**: Core utility functions and image container classes
- **stage_1a_preprocessing**: Image preprocessing including resizing, cropping, and shading correction
- **stage_2a1_bar_selection_new**: Advanced bar selection and cavity carrier image analysis
- **stage_3d_create_experimental_data**: Tools for creating experimental datasets
- **log_time_filter**: Utilities for filtering log files by date ranges
- **util_step_history_from_logfile**: Processing timing information from conveyor belt logs
- **util_img**: Image loading and manipulation utilities

```{toctree}
:titlesonly:
:maxdepth: 1

apidocs/chimcla/chimcla.util
apidocs/chimcla/chimcla.util_step_history_from_logfile
apidocs/chimcla/chimcla.stage_2a1_bar_selection_new
apidocs/chimcla/chimcla.log_time_filter
apidocs/chimcla/chimcla.stage_1a_preprocessing
apidocs/chimcla/chimcla.util_img
apidocs/chimcla/chimcla.stage_3d_create_experimental_data
```
