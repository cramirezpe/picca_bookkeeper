"""
This module defines type aliases using NumPy and Matplotlib for arrays and
objects commonly used throughout the picca_bookkeeper package.

These type hints standardize the typing of wavelength grids and related data
structures (e.g., wave_grid, wave_grid_rf) as well as Matplotlib Axes and
Figure objects.

By importing this file, other modules in the repository can annotate functions
and variables for improved code clarity and static type checking, facilitating
consistent development and easier maintenance across the package.
"""
from typing import Tuple, TypeVar

import matplotlib
import numpy as np

NWAVE = TypeVar("NWAVE", bound=int)
NWAVERF = TypeVar("NWAVERF", bound=int)

wave_grid = np.ndarray[Tuple[NWAVE], np.dtype[np.float_]]
wave_grid_int = np.ndarray[Tuple[NWAVE], np.dtype[np.int_]]
wave_grid_str = np.ndarray[Tuple[NWAVE], np.dtype[np.str_]]
wave_grid_bool = np.ndarray[Tuple[NWAVE], np.dtype[np.bool_]]

wave_grid_rf = np.ndarray[Tuple[NWAVERF], np.dtype[np.float_]]
wave_grid_rf_int = np.ndarray[Tuple[NWAVERF], np.dtype[np.int_]]

Axes = matplotlib.axes.Axes
Figure = matplotlib.figure.Figure
