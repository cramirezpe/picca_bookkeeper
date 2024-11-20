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
