"""Utils."""

import numpy as np

def find_bins(original_array, grid_array):

    idx = np.searchsorted(grid_array, original_array)
    np.clip(idx, 0, len(grid_array) - 1, out=idx)

    prev_index_closer = (grid_array[idx - 1] - original_array) ** 2 <= (
        grid_array[idx] - original_array
    ) ** 2
    return idx - prev_index_closer