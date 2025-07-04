import warnings

import numpy as np
from scipy.ndimage import median_filter

from asltk.utils import collect_data_volumes


def isotropic_median(data, size: int = 3):
    """Smooth the data using a median filter.

    This method applies a median filter with an isotropic kernel to reduce
    noise while preserving edges. The method uses the `scipy.ndimage` library
    to apply the filtering.

    Note:
        If the data is higher than 3D dimension, then the method will apply the
        filtering to all the volumes individually and reconstruct the original
        data again.

    Important:
        The kernel size should be an odd integer. Even values will be rounded
        down to the nearest odd integer. Typical values for ASL data are 3-7,
        depending on the desired noise reduction vs. edge preservation trade-off.

    Parameters
    ----------
    data : array_like
        The data to be smoothed.
    size : int
        The size of the median filter kernel. Must be a positive integer.
        Default is 3.

    Returns
    -------
    smoothed : ndarray
        The smoothed data.
    """
    # Check if size is a positive integer
    if not (isinstance(size, int) and size > 0):
        raise ValueError('size must be a positive integer.')

    # Check if the input data is a numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError(f'data is not a numpy array. Type {type(data)}')

    # Ensure size is odd
    if size % 2 == 0:
        size = size - 1
        warnings.warn(
            f'size was even, using {size} instead for proper median filtering.',
            UserWarning,
        )

    if data.ndim > 3:
        warnings.warn(
            'Input data is not a 3D volume. The filter will be applied for all volumes.',
            UserWarning,
        )

    volumes, _ = collect_data_volumes(data)
    processed = []
    for volume in volumes:
        filtered_volume = median_filter(volume, size=size)
        processed.append(filtered_volume)

    return np.array(processed).reshape(data.shape)
