import warnings
from multiprocessing import cpu_count
from typing import Any, Dict, Optional

import numpy as np
import psutil

from asltk.smooth import isotropic_gaussian, isotropic_median
from asltk.utils.io import ImageIO


def _check_mask_values(mask: ImageIO, label, ref_shape):
    """Validate mask array for brain mask processing.

    This function performs comprehensive validation of brain mask data to ensure
    it meets the requirements for ASL processing. It checks data type, binary
    format compliance, label presence, and dimensional compatibility.

    Args:
        mask (np.ndarray): The brain mask image to validate.
        label (int or float): The label value to search for in the mask.
        ref_shape (tuple): The reference shape that the mask should match.

    Raises:
        TypeError: If mask is not a numpy array or dimensions don't match.
        ValueError: If the specified label value is not found in the mask.

    Warnings:
        UserWarning: If mask contains more than 2 unique values (not strictly binary).
    """
    # Check wheter mask input is an ImageIO object
    if not isinstance(mask, ImageIO):
        raise TypeError(
            f'mask is not an ImageIO object. Type {type(mask)} is not allowed.'
        )

    mask_array = mask.get_as_numpy()

    # Check whether the mask provided is a binary image
    unique_values = np.unique(mask_array)
    if unique_values.size > 2:
        warnings.warn(
            'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
            UserWarning,
        )

    # Check whether the label value is found in the mask image
    label_ok = False
    for value in unique_values:
        if label == value:
            label_ok = True
            break
    if not label_ok:
        raise ValueError('Label value is not found in the mask provided.')

    # Check whether the dimensions between mask and input volume matches
    mask_shape = mask_array.shape
    if mask_shape != ref_shape:
        raise TypeError(
            f'Image mask dimension does not match with input 3D volume. Mask shape {mask_shape} not equal to {ref_shape}'
        )


def _apply_smoothing_to_maps(
    maps: Dict[str, ImageIO],
    smoothing: Optional[str] = None,
    smoothing_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, ImageIO]:
    """Apply smoothing filter to all maps in the dictionary.

    This function applies the specified smoothing filter to all map arrays
    in the input dictionary. It preserves the original structure and only
    modifies the numpy arrays.

    Parameters
    ----------
    maps : dict
        Dictionary containing map arrays (e.g., {'cbf': array, 'att': array}).
    smoothing : str, optional
        Type of smoothing filter to apply. Options:
        - None: No smoothing (default)
        - 'gaussian': Gaussian smoothing using isotropic_gaussian
        - 'median': Median filtering using isotropic_median
    smoothing_params : dict, optional
        Parameters for the smoothing filter. Defaults depend on filter type:
        - For 'gaussian': {'sigma': 1.0}
        - For 'median': {'size': 3}

    Returns
    -------
    dict
        Dictionary with the same keys but smoothed arrays.

    Raises
    ------
    ValueError
        If smoothing type is not supported.
    """
    # Check it the smoothing_params is ok
    if smoothing_params is not None and not isinstance(smoothing_params, dict):
        raise TypeError(
            f'smoothing_params must be a dictionary. Type {type(smoothing_params)}'
        )
    if isinstance(smoothing_params, dict):
        if smoothing_params.get('size') or smoothing_params.get('sigma'):
            if smoothing_params.get('size') and not isinstance(
                smoothing_params['size'], int
            ):
                raise TypeError(
                    'Invalid smoothing parameter type. Size/Sigma must be an integer.'
                )
            if smoothing_params.get('sigma') and not isinstance(
                smoothing_params['sigma'], float
            ):
                raise TypeError(
                    'Invalid smoothing parameter type. Size/Sigma must be a float.'
                )

    if smoothing is None:
        return maps

    # Set default parameters
    if smoothing_params is None:
        if smoothing == 'gaussian':
            smoothing_params = {'sigma': 1.0}
        elif smoothing == 'median':
            smoothing_params = {'size': 3}
        else:
            smoothing_params = {}

    # Select smoothing function
    if smoothing == 'gaussian':
        smooth_func = isotropic_gaussian
    elif smoothing == 'median':
        smooth_func = isotropic_median
    else:
        raise ValueError(
            f'Unsupported smoothing type: {smoothing}. '
            "Supported types are: None, 'gaussian', 'median'"
        )

    # Apply smoothing to all maps
    smoothed_maps = {}
    for key, map_array in maps.items():
        if isinstance(map_array, ImageIO):
            try:
                smoothed_maps[key] = smooth_func(map_array, **smoothing_params)
            except Exception as e:
                warnings.warn(
                    f'Failed to apply {smoothing} smoothing to {key} map: {e}. '
                    f'Using original map.',
                    UserWarning,
                )
                smoothed_maps[key] = map_array
        else:
            # Non-array values are passed through unchanged
            smoothed_maps[key] = map_array

    return smoothed_maps


def get_optimal_core_count(
    requested_cores: int = None, mb_per_core: int = 500
):
    """Determine optimal number of cores based on available memory.

    This function calculates the appropriate number of CPU cores to use for
    parallel processing based on the available system memory. It ensures
    that the process won't exhaust the system's memory during computation.

    This implementation is OS-agnostic and works consistently across
    Windows, Linux, and macOS platforms.

    Args:
        requested_cores (int or str, optional): User-requested number of cores.
            If an integer and > 0, uses this value (capped by system limits).
            If "auto" or None, calculates based on available memory.
        mb_per_core (int, optional): Memory required per core in MB.
            Defaults to 500MB per core as a safe estimate.

    Returns:
        int: Optimal number of cores to use (at least 1)
    """
    # If specific cores requested (and not "auto"), respect that choice
    if requested_cores not in (None, 'auto') and requested_cores > 0:
        return min(requested_cores, cpu_count())

    # Calculate based on available memory
    free_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    cores_by_memory = max(1, int(free_memory_mb / mb_per_core))

    # Return the smaller of: cores based on memory or total available cores
    return min(cores_by_memory, cpu_count())


def estimate_memory_usage(data: np.ndarray) -> float:
    """Estimate memory usage of a numpy array in MB.

    This function calculates the memory footprint of a given numpy array
    by determining its size in bytes and converting it to megabytes (MB).

    Args:
        data (np.ndarray): The numpy array for which to estimate memory usage.

    Returns:
        float: Estimated memory usage in megabytes (MB).
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f'Input must be a numpy array, got {type(data)} instead.'
        )

    total_bytes = data.nbytes
    total_mb = total_bytes / (1024**2)  # Convert bytes to MB
    return total_mb
