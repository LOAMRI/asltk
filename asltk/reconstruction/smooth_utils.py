"""
Utility functions for applying smoothing to reconstruction maps.
"""

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np

from asltk.smooth import isotropic_gaussian, isotropic_median


def apply_smoothing_to_maps(
    maps: Dict[str, np.ndarray],
    smoothing: Optional[str] = None,
    smoothing_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
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
        if isinstance(map_array, np.ndarray):
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
