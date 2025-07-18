import warnings
from typing import Any, Dict, Optional

import numpy as np

from asltk.smooth import isotropic_gaussian, isotropic_median


def _check_mask_values(mask, label, ref_shape):
    # Check wheter mask input is an numpy array
    if not isinstance(mask, np.ndarray):
        raise TypeError(f'mask is not an numpy array. Type {type(mask)}')

    # Check whether the mask provided is a binary image
    unique_values = np.unique(mask)
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
    mask_shape = mask.shape
    if mask_shape != ref_shape:
        raise TypeError(
            f'Image mask dimension does not match with input 3D volume. Mask shape {mask_shape} not equal to {ref_shape}'
        )


def _apply_smoothing_to_maps(
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
