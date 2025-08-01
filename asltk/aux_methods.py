import warnings

import numpy as np


def _check_mask_values(mask, label, ref_shape):
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
