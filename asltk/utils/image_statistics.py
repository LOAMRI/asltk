from typing import Dict

import numpy as np
from scipy.ndimage import center_of_mass


def calculate_snr(image: np.ndarray, roi: np.ndarray = None) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of a medical image.

    It is assumed the absolute value for SNR, i.e., SNR = |mean_signal| / |std_noise|.

    Parameters
    ----------
    image : np.ndarray
        The image to analyze.

    Returns
    -------
    float
        The SNR value of the image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError('Input must be a numpy array.')

    # TODO raise error roi higher than image OR different shape
    if isinstance(roi, np.ndarray):
        if any(r > i for r, i in zip(roi.shape, image.shape)):
            raise ValueError(
                'ROI must be smaller than or equal to image size in all dimensions.'
            )
        if roi.shape != image.shape:
            raise ValueError('ROI shape must be compatible to image shape.')
    else:
        raise ValueError('ROI must be a numpy array.')

    mean_signal = np.mean(image)
    noise = image - mean_signal

    try:
        snr = mean_signal / np.std(noise)
    except ZeroDivisionError:
        snr = float('inf')  # If noise is zero, SNR is infinite

    return float(abs(snr)) if snr is not np.nan else 0.0


def analyze_image_properties(image: np.ndarray) -> Dict[str, any]:
    """
    Analyze basic properties of a medical image for orientation assessment.

    Parameters
    ----------
    image : np.ndarray
        The image to analyze.

    Returns
    -------
    dict
        Dictionary containing image properties:
        - 'shape': tuple, image dimensions
        - 'center_of_mass': tuple, center of mass coordinates
        - 'intensity_stats': dict, intensity statistics
        - 'symmetry_axes': dict, symmetry analysis for each axis
    """
    # Basic properties
    shape = image.shape

    # Center of mass
    try:

        com = center_of_mass(image > np.mean(image))
    except ImportError:
        # Fallback calculation without scipy
        coords = np.argwhere(image > np.mean(image))
        com = np.mean(coords, axis=0) if len(coords) > 0 else (0, 0, 0)

    # Intensity statistics
    intensity_stats = {
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'median': float(np.median(image)),
    }

    # Symmetry analysis
    symmetry_axes = {}
    for axis in range(3):
        # Flip along axis and compare
        flipped = np.flip(image, axis=axis)
        correlation = _compute_correlation_simple(image, flipped)
        symmetry_axes[f'axis_{axis}'] = {
            'symmetry_correlation': correlation,
            'likely_symmetric': correlation > 0.8,
        }

    return {
        'shape': shape,
        'center_of_mass': com,
        'intensity_stats': intensity_stats,
        'symmetry_axes': symmetry_axes,
    }


def _compute_correlation_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple correlation computation without external dependencies."""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    if len(img1_flat) != len(img2_flat):
        return 0.0

    # Remove NaN values
    valid_mask = np.isfinite(img1_flat) & np.isfinite(img2_flat)
    if np.sum(valid_mask) < 2:
        return 0.0

    img1_valid = img1_flat[valid_mask]
    img2_valid = img2_flat[valid_mask]

    # Compute correlation
    mean1, mean2 = np.mean(img1_valid), np.mean(img2_valid)
    std1, std2 = np.std(img1_valid), np.std(img2_valid)

    if std1 == 0 or std2 == 0:
        return 0.0

    correlation = np.mean((img1_valid - mean1) * (img2_valid - mean2)) / (
        std1 * std2
    )
    return abs(correlation)
