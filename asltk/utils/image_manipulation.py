import os
from typing import Dict, List, Optional, Tuple, Union

import ants
import numpy as np
import SimpleITK as sitk
from rich import print

from asltk.logging_config import get_logger
from asltk.utils.image_statistics import (
    analyze_image_properties,
    calculate_mean_intensity,
    calculate_snr,
)

logger = get_logger(__name__)

# Set SimpleITK to use half of available CPU cores (at least 1)
num_cores = max(1, os.cpu_count() // 4 if os.cpu_count() else 1)
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_cores)


def collect_data_volumes(data: np.ndarray):
    """Collect the data volumes from a higher dimension array.

    This method is used to collect the data volumes from a higher dimension
    array. The method assumes that the data is a 4D array, where the first
    dimension is the number of volumes. The method will collect the volumes
    and return a list of 3D arrays.

    The method is used to separate the 3D volumes from the higher dimension
    array. This is useful when the user wants to apply a filter to each volume
    separately.

    Args:
        data (np.ndarray): The data to be separated.

    Returns:
        list: A list of 3D arrays, each one representing a volume.
        tuple: The original shape of the data.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data is not a numpy array.')

    if data.ndim < 3:
        raise ValueError('data is a 3D volume or higher dimensions')

    volumes = []
    # Calculate the number of volumes by multiplying all dimensions except the last three
    num_volumes = int(np.prod(data.shape[:-3]))
    reshaped_data = data.reshape((int(num_volumes),) + data.shape[-3:])
    for i in range(num_volumes):
        volumes.append(reshaped_data[i])

    return volumes, data.shape


def orientation_check(
    moving_image: np.ndarray, fixed_image: np.ndarray, threshold: float = 0.1
) -> Dict[str, any]:
    """
    Quick orientation compatibility check between two images.

    This function provides a fast assessment of whether two images
    have compatible orientations for registration without applying
    any corrections.

    Parameters
    ----------
    moving_image : np.ndarray
        The moving image to be checked.
    fixed_image : np.ndarray
        The reference/fixed image.
    threshold : float, optional
        Correlation threshold to consider orientations compatible. Default is 0.1.

    Returns
    -------
    dict
        Dictionary containing:
        - 'compatible': bool, whether orientations are compatible
        - 'correlation': float, normalized correlation between images
        - 'recommendation': str, action recommendation
    """
    # Normalize images
    moving_norm = _normalize_image_intensity(moving_image)
    fixed_norm = _normalize_image_intensity(fixed_image)

    # Resize if needed for comparison
    # Resize the larger image to match the smaller one to minimize memory overhead
    if moving_norm.shape != fixed_norm.shape:
        if np.prod(moving_norm.shape) > np.prod(fixed_norm.shape):
            moving_norm = _resize_image_to_match(moving_norm, fixed_norm.shape)
        else:
            fixed_norm = _resize_image_to_match(fixed_norm, moving_norm.shape)

    # Compute correlation
    correlation = _compute_normalized_correlation(moving_norm, fixed_norm)

    # Determine compatibility
    compatible = correlation > threshold

    if compatible:
        recommendation = 'Images appear to have compatible orientations. Registration should proceed normally.'
    elif correlation > 0.05:
        recommendation = 'Possible orientation mismatch detected. Consider using orientation correction.'
    else:
        recommendation = 'Strong orientation mismatch detected. Orientation correction is highly recommended.'

    return {
        'compatible': compatible,
        'correlation': correlation,
        'recommendation': recommendation,
    }


# TODO Evaluate this method and decide if it is needed (or useful...)
# def preview_orientation_correction(
#     moving_image: np.ndarray,
#     fixed_image: np.ndarray,
#     slice_index: Optional[int] = None
# ) -> Dict[str, np.ndarray]:
#     """
#     Preview the effect of orientation correction on a specific slice.

#     This function shows the before and after effect of orientation
#     correction on a 2D slice, useful for visual validation.

#     Parameters
#     ----------
#     moving_image : np.ndarray
#         The moving image to be corrected.
#     fixed_image : np.ndarray
#         The reference/fixed image.
#     slice_index : int, optional
#         Index of the axial slice to preview. If None, uses middle slice.

#     Returns
#     -------
#     dict
#         Dictionary containing:
#         - 'original_slice': np.ndarray, original moving image slice
#         - 'corrected_slice': np.ndarray, corrected moving image slice
#         - 'fixed_slice': np.ndarray, corresponding fixed image slice
#         - 'slice_index': int, the slice index used
#     """
#     # Get orientation correction
#     corrected_moving, _ = check_and_fix_orientation(
#         moving_image, fixed_image, verbose=False
#     )

#     # Determine slice index
#     if slice_index is None:
#         slice_index = moving_image.shape[0] // 2

#     # Ensure slice index is valid
#     slice_index = max(0, min(slice_index, moving_image.shape[0] - 1))
#     corrected_slice_idx = max(0, min(slice_index, corrected_moving.shape[0] - 1))
#     fixed_slice_idx = max(0, min(slice_index, fixed_image.shape[0] - 1))

#     return {
#         'original_slice': moving_image[slice_index, :, :],
#         'corrected_slice': corrected_moving[corrected_slice_idx, :, :],
#         'fixed_slice': fixed_image[fixed_slice_idx, :, :],
#         'slice_index': slice_index
#     }


def check_and_fix_orientation(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    moving_spacing: tuple = None,
    fixed_spacing: tuple = None,
    verbose: bool = False,
):
    """
    Check and fix orientation mismatches between moving and fixed images.

    This function analyzes the anatomical orientations of both images and
    applies necessary transformations to align them before registration.
    It handles common orientation issues like axial, sagittal, or coronal flips.

    The method uses both intensity-based and geometric approaches to determine
    the best orientation alignment between images.

    Parameters
    ----------
    moving_image : np.ndarray
        The moving image that needs to be aligned.
    fixed_image : np.ndarray
        The reference/fixed image.
    moving_spacing : tuple, optional
        Voxel spacing for the moving image (x, y, z). If None, assumes isotropic.
    fixed_spacing : tuple, optional
        Voxel spacing for the fixed image (x, y, z). If None, assumes isotropic.
    verbose : bool, optional
        If True, prints detailed orientation analysis. Default is False.

    Returns
    -------
    corrected_moving : np.ndarray
        The moving image with corrected orientation.
    orientation_transform : dict
        Dictionary containing the applied transformations for reproducibility.
    """
    if verbose:
        print('Analyzing image orientations...')

    # Convert to SimpleITK images for orientation analysis
    moving_sitk = sitk.GetImageFromArray(moving_image)
    fixed_sitk = sitk.GetImageFromArray(fixed_image)

    # Set spacing if provided
    if moving_spacing is not None:
        moving_sitk.SetSpacing(moving_spacing)
    if fixed_spacing is not None:
        fixed_sitk.SetSpacing(fixed_spacing)

    # Get image dimensions and properties
    moving_size = moving_sitk.GetSize()
    fixed_size = fixed_sitk.GetSize()

    if verbose:
        print(f'Moving image size: {moving_size}')
        print(f'Fixed image size: {fixed_size}')

    # Analyze anatomical orientations using intensity patterns
    orientation_transform = _analyze_anatomical_orientation(
        moving_image, fixed_image, verbose
    )

    # Apply orientation corrections
    corrected_moving = _apply_orientation_correction(
        moving_image, orientation_transform, verbose
    )

    # Verify the correction using cross-correlation
    if verbose:
        original_corr = _compute_normalized_correlation(
            moving_image, fixed_image
        )
        corrected_corr = _compute_normalized_correlation(
            corrected_moving, fixed_image
        )
        print(f'Original correlation: {original_corr:.4f}')
        print(f'Corrected correlation: {corrected_corr:.4f}')
        if corrected_corr > original_corr:
            print('Orientation correction improved alignment')
        else:
            print('Orientation correction may not have improved alignment')

    return corrected_moving, orientation_transform


def create_orientation_report(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    output_path: Optional[str] = None,
) -> str:
    """
    Create a comprehensive orientation analysis report.

    Parameters
    ----------
    moving_image : np.ndarray
        The moving image to analyze.
    fixed_image : np.ndarray
        The reference/fixed image.
    output_path : str, optional
        Path to save the report. If None, returns the report as string.

    Returns
    -------
    str
        The orientation analysis report.
    """
    # Perform analysis
    quick_check = orientation_check(moving_image, fixed_image)
    moving_props = analyze_image_properties(moving_image)
    fixed_props = analyze_image_properties(fixed_image)

    # Get correction info
    corrected_moving, orientation_transform = check_and_fix_orientation(
        moving_image, fixed_image, verbose=False
    )

    # Generate report
    report = f"""
    ORIENTATION ANALYSIS REPORT
    ============================

    QUICK COMPATIBILITY CHECK:
    - Orientation Compatible: {quick_check['compatible']}
    - Correlation Score: {quick_check['correlation']:.4f}
    - Recommendation: {quick_check['recommendation']}

    MOVING IMAGE PROPERTIES:
    - Shape: {moving_props['shape']}
    - Center of Mass: {moving_props['center_of_mass']}
    - Intensity Range: {moving_props['intensity_stats']['min']:.2f} - {moving_props['intensity_stats']['max']:.2f}
    - Mean Intensity: {moving_props['intensity_stats']['mean']:.2f}

    FIXED IMAGE PROPERTIES:
    - Shape: {fixed_props['shape']}
    - Center of Mass: {fixed_props['center_of_mass']}
    - Intensity Range: {fixed_props['intensity_stats']['min']:.2f} - {fixed_props['intensity_stats']['max']:.2f}
    - Mean Intensity: {fixed_props['intensity_stats']['mean']:.2f}

    ORIENTATION CORRECTION APPLIED:
    - X-axis flip: {orientation_transform.get('flip_x', False)}
    - Y-axis flip: {orientation_transform.get('flip_y', False)}
    - Z-axis flip: {orientation_transform.get('flip_z', False)}
    - Axis transpose: {orientation_transform.get('transpose_axes', 'None')}

    RECOMMENDATIONS:
    {quick_check['recommendation']}
        """.strip()

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f'Report saved to: {output_path}')

    return report


def select_reference_volume(
    asl_data: Union['ASLData', list[np.ndarray]],
    roi: np.ndarray = None,
    method: str = 'snr',
):
    from asltk.asldata import ASLData

    """
    Select a reference volume from the ASL data based on a specified method.

    Parameters
    ----------
    asl_data : ASLData
        The ASL data object containing the image volumes.
    roi : np.ndarray, optional
        Region of interest mask to limit the analysis.
    method : str
        The method to use for selecting the reference volume. Options are:
        - 'snr': Select the volume with the highest signal-to-noise ratio.
        - 'mean': Select the volume with the highest mean signal intensity.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple informing the selected reference volume and its index in the ASL `pcasl` data.
    """
    if method not in ('snr', 'mean'):
        raise ValueError(f'Invalid method: {method}')

    if roi is not None:
        if not isinstance(roi, np.ndarray):
            raise TypeError('ROI must be a numpy array.')
        if roi.ndim != 3:
            raise ValueError('ROI must be a 3D array.')

    if isinstance(asl_data, ASLData):
        volumes, _ = collect_data_volumes(asl_data('pcasl'))
    elif isinstance(asl_data, list) and all(
        isinstance(vol, np.ndarray) for vol in asl_data
    ):
        volumes = asl_data
    else:
        raise TypeError(
            'asl_data must be an ASLData object or a list of numpy arrays.'
        )

    if method == 'snr':
        logger.info('Estimating maximum SNR from provided volumes...')
        ref_volume, vol_idx = _estimate_max_snr(volumes, roi=roi)
        logger.info(
            f'Selected volume index: {vol_idx} with SNR: {calculate_snr(ref_volume):.2f}'
        )

    elif method == 'mean':
        logger.info('Estimating maximum mean from provided volumes...')
        ref_volume, vol_idx = _estimate_max_mean(volumes, roi=roi)
        logger.info(
            f'Selected volume index: {vol_idx} with mean: {ref_volume.mean():.2f}'
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    return ref_volume, vol_idx


def _estimate_max_snr(
    volumes: List[np.ndarray], roi: np.ndarray = None
) -> Tuple[np.ndarray, int]:   # pragma: no cover
    """
    Estimate the maximum SNR from a list of volumes.

    Args:
        volumes (List[np.ndarray]): A list of 3D numpy arrays representing the image volumes.

    Raises:
        TypeError: If any volume is not a numpy array.

    Returns:
        Tuple[np.ndarray, int]: The reference volume and its index.
    """
    max_snr_idx = 0
    max_snr_value = 0
    for idx, vol in enumerate(volumes):
        if not isinstance(vol, np.ndarray):
            logger.error(f'Volume at index {idx} is not a numpy array.')
            raise TypeError('All volumes must be numpy arrays.')

        snr_value = calculate_snr(vol, roi=roi)
        if snr_value > max_snr_value:
            max_snr_value = snr_value
            max_snr_idx = idx

    ref_volume = volumes[max_snr_idx]

    return ref_volume, max_snr_idx


def _estimate_max_mean(
    volumes: List[np.ndarray], roi: np.ndarray = None
) -> Tuple[np.ndarray, int]:
    """
    Estimate the maximum mean from a list of volumes.

    Args:
        volumes (List[np.ndarray]): A list of 3D numpy arrays representing the image volumes.

    Raises:
        TypeError: If any volume is not a numpy array.

    Returns:
        Tuple[np.ndarray, int]: The reference volume and its index.
    """
    max_mean_idx = 0
    max_mean_value = 0
    for idx, vol in enumerate(volumes):
        if not isinstance(vol, np.ndarray):
            logger.error(f'Volume at index {idx} is not a numpy array.')
            raise TypeError('All volumes must be numpy arrays.')

        mean_value = calculate_mean_intensity(vol, roi=roi)
        if mean_value > max_mean_value:
            max_mean_value = mean_value
            max_mean_idx = idx

    ref_volume = volumes[max_mean_idx]

    return ref_volume, max_mean_idx


def _analyze_anatomical_orientation(moving_image, fixed_image, verbose=False):
    """
    Analyze anatomical orientations by comparing intensity patterns
    and geometric properties of brain images.
    """
    orientation_transform = {
        'flip_x': False,
        'flip_y': False,
        'flip_z': False,
        'transpose_axes': None,
    }

    # Normalize images for comparison
    moving_norm = _normalize_image_intensity(moving_image)
    fixed_norm = _normalize_image_intensity(fixed_image)

    # Determine the smaller shape for comparison
    moving_size = np.prod(moving_norm.shape)
    fixed_size = np.prod(fixed_norm.shape)
    if moving_size <= fixed_size:
        ref_shape = moving_norm.shape
    else:
        ref_shape = fixed_norm.shape

    # Test different orientation combinations
    best_corr = -1
    best_transform = orientation_transform.copy()

    # Test axis flips
    for flip_x in [False, True]:
        for flip_y in [False, True]:
            for flip_z in [False, True]:
                # Apply test transformation
                test_img = moving_norm.copy()
                if flip_x:
                    test_img = np.flip(test_img, axis=2)  # X axis
                if flip_y:
                    test_img = np.flip(test_img, axis=1)  # Y axis
                if flip_z:
                    test_img = np.flip(test_img, axis=0)  # Z axis

                # Resize to match reference shape if needed
                if test_img.shape != ref_shape:
                    test_img = _resize_image_to_match(test_img, ref_shape)

                # Also resize fixed_norm if needed
                ref_img = fixed_norm
                if fixed_norm.shape != ref_shape:
                    ref_img = _resize_image_to_match(fixed_norm, ref_shape)

                # Compute correlation
                corr = _compute_normalized_correlation(test_img, ref_img)

                if corr > best_corr:
                    best_corr = corr
                    best_transform = {
                        'flip_x': flip_x,
                        'flip_y': flip_y,
                        'flip_z': flip_z,
                        'transpose_axes': None,
                    }

                if verbose:
                    print(
                        f'Flip X:{flip_x}, Y:{flip_y}, Z:{flip_z} -> Correlation: {corr:.4f}'
                    )

    # Test common axis permutations for different acquisition orientations
    axis_permutations = [
        (0, 1, 2),  # Original
        (0, 2, 1),  # Swap Y-Z
        (1, 0, 2),  # Swap X-Y
        (1, 2, 0),  # Rotate axes
        (2, 0, 1),  # Rotate axes
        (2, 1, 0),  # Swap X-Z
    ]

    for axes in axis_permutations[1:]:  # Skip original
        try:
            test_img = np.transpose(moving_norm, axes)
            # Resize to match reference shape if needed
            if test_img.shape != ref_shape:
                test_img = _resize_image_to_match(test_img, ref_shape)

            # Also resize fixed_norm if needed
            ref_img = fixed_norm
            if fixed_norm.shape != ref_shape:
                ref_img = _resize_image_to_match(fixed_norm, ref_shape)

            corr = _compute_normalized_correlation(test_img, ref_img)

            if corr > best_corr:
                best_corr = corr
                best_transform = {
                    'flip_x': False,
                    'flip_y': False,
                    'flip_z': False,
                    'transpose_axes': axes,
                }

            if verbose:
                print(f'Transpose {axes} -> Correlation: {corr:.4f}')
        except Exception as e:
            if verbose:
                print(f'Failed transpose {axes}: {e}')
            continue

    if verbose:
        print(f'Best orientation transform: {best_transform}')
        print(f'Best correlation: {best_corr:.4f}')

    return best_transform


def _apply_orientation_correction(image, orientation_transform, verbose=False):
    """Apply the determined orientation corrections to the image."""
    corrected = image.copy()

    # Apply axis transposition first if needed
    if orientation_transform['transpose_axes'] is not None:
        corrected = np.transpose(
            corrected, orientation_transform['transpose_axes']
        )
        if verbose:
            print(
                f"Applied transpose: {orientation_transform['transpose_axes']}"
            )

    # Apply axis flips
    if orientation_transform['flip_x']:
        corrected = np.flip(corrected, axis=2)
        if verbose:
            print('Applied X-axis flip')

    if orientation_transform['flip_y']:
        corrected = np.flip(corrected, axis=1)
        if verbose:
            print('Applied Y-axis flip')

    if orientation_transform['flip_z']:
        corrected = np.flip(corrected, axis=0)
        if verbose:
            print('Applied Z-axis flip')

    return corrected


def _normalize_image_intensity(image):
    """Normalize image intensity to [0, 1] range for comparison."""
    img = image.astype(np.float64)
    img_min, img_max = np.min(img), np.max(img)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def _resize_image_to_match(source_image, resample_shape):
    """Resize source image to match target shape using antsPy (ants)."""

    # Convert numpy array to ANTsImage (assume float32 for compatibility)
    ants_img = ants.from_numpy(source_image.astype(np.float32))

    # Resample to target shape
    resampled_img = ants.resample_image(
        ants_img, resample_shape, use_voxels=True, interp_type=0
    )

    # Convert back to numpy array
    return resampled_img.numpy()


def _compute_normalized_correlation(img1, img2):
    """Compute normalized cross-correlation between two images."""
    # Ensure same shape
    if img1.shape != img2.shape:
        return -1

    # Flatten images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Remove NaN and infinite values
    valid_mask = np.isfinite(img1_flat) & np.isfinite(img2_flat)
    if np.sum(valid_mask) == 0:
        return -1

    img1_valid = img1_flat[valid_mask]
    img2_valid = img2_flat[valid_mask]

    # Compute correlation coefficient
    try:
        corr_matrix = np.corrcoef(img1_valid, img2_valid)
        correlation = corr_matrix[0, 1]
        if np.isnan(correlation):
            return -1
        return abs(
            correlation
        )  # Use absolute value for orientation independence
    except:
        return -1
