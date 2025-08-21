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
from asltk.utils.io import ImageIO, clone_image

logger = get_logger(__name__)

# Set SimpleITK to use half of available CPU cores (at least 1)
num_cores = max(1, os.cpu_count() // 4 if os.cpu_count() else 1)
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_cores)


def collect_data_volumes(
    data: ImageIO,
) -> Tuple[List[ImageIO], Tuple[int, ...]]:
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
        list: A list of ImageIO, each one representing a volume.
        tuple: The original shape of the data.
    """
    if not isinstance(data, ImageIO):
        raise TypeError('data is not an ImageIO object.')

    dimension = data.get_as_numpy().ndim
    if dimension < 3:
        raise ValueError('data is not a 3D volume or higher dimensions')

    volumes = []
    # Calculate the number of volumes by multiplying all dimensions except the last three
    num_volumes = int(np.prod(data.get_as_numpy().shape[:-3]))
    reshaped_data = data.get_as_numpy().reshape(
        (int(num_volumes),) + data.get_as_numpy().shape[-3:]
    )
    for i in range(num_volumes):
        base_data = ImageIO(image_array=reshaped_data[i])
        base_data.update_image_spacing(data._image_as_sitk.GetSpacing()[:3])
        base_data.update_image_origin(data._image_as_sitk.GetOrigin()[:3])

        tmp_dir_array = np.array(data._image_as_sitk.GetDirection()).reshape(
            dimension, dimension
        )
        base_data.update_image_direction(
            tuple(tmp_dir_array[:3, :3].flatten())
        )

        volumes.append(base_data)

    return volumes, data.get_as_numpy().shape


def select_reference_volume(
    asl_data: Union['ASLData', list[ImageIO]],
    roi: ImageIO = None,
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
        if not isinstance(roi, ImageIO):
            raise TypeError('ROI must be an ImageIO object.')
        if roi.get_as_numpy().ndim != 3:
            raise ValueError('ROI must be a 3D array.')

    if isinstance(asl_data, ASLData):
        volumes, _ = collect_data_volumes(asl_data('pcasl'))
    elif isinstance(asl_data, list) and all(
        isinstance(vol, ImageIO) for vol in asl_data
    ):
        volumes = asl_data
    else:
        raise TypeError(
            'asl_data must be an ASLData object or a list of ImageIO objects.'
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
            f'Selected volume index: {vol_idx} with mean: {ref_volume.get_as_numpy().mean():.2f}'
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    return ref_volume, vol_idx


def _estimate_max_snr(
    volumes: List[ImageIO], roi: ImageIO = None
) -> Tuple[ImageIO, int]:   # pragma: no cover
    """
    Estimate the maximum SNR from a list of volumes.

    Args:
        volumes (List[ImageIO]): A list of ImageIO objects representing the image volumes.

    Raises:
        TypeError: If any volume is not a numpy array.

    Returns:
        Tuple[np.ndarray, int]: The reference volume and its index.
    """
    max_snr_idx = 0
    max_snr_value = 0
    for idx, vol in enumerate(volumes):
        if not isinstance(vol, ImageIO):
            logger.error(f'Volume at index {idx} is not an ImageIO object.')
            raise TypeError('All volumes must be ImageIO objects.')

        if roi is not None:
            snr_value = calculate_snr(vol, roi=roi)
        else:
            snr_value = calculate_snr(vol)

        if snr_value > max_snr_value:
            max_snr_value = snr_value
            max_snr_idx = idx

    ref_volume = volumes[max_snr_idx]

    return ref_volume, max_snr_idx


def _estimate_max_mean(
    volumes: List[ImageIO], roi: ImageIO = None
) -> Tuple[ImageIO, int]:
    """
    Estimate the maximum mean from a list of volumes.

    Args:
        volumes (List[ImageIO]): A list of ImageIO objects representing the image volumes.

    Raises:
        TypeError: If any volume is not an ImageIO object.

    Returns:
        Tuple[np.ndarray, int]: The reference volume and its index.
    """
    max_mean_idx = 0
    max_mean_value = 0
    for idx, vol in enumerate(volumes):
        if not isinstance(vol, ImageIO):
            logger.error(f'Volume at index {idx} is not an ImageIO object.')
            raise TypeError('All volumes must be ImageIO objects.')

        if roi is not None:
            mean_value = calculate_mean_intensity(vol, roi=roi)
        else:
            mean_value = calculate_mean_intensity(vol)
        if mean_value > max_mean_value:
            max_mean_value = mean_value
            max_mean_idx = idx

    ref_volume = volumes[max_mean_idx]

    return ref_volume, max_mean_idx
