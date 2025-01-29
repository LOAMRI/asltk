import warnings

import numpy as np
import SimpleITK as sitk

from asltk.utils import collect_data_volumes


def isotropic_gaussian(data, sigma: float = 1.0):
    """Smooth the data using a isotropic Gaussian kernel.

    This method assumes that the same kernal size will be applied over all the
    volume dimension. The method uses the `SimpleITK` library to apply the
    smoothing.

    Note:
        If the data is higher than 3D dimension, then the method will apply the
        smoothing to all the volumes individually and reconstruct the original
        data again.

    Important:
        The kernel size, given by the sigma value, is referred to number of
        voxels considered to apply the smoothing. Therefore, when the voxel
        resolution is low (tipically for ASL data is around 3-4 mm), the sigma
        value should be around 0.5-2, depending on the desired smoothing effect.

    Parameters
    ----------
    data : array_like
        The data to be smoothed.
    sigma : float
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    smoothed : ndarray
        The smoothed data.
    """
    # Check if sigma is a positive odd number
    if not (isinstance(sigma, (int, float)) and sigma > 0):
        raise ValueError('sigma must be a positive number.')

    # Check if the input data is a numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError(f'data is not a numpy array. Type {type(data)}')

    # Make the Gaussian instance using the kernel size based on sigma parameter
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(sigma)

    if data.ndim > 3:
        warnings.warn(
            'Input data is not a 3D volume. The filter will be applied for all volumes.',
            UserWarning,
        )
    volumes, _ = collect_data_volumes(data)
    processed = []
    for volume in volumes:
        processed.append(gaussian.Execute(sitk.GetImageFromArray(volume)))

    return np.array(processed).reshape(data.shape)
