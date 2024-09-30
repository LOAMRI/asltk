import os

import numpy as np
import SimpleITK as sitk


def load_image(full_path: str):
    """Load an image file path using the standard SimpleITK API.

    The output format for object handler is a numpy array, collected from
    the SimpleITK reading data method.

    For more details about the image formatts accepted, check the official
    documentation at: https://simpleitk.org/

    The ASLData class assumes as a caller method to expose the image array
    directly to the user, hence calling the object instance will return the
    image array directly.

    Args:
        full_path (str): Absolute path for the image file
        spec (str): The type of image data that is being loaded. Use the
        following options: 'm0', 'pcasl'.

    Examples:
        >>> data=load_image("./tests/files/t1-mri.nrrd")
        >>> type(data)
        <class 'numpy.ndarray'>

    Returns:
        (numpy.array): The loaded image as the determined type
    """
    _check_input_path(full_path)
    img = sitk.ReadImage(full_path)

    return sitk.GetArrayFromImage(img)


def save_image(img: np.ndarray, full_path: str):
    """Save image to a file path.

    All the available image formats provided in the SimpleITK API can be
    used here.

    Args:
        full_path (str): Full absolute path with image file name provided.
    """
    sitk_img = sitk.GetImageFromArray(img)
    sitk.WriteImage(sitk_img, full_path)


def _check_input_path(path):
    if not os.path.exists(path):
        raise ValueError('Image path is not valid or image not found.')
