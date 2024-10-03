import math
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


def asl_model_buxton(
    tau: list,
    w: list,
    m0: float,
    cbf: float,
    att: float,
    lambda_value: float = 0.98,
    t1b: float = 1650.0,
    alpha: float = 0.85,
):
    """Buxton model to calculate the ASL magnetization values.

    It is assumed that the LD and PLD values are coherent with the ASl Buxton
    model, i.e. the both has the same array size.

    The calculations is given assuming a voxel value. Hence, all the `tau`, 
    `w`, `cbf` and `att` values must representas a voxel in the image.
    
    Note:
        The CBF value is the original scale, without assuming the normalized
        CBF value. See more details at the CBFMapping class documentation.

    Args:
        tau (list): LD values
        w (list): PLD values
        m0 (float): The M0 magnetization value
        cbf (float): The CBF value, not been assumed as normalized.
        att (float): The ATT value
        lambda_value (float, optional): The blood-brain partition coefficient (0 to 1.0). Defaults to 0.98.
        t1b (float, optional): The T1 relaxation value of the blood. Defaults to 1650.0.
        alpha (float, optional): The labeling efficiency. Defaults to 0.85.

    Returns:
        (numpy.ndarray): A numpy array with the magnetization values calculated
    """

    # if type(tau) is not list or type(tau) is not tuple:
    #     raise ValueError("tau parameters must be a list or tuple")

    # if type(w) is not list or type(w) is not tuple:
    #     raise ValueError("w parameters must be a list or tuple")

    # if len(tau) != len(w):
    #     raise SyntaxError("tau and w parameters must be at the same size.")

    t = np.add(tau, w).tolist()

    t1bp = 1 / ((1 / t1b) + (cbf / lambda_value))
    m_values = np.zeros(len(tau))

    for i in range(0, len(tau)):
        try:
            if t[i] < att:
                m_values[i] = 0.0
            elif (att <= t[i]) and (t[i] < tau[i] + att):
                q = 1 - math.exp(-(t[i] - att) / t1bp)
                m_values[i] = (
                    2.0 * m0 * cbf * t1bp * alpha * q * math.exp(-att / t1b)
                )
            else:
                q = 1 - math.exp(-tau[i] / t1bp)
                m_values[i] = (
                    2.0
                    * m0
                    * cbf
                    * t1bp
                    * alpha
                    * q
                    * math.exp(-att / t1b)
                    * math.exp(-(t[i] - tau[i] - att) / t1bp)
                )
        except OverflowError:
            m_values[i] = 0.0

    return m_values
