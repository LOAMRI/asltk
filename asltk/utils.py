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
    tau = tau.tolist() if isinstance(tau, np.ndarray) else tau
    w = w.tolist() if isinstance(w, np.ndarray) else w

    if not (isinstance(tau, list) ^ isinstance(tau, tuple)):
        raise ValueError('tau parameter must be a list or tuple of values.')

    if not isinstance(w, list) ^ isinstance(w, tuple):
        raise ValueError('w parameter must be a list or tuple of values.')

    for v in tau:
        if not isinstance(v, float) ^ isinstance(v, int):
            raise ValueError('tau list must contain float or int values')

    for v in w:
        if not isinstance(v, float) ^ isinstance(v, int):
            raise ValueError('w list must contain float or int values')

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


def asl_model_multi_te(
    tau: list,
    w: list,
    te: list,
    m0: float,
    cbf: float,
    att: float,
    t2b: float = 165.0,
    t2csf: float = 75.0,
    tblcsf: float = 1400.0,
    alpha: float = 0.85,
    t1b: float = 1650.0,
    t1csf: float = 1400.0,
):
    """Multi Time of Echos (TE) ASL model to calculate the T1 relaxation time for
    blood and Grey Matter exchange.

    This model is directly used on the MultiTE_ASLMapping class.

    Reference: Ultra-long-TE arterial spin labeling reveals rapid and
    brain-wide blood-to-CSF water transport in humans, NeuroImage,
    doi: 10.1016/j.neuroimage.2021.118755

    Args:
        tau (list): The LD values
        w (list): The PLD values
        te (list): The TE values
        m0 (float): The M0 voxel value
        cbf (float): The CBF voxel value
        att (float): The ATT voxel value
        t2b (float, optional): The T2 relaxation value for blood. Defaults to 165.0.
        t2csf (float, optional): The T2 relaxation value for CSF. Defaults to 75.0.
        tblcsf (float, optional): The T1 relaxation value between blood and CSF. Defaults to 1400.0.
        alpha (float, optional): The pulse labeling efficiency. Defaults to 0.85.
        t1b (float, optional): The T1 relaxation value for blood. Defaults to 1650.0.
        t1csf (float, optional): The T1 relaxation value for CSF. Defaults to 1400.0.

    Returns:
        (nd.ndarray): The magnetization values for T1-Blood-GM
    """
    t1bp = 1 / ((1 / t1b) + (1 / tblcsf))
    t1csfp = 1 / ((1 / t1csf) + (1 / tblcsf))

    t2bp = 1 / ((1 / t2b) + (1 / tblcsf))
    t2csfp = 1 / ((1 / t2csf) + (1 / tblcsf))

    t = np.add(tau, w).tolist()

    mag_total = np.zeros(len(tau))

    for i in range(0, len(tau)):
        try:
            if t[i] < att:
                S1b = 0.0
                S1csf = 0.0
                if te[i] < (att - t[i]):
                    Sb = 0
                    Scsf = 0
                elif (att - t[i]) <= te[i] and te[i] < (att + tau[i] - t[i]):
                    Sb = (
                        2
                        * alpha
                        * m0
                        * cbf
                        * t2bp
                        * math.exp(-att / t1b)
                        * math.exp(-te[i] / t2b)
                        * (1 - math.exp(-(te[i] - att + t[i]) / t2bp))
                    )   #% measured signal = S2
                    Scsf = (
                        2
                        * alpha
                        * m0
                        * cbf
                        * math.exp(-att / t1b)
                        * math.exp(-te[i] / t2b)
                        * (
                            t2csf
                            * (1 - math.exp(-(te[i] - att + t[i]) / t2csf))
                            - t2csfp
                            * (1 - math.exp(-(te[i] - att + t[i]) / t2csfp))
                        )
                    )
                else:   #% att + tau - t <= te
                    Sb = (
                        2
                        * alpha
                        * m0
                        * cbf
                        * t2bp
                        * math.exp(-att / t1b)
                        * math.exp(-te[i] / t2b)
                        * math.exp(-(te[i] - att + t[i]) / t2bp)
                        * (math.exp(tau[i] / t2bp) - 1)
                    )
                    Scsf = (
                        2
                        * alpha
                        * m0
                        * cbf
                        * math.exp(-att / t1b)
                        * math.exp(-te[i] / t2b)
                        * (
                            t2csf
                            * math.exp(-(te[i] - att + t[i]) / t2csf)
                            * (math.exp(tau[i] / t2csf) - 1)
                            - t2csfp
                            * math.exp(-(te[i] - att + t[i]) / t2csfp)
                            * (math.exp(tau[i] / t2csfp) - 1)
                        )
                    )
            elif (att <= t[i]) and (t[i] < (att + tau[i])):
                S1b = (
                    2
                    * alpha
                    * m0
                    * cbf
                    * t1bp
                    * math.exp(-att / t1b)
                    * (1 - math.exp(-(t[i] - att) / t1bp))
                )
                S1csf = (
                    2
                    * alpha
                    * m0
                    * cbf
                    * math.exp(-att / t1b)
                    * (
                        t1csf * (1 - math.exp(-(t[i] - att) / t1csf))
                        - t1csfp * (1 - math.exp(-(t[i] - att) / t1csfp))
                    )
                )
                if te[i] < (att + tau[i] - t[i]):
                    Sb = S1b * math.exp(
                        -te[i] / t2bp
                    ) + 2 * alpha * m0 * cbf * t2bp * math.exp(
                        -att / t1b
                    ) * math.exp(
                        -te[i] / t2b
                    ) * (
                        1 - math.exp(-te[i] / t2bp)
                    )
                    Scsf = (
                        S1b
                        * (1 - math.exp(-te[i] / tblcsf))
                        * math.exp(-te[i] / t2csf)
                        + S1csf * math.exp(-te[i] / t2csf)
                        + 2
                        * alpha
                        * m0
                        * cbf
                        * math.exp(-att / t1b)
                        * math.exp(-te[i] / t2b)
                        * (
                            t2csf * (1 - math.exp(-te[i] / t2csf))
                            - t2csfp * (1 - math.exp(-te[i] / t2csfp))
                        )
                    )
                else:   # att + tau - t <= te
                    Sb = S1b * math.exp(
                        -te[i] / t2bp
                    ) + 2 * alpha * m0 * cbf * t2bp * math.exp(
                        -att / t1b
                    ) * math.exp(
                        -te[i] / t2b
                    ) * math.exp(
                        -te[i] / t2bp
                    ) * (
                        math.exp((att + tau[i] - t[i]) / t2bp) - 1
                    )
                    Scsf = (
                        S1b
                        * (1 - math.exp(-te[i] / tblcsf))
                        * math.exp(-te[i] / t2csf)
                        + S1csf * math.exp(-te[i] / t2csf)
                        + 2
                        * alpha
                        * m0
                        * cbf
                        * math.exp(-att / t1b)
                        * math.exp(-te[i] / t2b)
                        * (
                            t2csf
                            * math.exp(-te[i] / t2csf)
                            * (math.exp((att + tau[i] - t[i]) / t2csf) - 1)
                            - t2csfp
                            * math.exp(-te[i] / t2csfp)
                            * (math.exp((att + tau[i] - t[i]) / t2csfp) - 1)
                        )
                    )
            else:   # att+tau < t
                S1b = (
                    2
                    * alpha
                    * m0
                    * cbf
                    * t1bp
                    * math.exp(-att / t1b)
                    * math.exp(-(t[i] - att) / t1bp)
                    * (math.exp(tau[i] / t1bp) - 1)
                )
                S1csf = (
                    2
                    * alpha
                    * m0
                    * cbf
                    * math.exp(-att / t1b)
                    * (
                        t1csf
                        * math.exp(-(t[i] - att) / t1csf)
                        * (math.exp(tau[i] / t1csf) - 1)
                        - t1csfp
                        * math.exp(-(t[i] - att) / t1csfp)
                        * (math.exp(tau[i] / t1csfp) - 1)
                    )
                )

                Sb = S1b * math.exp(-te[i] / t2bp)
                Scsf = S1b * (1 - math.exp(-te[i] / tblcsf)) * math.exp(
                    -te[i] / t2csf
                ) + S1csf * math.exp(-te[i] / t2csf)
        except (OverflowError, RuntimeError):
            Sb = 0.0
            Scsf = 0.0

        mag_total[i] = Sb + Scsf

    return mag_total
