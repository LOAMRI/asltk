import fnmatch
import math
import os

import dill
import numpy as np
import SimpleITK as sitk
from bids import BIDSLayout

from asltk import AVAILABLE_IMAGE_FORMATS, BIDS_IMAGE_FORMATS


def _check_input_path(full_path: str):
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'The file {full_path} does not exist.')


def load_image(
    full_path: str,
    subject: str = None,
    session: str = None,
    modality: str = None,
    suffix: str = None,
):
    """Load an image file from a BIDS directory using the standard SimpleITK API.

    The output format for object handler is a numpy array, collected from
    the SimpleITK reading data method.

    For more details about the image formats accepted, check the official
    documentation at: https://simpleitk.org/

    The ASLData class assumes as a caller method to expose the image array
    directly to the user, hence calling the object instance will return the
    image array directly.

    Note:
        This method accepts a full path to a file or a BIDS directory. If the
        BIDS file is provided, then the `subject`, `session`, `modality` and
        `suffix` must be provided. Otherwise, the method will search for the
        first image file found in the BIDS directory that can be an estimate
        ASL image.

    Tip:
        To be sure that the input BIDS structure is correct, use the
        `bids-validator` tool to check the BIDS structure. See more details at:
        https://bids-standard.github.io/bids-validator/. For more deteils about
        ASL BIDS structure, check the official documentation at:
        https://bids-specification.readthedocs.io/en/latest

    Args:
        full_path (str): Path to the BIDS directory
        subject (str): Subject identifier
        session (str, optional): Session identifier. Defaults to None.
        modality (str, optional): Modality folder name. Defaults to 'asl'.
        suffix (str, optional): Suffix of the file to load. Defaults to 'T1w'.

    Examples:
        >>> data = load_image("./tests/files/bids-example/asl001")
        >>> type(data)
        <class 'numpy.ndarray'>

        In this form the input data is a BIDS directory. It all the BIDS
        parameters are kept as `None`, then the method will search for the
        first image that is an ASL image.

        One can choose to load a determined BIDS data using more deatail, such
        as the subject, session, modality and suffix:
        >>> data = load_image("./tests/files/bids-example/asl001", subject='103', suffix='asl')
        >>> type(data)
        <class 'numpy.ndarray'>

    Returns:
        (numpy.array): The loaded image
    """
    _check_input_path(full_path)

    if full_path.endswith(AVAILABLE_IMAGE_FORMATS):
        # If the full path is a file, then load the image directly
        img = sitk.ReadImage(full_path)
        return sitk.GetArrayFromImage(img)

    # Check if the full path is a directory using BIDS structure
    selected_file = ''
    layout = BIDSLayout(full_path)
    if all(param is None for param in [subject, session, modality, suffix]):
        for root, _, files in os.walk(full_path):
            for file in files:
                if '_asl' in file and file.endswith(BIDS_IMAGE_FORMATS):
                    selected_file = os.path.join(root, file)
    else:
        layout_files = layout.files.keys()
        matching_files = []
        for f in layout_files:
            search_pattern = ''
            if subject:
                search_pattern = f'*sub-*{subject}*'
            if session:
                search_pattern += search_pattern + f'*ses-*{session}'
            if modality:
                search_pattern += search_pattern + f'*{modality}*'
            if suffix:
                search_pattern += search_pattern + f'*{suffix}*'

            if fnmatch.fnmatch(f, search_pattern) and f.endswith(
                BIDS_IMAGE_FORMATS
            ):
                matching_files.append(f)

        if not matching_files:
            raise FileNotFoundError(
                f'ASL image file is missing for subject {subject} in directory {full_path}'
            )
        selected_file = matching_files[0]

    img = sitk.ReadImage(selected_file)
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


def save_asl_data(asldata, fullpath: str):
    """
    Save ASL data to a pickle file.

    This method saves the ASL data to a pickle file using the dill library. All
    the ASL data will be saved in a single file. After the file being saved, it
    can be loaded using the `load_asl_data` method.

    This method can be helpful when one wants to save the ASL data to a file
    and share it with others or use it in another script. The entire ASLData
    object will be loaded from the file, maintaining all the data and
    parameters described in the `ASLData` class.

    Examples:
        >>> from asltk.asldata import ASLData
        >>> asldata = ASLData(pcasl='./tests/files/pcasl_mte.nii.gz', m0='./tests/files/m0.nii.gz',ld_values=[1.8, 1.8, 1.8], pld_values=[1.8, 1.8, 1.8], te_values=[1.8, 1.8, 1.8])
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
        ...     temp_file_path = temp_file.name
        >>> save_asl_data(asldata, temp_file_path)


    Note:
        This method only accepts the ASLData object as input. If you want to
        save an image, then use the `save_image` method.

    Parameters:
        asldata : ASLData
            The ASL data to be saved. This can be any Python object that is serializable by dill.
        fullpath : str
            The full path where the pickle file will be saved. The filename must end with '.pkl'.

    Raises:
    ValueError:
        If the provided filename does not end with '.pkl'.
    """
    if not fullpath.endswith('.pkl'):
        raise ValueError('Filename must be a pickle file (.pkl)')

    dill.dump(asldata, open(fullpath, 'wb'))


def load_asl_data(fullpath: str):
    """
    Load ASL data from a specified file path to ASLData object previously save
    on hard drive.

    This function uses the `dill` library to load and deserialize data from a
    file. Therefore, the file must have been saved using the `save_asl_data`.

    This method can be helpful when one wants to save the ASL data to a file
    and share it with others or use it in another script. The entire ASLData
    object will be loaded from the file, maintaining all the data and
    parameters described in the `ASLData` class.

    Examples:
        >>> from asltk.asldata import ASLData
        >>> asldata = ASLData(pcasl='./tests/files/pcasl_mte.nii.gz', m0='./tests/files/m0.nii.gz',ld_values=[1.8, 1.8, 1.8], pld_values=[1.8, 1.8, 1.8], te_values=[1.8, 1.8, 1.8])
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
        ...     temp_file_path = temp_file.name
        >>> save_asl_data(asldata, temp_file_path)
        >>> loaded_asldata = load_asl_data(temp_file_path)
        >>> loaded_asldata.get_ld()
        [1.8, 1.8, 1.8]
        >>> loaded_asldata('pcasl').shape
        (8, 7, 5, 35, 35)

    Parameters:
        fullpath (str): The full path to the file containing the serialized ASL data.

    Returns:
        ASLData: The deserialized ASL data object from the file.
    """
    _check_input_path(fullpath)
    return dill.load(open(fullpath, 'rb'))


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
        except OverflowError:   # pragma: no cover
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
        except (OverflowError, RuntimeError):   # pragma: no cover
            Sb = 0.0
            Scsf = 0.0

        mag_total[i] = Sb + Scsf

    return mag_total


def asl_model_multi_dw(
    b_values: list, A1: list, D1: float, A2: list, D2: float
):
    mag_total = np.zeros(len(b_values))

    for i in range(0, len(b_values)):
        try:
            mag_total[i] = A1 * math.exp(-b_values[i] * D1) + A2 * math.exp(
                -b_values[i] * D2
            )
        except (OverflowError, RuntimeError):   # pragma: no cover
            mag_total[i] = 0.0

    return mag_total
