import fnmatch
import os

import dill
import numpy as np
import SimpleITK as sitk
from bids import BIDSLayout

from asltk import AVAILABLE_IMAGE_FORMATS, BIDS_IMAGE_FORMATS


def load_image(
    full_path: str,
    subject: str = None,
    session: str = None,
    modality: str = None,
    suffix: str = None,
    **kwargs,
):
    """
    Load an image file from a BIDS directory or file using the SimpleITK API.

    The output is always a numpy array, converted from the SimpleITK image object.

    Supported image formats include: .nii, .nii.gz, .nrrd, .mha, .tif, and other formats supported by SimpleITK.

    Note:
        - The default values for `modality` and `suffix` are None. If not provided, the function will search for the first matching ASL image in the directory.
        - If `full_path` is a file, it is loaded directly. If it is a directory, the function searches for a BIDS-compliant image using the provided parameters.
        - If both a file and a BIDS directory are provided, the file takes precedence.

    Tip:
        To validate your BIDS structure, use the `bids-validator` tool: https://bids-standard.github.io/bids-validator/
        For more details about ASL BIDS structure, see: https://bids-specification.readthedocs.io/en/latest

    Note:
        The image file is assumed to be an ASL subtract image (control-label). If not, use helper functions in `asltk.utils` to create one.

    Args:
        full_path (str): Path to the image file or BIDS directory.
        subject (str, optional): Subject identifier. Defaults to None.
        session (str, optional): Session identifier. Defaults to None.
        modality (str, optional): Modality folder name. Defaults to None.
        suffix (str, optional): Suffix of the file to load. Defaults to None.

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
        numpy.ndarray: The loaded image array.
    """
    _check_input_path(full_path)
    img = None

    if full_path.endswith(AVAILABLE_IMAGE_FORMATS):
        # If the full path is a file, then load the image directly
        img = sitk.GetArrayFromImage(sitk.ReadImage(full_path))
    else:
        # If the full path is a directory, then use BIDSLayout to find the file
        selected_file = _get_file_from_folder_layout(
            full_path, subject, session, modality, suffix
        )
        img = sitk.GetArrayFromImage(sitk.ReadImage(selected_file))

    # Check if there are additional parameters
    if kwargs.get('average_m0', False):
        # If average_m0 is True, then average the M0 image
        if img.ndim > 3:
            img = np.mean(img, axis=0)

    return img


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
    Save ASL data to a pickle file using dill serialization.

    This method saves the ASL data to a pickle file using the dill library. All
    the ASL data will be saved in a single file. After the file is saved, it
    can be loaded using the `load_asl_data` method.

    Note:
        This method only accepts the ASLData object as input. If you want to
        save an image, use the `save_image` method.
        The file is serialized with dill, which supports more Python objects than standard pickle. However, files saved with dill may not be compatible with standard pickle, especially for custom classes.

    Parameters:
        asldata : ASLData
            The ASL data to be saved. This can be any Python object that is serializable by dill.
        fullpath : str
            The full path where the pickle file will be saved. The filename must end with '.pkl'.

    Examples:
        >>> from asltk.asldata import ASLData
        >>> asldata = ASLData(pcasl='./tests/files/pcasl_mte.nii.gz', m0='./tests/files/m0.nii.gz',ld_values=[1.8, 1.8, 1.8], pld_values=[1.8, 1.8, 1.8], te_values=[1.8, 1.8, 1.8])
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
        ...     temp_file_path = temp_file.name
        >>> save_asl_data(asldata, temp_file_path)

    Raises:
        ValueError: If the provided filename does not end with '.pkl'.
    """
    if not fullpath.endswith('.pkl'):
        raise ValueError('Filename must be a pickle file (.pkl)')

    dill.dump(asldata, open(fullpath, 'wb'))


def load_asl_data(fullpath: str):
    """
    Load ASL data from a specified file path to an ASLData object previously saved on disk.

    This function uses the `dill` library to load and deserialize data from a
    file. Therefore, the file must have been saved using the `save_asl_data` function.

    Note:
        The file must have been saved with dill. Files saved with dill may not be compatible with standard pickle, especially for custom classes.

    Parameters:
        fullpath (str): The full path to the file containing the serialized ASL data.

    Returns:
        ASLData: The deserialized ASL data object from the file.

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
    """
    _check_input_path(fullpath)
    return dill.load(open(fullpath, 'rb'))


def _check_input_path(full_path: str):
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'The file {full_path} does not exist.')


def _get_file_from_folder_layout(
    full_path: str,
    subject: str = None,
    session: str = None,
    modality: str = None,
    suffix: str = None,
):
    selected_file = None
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

    return selected_file
