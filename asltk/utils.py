import fnmatch
import os

import dill
import numpy as np
import SimpleITK as sitk
from bids import BIDSLayout

from asltk import AVAILABLE_IMAGE_FORMATS, BIDS_IMAGE_FORMATS


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
        `suffix` are used. Otherwise, the method will search for the
        first image file found in the BIDS directory that can be an estimate
        ASL image. If the file full path is provided, then the method will
        load the image directly.

    Tip:
        To be sure that the input BIDS structure is correct, use the
        `bids-validator` tool to check the BIDS structure. See more details at:
        https://bids-standard.github.io/bids-validator/. For more deteils about
        ASL BIDS structure, check the official documentation at:
        https://bids-specification.readthedocs.io/en/latest

    Note:
        The image file is assumed to be an ASL subtract image, that is an image
        that has the subtraction of the control and label images. If the input
        image is not in this format, then the user can use a set of helper
        functions to create the ASL subtract image. See the `asltk.utils`
        module for more details.

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
    selected_file = _get_file_from_folder_layout(
        full_path, subject, session, modality, suffix
    )

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
