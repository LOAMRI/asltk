import copy
import fnmatch
import os
import warnings
from typing import Union

import ants
import dill
import numpy as np
import SimpleITK as sitk
from ants.utils.sitk_to_ants import from_sitk
from bids import BIDSLayout
from rich import print

from asltk import AVAILABLE_IMAGE_FORMATS, BIDS_IMAGE_FORMATS


class ImageIO:
    def __init__(
        self, image_path: str = None, image_array: np.ndarray = None, **kwargs
    ):
        # Image parameters and objects
        self._image_path = image_path
        self._image_as_numpy = image_array
        self._image_as_sitk = None
        self._image_as_ants = None

        # BIDS standard parameters for saving/loading
        self._subject = kwargs.get('subject', None)
        self._session = kwargs.get('session', None)
        self._modality = kwargs.get('modality', None)
        self._suffix = kwargs.get('suffix', None)

        # Loading parameters
        self._average_m0 = kwargs.get('average_m0', False)
        self._verbose = kwargs.get('verbose', False)

        self._check_init_images()

        self.load_image()

    def __str__(self) -> str:
        # Section 1: Image parameters
        image_ext = (
            os.path.splitext(self._image_path)[-1]
            if self._image_path
            else 'N/A'
        )
        if self._image_as_sitk is not None:
            img_dim = self._image_as_sitk.GetDimension()
            img_spacing = self._image_as_sitk.GetSpacing()
            img_origin = self._image_as_sitk.GetOrigin()
        else:
            img_dim = img_spacing = img_origin = 'N/A'
        if self._image_as_numpy is not None:
            img_max = np.max(self._image_as_numpy)
            img_min = np.min(self._image_as_numpy)
            img_mean = np.mean(self._image_as_numpy)
            img_std = np.std(self._image_as_numpy)
        else:
            img_max = img_min = img_mean = img_std = 'N/A'

        image_section = [
            '[Image parameters]',
            f'  Path: {self._image_path}',
            f'  File extension: {image_ext}',
            f'  Dimension: {img_dim}',
            f'  Spacing: {img_spacing}',
            f'  Origin: {img_origin}',
            f'  Max value: {img_max}',
            f'  Min value: {img_min}',
            f'  Mean: {img_mean}',
            f'  Std: {img_std}',
        ]

        # Section 2: BIDS information
        bids_section = [
            '[BIDS information]',
            f'  Subject: {self._subject}',
            f'  Session: {self._session}',
            f'  Modality: {self._modality}',
            f'  Suffix: {self._suffix}',
        ]

        # Section 3: Loading parameters
        loading_section = [
            '[Loading parameters]',
            f'  average_m0: {self._average_m0}',
            f'  verbose: {self._verbose}',
        ]

        return '\n'.join(image_section + bids_section + loading_section)

    def set_image_path(self, image_path: str):
        check_path(image_path)
        self._image_path = image_path

    def get_image_path(self):
        return self._image_path

    # def set_full_representation(self, sitk_image: sitk.Image):
    #     check_image_properties(sitk_image, self._image_as_numpy)

    #     self._image_as_sitk = sitk_image
    #     self._image_as_ants = from_sitk(self._image_as_sitk)

    def get_as_sitk(self):
        self._check_image_representation('sitk')

        return copy.deepcopy(self._image_as_sitk)

    def get_as_ants(self):
        self._check_image_representation('ants')

        return self._image_as_ants.clone()

    def get_as_numpy(self):
        self._check_image_representation('numpy')

        return self._image_as_numpy.copy()

    def load_image(self):
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
            Load a single image file directly:
            >>> data = ImageIO("./tests/files/pcasl_mte.nii.gz").get_as_numpy()
            >>> type(data)
            <class 'numpy.ndarray'>
            >>> data.shape  # Example: 5D ASL data
            (8, 7, 5, 35, 35)

            Load M0 reference image:
            >>> m0_data = ImageIO("./tests/files/m0.nii.gz").get_as_numpy()
            >>> m0_data.shape  # Example: 3D reference image
            (5, 35, 35)

            Load from BIDS directory (automatic detection):
            >>> data = ImageIO("./tests/files/bids-example/asl001").get_as_numpy()
            >>> type(data)
            <class 'numpy.ndarray'>

            Load specific BIDS data with detailed parameters:
            >>> data = ImageIO("./tests/files/bids-example/asl001", subject='Sub103', suffix='asl').get_as_numpy()
            >>> type(data)
            <class 'numpy.ndarray'>

            # Load NRRD format
            >>> nrrd_data = ImageIO("./tests/files/t1-mri.nrrd").get_as_numpy()
            >>> type(nrrd_data)
            <class 'numpy.ndarray'>

        Returns:
            numpy.ndarray: The loaded image array.
        """

        if self._image_path is not None:
            check_path(self._image_path)

            if self._image_path.endswith(AVAILABLE_IMAGE_FORMATS):
                # If the full path is a file, then load the image directly
                self._image_as_sitk = sitk.ReadImage(self._image_path)
                self._image_as_numpy = sitk.GetArrayFromImage(
                    self._image_as_sitk
                )
                if self._image_as_numpy.ndim <= 3:
                    self._image_as_ants = from_sitk(self._image_as_sitk)
            else:
                # If the full path is a directory, then use BIDSLayout to find the file
                selected_file = self._get_file_from_folder_layout()
                self._image_as_sitk = sitk.ReadImage(selected_file)
                self._image_as_numpy = sitk.GetArrayFromImage(
                    self._image_as_sitk
                )
                if self._image_as_numpy.ndim <= 3:
                    self._image_as_ants = from_sitk(self._image_as_sitk)
        elif self._image_as_numpy is not None:
            # If the image is already provided as a numpy array, convert it to SimpleITK
            self._image_as_sitk = sitk.GetImageFromArray(self._image_as_numpy)
            if self._image_as_numpy.ndim <= 3:
                self._image_as_ants = from_sitk(self._image_as_sitk)
        else:
            raise ValueError(
                'Either image_path or image_array must be provided to load the image.'
            )

        # Check if there are additional parameters
        if self._average_m0:
            # If average_m0 is True, then average the M0 image
            if self._image_as_numpy.ndim > 3:
                avg_img = np.mean(self._image_as_numpy, axis=0)
                self.update_image_data(avg_img)

    def update_image_data(self, new_array: np.ndarray):
        """
        Update the image data with a new numpy array, preserving the original image metadata.

        Args:
            new_array (np.ndarray): The new image data array. Must match the shape of the original image.
        """
        # Create new SimpleITK image from array
        new_sitk_img = sitk.GetImageFromArray(new_array)
        # Copy metadata
        # Copy all metadata from the original image
        new_sitk_img.SetOrigin(self._image_as_sitk.GetOrigin())
        new_sitk_img.SetSpacing(self._image_as_sitk.GetSpacing())
        new_sitk_img.SetDirection(self._image_as_sitk.GetDirection())
        # Copy all key-value metadata
        for k in self._image_as_sitk.GetMetaDataKeys():
            new_sitk_img.SetMetaData(k, self._image_as_sitk.GetMetaData(k))

        # Update internal representations
        self._image_as_numpy = new_array
        self._image_as_sitk = new_sitk_img
        if new_array.ndim <= 3:
            # ANTsPy does not support higher dimension images, so we skip conversion for lower than 3D arrays
            self._image_as_ants = from_sitk(new_sitk_img)

    def save_image(
        self,
        full_path: str = None,
        *,
        bids_root: str = None,
        subject: str = None,
        session: str = None,
        **kwargs,
    ):
        """
        Save an image to a file path using SimpleITK.

        All available image formats provided in the SimpleITK API can be used here. Supported formats include: .nii, .nii.gz, .nrrd, .mha, .tif, and others.

        Note:
            If the file extension is not recognized by SimpleITK, an error will be raised.
            The input array should be 2D, 3D, or 4D. For 4D arrays, only the first volume may be saved unless handled explicitly.

        Args:
            img (np.ndarray): The image array to be saved. Can be 2D, 3D, or 4D.
            full_path (str): Full absolute path with image file name provided.
            bids_root (str): Optional BIDS root directory to save in BIDS structure.
            subject (str): Subject ID for BIDS saving.
            session (str): Optional session ID for BIDS saving.

        Examples:
            Save an image using a direct file path:
            >>> import tempfile
            >>> import numpy as np
            >>> img = np.random.rand(10, 10, 10)
            >>> with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
            ...     save_image(img, f.name)

            Save an image using BIDS structure:
            >>> import tempfile
            >>> img = np.random.rand(10, 10, 10)
            >>> with tempfile.TemporaryDirectory() as temp_dir:
            ...     save_image(img, bids_root=temp_dir, subject='001', session='01')

            Save processed ASL results:
            >>> from asltk.asldata import ASLData
            >>> asl_data = ASLData(pcasl='./tests/files/pcasl_mte.nii.gz', m0='./tests/files/m0.nii.gz')
            >>> processed_img = asl_data('pcasl')[0]  # Get first volume
            >>> import tempfile
            >>> with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
            ...     save_image(processed_img, f.name)

        Raises:
            ValueError: If neither full_path nor (bids_root + subject) are provided.
            RuntimeError: If the file extension is not recognized by SimpleITK.
        """
        if bids_root and subject:
            full_path = _make_bids_path(bids_root, subject, session)

        if not full_path:
            raise ValueError(
                'Either full_path or bids_root + subject must be provided.'
            )

        # sitk_img = sitk.GetImageFromArray(img)
        useCompression = kwargs.get('useCompression', False)
        compressionLevel = kwargs.get('compressionLevel', -1)
        compressor = kwargs.get('compressor', '')
        sitk.WriteImage(
            self._image_as_sitk,
            full_path,
            useCompression=useCompression,
            compressionLevel=compressionLevel,
            compressor=compressor,
        )

    def _check_image_representation(self, representation):
        if representation == 'sitk' and self._image_as_sitk is None:
            raise ValueError(
                'Image is not loaded as SimpleITK. Please load the image first.'
            )
        elif representation == 'ants' and self._image_as_ants is None:
            raise ValueError(
                'Image is not loaded as ANTsPy. Please load the image first.'
            )
        elif representation == 'numpy' and self._image_as_numpy is None:
            raise ValueError(
                'Image is not loaded as numpy array. Please load the image first.'
            )

    def _get_file_from_folder_layout(self):
        selected_file = None
        layout = BIDSLayout(self._image_path)
        if all(
            param is None
            for param in [
                self._subject,
                self._session,
                self._modality,
                self._suffix,
            ]
        ):
            for root, _, files in os.walk(self._image_path):
                for file in files:
                    if '_asl' in file and file.endswith(BIDS_IMAGE_FORMATS):
                        selected_file = os.path.join(root, file)
        else:
            layout_files = layout.files.keys()
            matching_files = []
            for f in layout_files:
                search_pattern = ''
                if self._subject:
                    search_pattern = f'*sub-*{self._subject}*'
                if self._session:
                    search_pattern += search_pattern + f'*ses-*{self._session}'
                if self._modality:
                    search_pattern += search_pattern + f'*{self._modality}*'
                if self._suffix:
                    search_pattern += search_pattern + f'*{self._suffix}*'

                if fnmatch.fnmatch(f, search_pattern) and f.endswith(
                    BIDS_IMAGE_FORMATS
                ):
                    matching_files.append(f)

            if not matching_files:
                raise FileNotFoundError(
                    f'ASL image file is missing for subject {self._subject} in directory {self._image_path}'
                )
            selected_file = matching_files[0]

        return selected_file

    def _check_init_images(self):
        """
        Check if the image is initialized correctly.
        If both image_path and image_array are None, raise an error.
        """

        if self._image_path is None and self._image_as_numpy is None:
            raise ValueError(
                'Either image_path or image_array must be provided to initialize the ImageIO object.'
            )
        if self._image_path is not None and self._image_as_numpy is not None:
            raise ValueError(
                'Both image_path and image_array are provided. Please provide only one.'
            )
        if self._image_path is None and self._image_as_numpy is not None:
            warnings.warn(
                'image_array is provided but image_path is not set. The image will be loaded as a numpy array only and the image metadata will be set as default. For complex image processing it is better to provide the image_path instead.',
            )


def check_image_properties(
    first_image: Union[sitk.Image, np.ndarray, ants.ANTsImage, ImageIO],
    ref_image: ImageIO,
):
    # Check the image size, dimension, spacing and all the properties to see if the first_image is equal to ref_image
    if not isinstance(ref_image, ImageIO):
        raise TypeError('Reference image must be a ImageIO object')

    if isinstance(first_image, sitk.Image):
        # Compare with ref_image's sitk representation
        ref_sitk = ref_image._image_as_sitk

        if first_image.GetSize() != ref_sitk.GetSize():
            raise ValueError('Image size mismatch.')
        if first_image.GetSpacing() != ref_sitk.GetSpacing():
            raise ValueError('Image spacing mismatch.')
        if first_image.GetOrigin() != ref_sitk.GetOrigin():
            raise ValueError('Image origin mismatch.')
        if first_image.GetDirection() != ref_sitk.GetDirection():
            raise ValueError('Image direction mismatch.')

    elif isinstance(first_image, np.ndarray):
        ref_np = ref_image._image_as_numpy

        if first_image.shape != ref_np.shape:
            raise ValueError('Numpy array shape mismatch.')
        if first_image.dtype != ref_np.dtype:
            raise ValueError('Numpy array dtype mismatch.')

        warnings.warn(
            'Numpy arrays does not has spacing and origin image information.'
        )

    elif isinstance(first_image, ants.ANTsImage):
        ref_ants = (
            ref_image._image_as_ants
            if isinstance(ref_image, ImageIO)
            else ref_image
        )
        if not isinstance(ref_ants, ants.ANTsImage):
            raise ValueError('Reference image is not an ANTsPy image.')
        if first_image.shape != ref_ants.shape:
            raise ValueError('ANTs image shape mismatch.')
        if not np.allclose(first_image.spacing, ref_ants.spacing):
            raise ValueError('ANTs image spacing mismatch.')
        if not np.allclose(first_image.origin, ref_ants.origin):
            raise ValueError('ANTs image origin mismatch.')
        if not np.allclose(first_image.direction, ref_ants.direction):
            raise ValueError('ANTs image direction mismatch.')

    elif isinstance(first_image, ImageIO):
        # Recursively check using numpy representation
        check_image_properties(first_image.get_as_sitk(), ref_image)
    else:
        raise TypeError('Unsupported image type for comparison.')


def clone_image(source: ImageIO, include_path: bool = False):
    if not isinstance(source, ImageIO):
        raise TypeError('Source image must be a ImageIO object')

    cloned = copy.deepcopy(source)
    if not include_path:
        cloned._image_path = None

    return cloned


def check_path(path: str):
    if path is None:
        raise ValueError(
            'Image path is not set. Please set the image path first.'
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f'The file {path} does not exist.')


def _make_bids_path(
    bids_root, subject, session=None, suffix='asl', extension='.nii.gz'
):
    subj_dir = f'sub-{subject}'
    ses_dir = f'ses-{session}' if session else None
    modality_dir = 'asl'

    if ses_dir:
        out_dir = os.path.join(bids_root, subj_dir, ses_dir, modality_dir)
    else:
        out_dir = os.path.join(bids_root, subj_dir, modality_dir)

    os.makedirs(out_dir, exist_ok=True)

    filename = f'sub-{subject}'
    if session:
        filename += f'_ses-{session}'
    filename += f'_{suffix}{extension}'

    return os.path.join(out_dir, filename)


def save_asl_data(
    asldata,
    fullpath: str = None,
    *,
    bids_root: str = None,
    subject: str = None,
    session: str = None,
):
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
    if bids_root and subject:
        fullpath = _make_bids_path(
            bids_root, subject, session, suffix='asl', extension='.pkl'
        )

    if not fullpath:
        raise ValueError(
            'Either fullpath or bids_root + subject must be provided.'
        )

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
    check_path(fullpath)
    return dill.load(open(fullpath, 'rb'))
