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
    """ImageIO is the base class in `asltk` for loading, manipulating,
    and saving ASL images.

    The basic functionality includes:
        - Loading images from a file path or a numpy array.
        - Converting images to different representations (SimpleITK, ANTsPy, numpy).
        - Saving images to a file path in various formats.
    """

    def __init__(
        self, image_path: str = None, image_array: np.ndarray = None, **kwargs
    ):
        """The constructor initializes the ImageIO object with an image path or a numpy array.

        It is needed to provide either an image path or a numpy array to load the image.
        If both are provided, an error will be raised because it is ambiguous which one to use.

        Note:
            - If `image_path` is provided, the image will be loaded from the file.
            - If `image_array` is provided, the image will be loaded as a numpy array.
            - If both are provided, an error will be raised.
            - If neither is provided, an error will be raised.

        Important:
            The image path should be a valid file path to an image file or a directory containing BIDS-compliant images.
            It is also recommended to provide the image path for complex image processing, as it allows to preserve the image metadata and properties, as seen for the SimpleITK and ANTsPy representations.

        Only the SimpleITK and Numpy representations are availble to manipulate higher dimensional images (4D, 5D, etc.).
        The ANTsPy representation is limited up to 3D images, mainly due to the specificity to image normalization applications.

        Args:
            image_path (str, optional): The file path to the image. Defaults to None.
            image_array (np.ndarray, optional): The image as a numpy array. Defaults to None.
            average_m0 (bool, optional): If True, averages the M0 image if it is provided. Defaults to False.
            verbose (bool, optional): If True, prints additional information during loading. Defaults to False
        """
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

        if kwargs.get('verbose', False):
            print(
                f'[bold green]ImageIO initialized with path:[/bold green] {self._image_path}'
            )
            print(self)

    def __str__(self) -> str:
        """Returns a string representation of the ImageIO object.

        Returns:
            str: A summary of the image parameters, BIDS information, and loading parameters.
        """
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
        """Set the image path for loading.

        Args:
            image_path (str): Path to the image file.
        """
        check_path(image_path)
        self._image_path = image_path

    def get_image_path(self):
        """Get the image path for loading.

        Returns:
            str: Path to the image file.
        """
        return self._image_path

    def get_as_sitk(self):
        """Get the image as a SimpleITK image object.

        Important:
            The methods returns a copy of the SimpleITK image object.
            This is to ensure that the original image is not modified unintentionally.

        Returns:
            SimpleITK.Image: The image as a SimpleITK image object.
        """
        self._check_image_representation('sitk')

        return copy.deepcopy(self._image_as_sitk)

    def get_as_ants(self):
        """Get the image as an ANTsPy image object.

        Important:
            The methods returns a copy of the ANTsPy image object.
            This is to ensure that the original image is not modified unintentionally.

        Returns:
            ants.image: The image as an ANTsPy image object.
        """
        self._check_image_representation('ants')

        return self._image_as_ants.clone()

    def get_as_numpy(self):
        """Get the image as a NumPy array.

        Important:
            The methods returns a copy of the NumPy array.
            This is to ensure that the original image is not modified unintentionally.
            Also, the image representation as numpy array does not preserve the image metadata, such as spacing, origin, and direction.
            For a complete image representation, use the SimpleITK or ANTsPy representations.

        Returns:
            numpy.ndarray: The image as a NumPy array.
        """
        self._check_image_representation('numpy')

        return self._image_as_numpy.copy()

    def load_image(self):
        """
        Load an image file from a BIDS directory or file using the SimpleITK and ANTsPy representation (if applicable).

        The output is allocated internaly to a ImageIO object that contains up to three image representations: a
         SimpleITK image, a numpy array and (if applicable) a ANTsPy image.

        Note:
            - The general image loading is done using SimpleITK, which supports a wide range of image formats.
            - The image is loaded as a SimpleITK image, and then converted to a numpy array.
            - If the image is 3D or lower, it is also converted to an ANTsPy image.

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

        The information passed to the ImageIO constructor is used to load the image.

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
            ImageIO: The loaded image as a ImageIO object.
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
            # is_vector = True
            # if self._image_as_numpy.ndim > 3:
            #     is_vector = False

            self._image_as_sitk = sitk.GetImageFromArray(
                self._image_as_numpy, isVector=False
            )
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
                self.update_image_data(avg_img, enforce_new_dimension=True)

    def update_image_spacing(self, new_spacing: tuple):
        """
        Update the image spacing with a new tuple, preserving the original image metadata.

        Important:
            - The new spacing must be a tuple of the same length as the original image dimension.

        Args:
            new_spacing (tuple): The new spacing for the image.
        """
        if not isinstance(new_spacing, tuple):
            raise TypeError('new_spacing must be a tuple.')

        # Update spacing in SimpleITK image
        self._image_as_sitk.SetSpacing(new_spacing)

        # Update internal numpy representation
        self._image_as_numpy = sitk.GetArrayFromImage(self._image_as_sitk)
        if self._image_as_numpy.ndim <= 3:
            self._image_as_ants = from_sitk(self._image_as_sitk)

    def update_image_origin(self, new_origin: tuple):
        """
        Update the image origin with a new tuple, preserving the original image metadata.

        Important:
            - The new origin must be a tuple of the same length as the original image dimension.

        Args:
            new_origin (tuple): The new origin for the image.
        """
        if not isinstance(new_origin, tuple):
            raise TypeError('new_origin must be a tuple.')

        # Update origin in SimpleITK image
        self._image_as_sitk.SetOrigin(new_origin)

        # Update internal numpy representation
        self._image_as_numpy = sitk.GetArrayFromImage(self._image_as_sitk)
        if self._image_as_numpy.ndim <= 3:
            self._image_as_ants = from_sitk(self._image_as_sitk)

    def update_image_direction(self, new_direction: tuple):
        """
        Update the image direction with a new tuple, preserving the original image metadata.

        Important:
            - The new direction must be a tuple of the same length as the original image dimension.

        Args:
            new_direction (tuple): The new direction for the image.
        """
        if not isinstance(new_direction, tuple):
            raise TypeError('new_direction must be a tuple.')

        # Update direction in SimpleITK image
        self._image_as_sitk.SetDirection(new_direction)

        # Update internal numpy representation
        self._image_as_numpy = sitk.GetArrayFromImage(self._image_as_sitk)
        if self._image_as_numpy.ndim <= 3:
            self._image_as_ants = from_sitk(self._image_as_sitk)

    def update_image_data(
        self, new_array: np.ndarray, enforce_new_dimension=False
    ):
        """
        Update the image data with a new numpy array, preserving the original image metadata.

        This is particularly useful for updating the image data after processing or when new data is available.
        Hence, it allows to change the image data without losing the original metadata such as spacing, origin, and direction.

        Another application for this method is to create a new image using a processed numpy array and then copy the metadata from the original image that was loaded using a file path, which contains the original metadata.

        Examples:
            >>> import numpy as np
            >>> array = np.random.rand(5, 35, 35)
            >>> image1 = ImageIO(image_array=array)# Example 3D image from a numpy array (without metadata)
            >>> image2 = ImageIO(image_path="./tests/files/m0.nii.gz") # Example 3D image with metadata
            >>> full_image = ImageIO(image_path="./tests/files/m0.nii.gz") # Example 3D image with metadata

            Both images has the same shape, so we can update the image data:
            >>> image1.get_as_numpy().shape == image2.get_as_numpy().shape
            True

            >>> image2.update_image_data(image1.get_as_numpy())

            Now the `image2` has the same data as `image1`, but retains its original metadata.

        Important:
            - The new array must match the shape of the original image unless `enforce_new_dimension` is set to True.
            - If `enforce_new_dimension` is True, the new array can have a different shape than the original image, but
            it will be assumed the first dimensions to get averaged.

        Args:
            new_array (np.ndarray): The new image data array. Must match the shape of the original image.
            enforce_new_dimension (bool): If True, allows the new array to have a different shape than the original image.

        """
        if not isinstance(new_array, np.ndarray):
            raise TypeError('new_array must be a numpy array.')
        if new_array.shape != self._image_as_numpy.shape:
            if not enforce_new_dimension:
                raise ValueError(
                    'new_array must match the shape of the original image.'
                )

        # Get the dimension difference
        dim_diff = self._image_as_numpy.ndim - new_array.ndim

        if dim_diff < 0 or abs(dim_diff) >= 2:
            raise TypeError(
                'The new array is too much different from the original image. '
                'The new array must have the same number of dimensions as the original image or at most one dimension less.'
            )

        # Create new SimpleITK image from array
        new_sitk_img = sitk.GetImageFromArray(new_array, isVector=False)

        if dim_diff != 0:
            base_origin = self._image_as_sitk.GetOrigin()[:3]
            base_spacing = self._image_as_sitk.GetSpacing()[:3]
            base_direction = tuple(
                np.array(self._image_as_sitk.GetDirection())
                .reshape(self._image_as_numpy.ndim, self._image_as_numpy.ndim)[
                    :3, :3
                ]
                .flatten()
            )
        else:
            base_origin = self._image_as_sitk.GetOrigin()
            base_spacing = self._image_as_sitk.GetSpacing()
            base_direction = self._image_as_sitk.GetDirection()

        # Copy metadata
        # Copy all metadata from the original image
        new_sitk_img.SetOrigin(base_origin)
        new_sitk_img.SetSpacing(base_spacing)
        new_sitk_img.SetDirection(base_direction)
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
        Save the current image to a file path using SimpleITK.

        All available image formats provided in the SimpleITK API can be used here. Supported formats include: .nii, .nii.gz, .nrrd, .mha, .tif, and others.

        Note:
            If the file extension is not recognized by SimpleITK, an error will be raised.
            The image array should be 2D, 3D, or 4D. For 4D arrays, only the first volume may be saved unless handled explicitly.

        Args:
            full_path (str): Full absolute path with image file name provided.
            bids_root (str): Optional BIDS root directory to save in BIDS structure.
            subject (str): Subject ID for BIDS saving.
            session (str): Optional session ID for BIDS saving.

        Examples:
            Save an image using a direct file path:
            >>> import tempfile
            >>> from asltk.utils.io import ImageIO
            >>> import numpy as np
            >>> img = np.random.rand(10, 10, 10)
            >>> io = ImageIO(image_array=img)
            >>> with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
            ...     io.save_image(f.name)

            Save an image using BIDS structure:
            >>> import tempfile
            >>> from asltk.utils.io import ImageIO
            >>> import numpy as np
            >>> img = np.random.rand(10, 10, 10)
            >>> io = ImageIO(image_array=img)
            >>> with tempfile.TemporaryDirectory() as temp_dir:
            ...     io.save_image(bids_root=temp_dir, subject='001', session='01')

            Save processed ASL results:
            >>> from asltk.asldata import ASLData
            >>> from asltk.utils.io import ImageIO
            >>> asl_data = ASLData(pcasl='./tests/files/pcasl_mte.nii.gz', m0='./tests/files/m0.nii.gz')
            >>> processed_img = asl_data('pcasl').get_as_numpy()[0]  # Get first volume
            >>> io = ImageIO(image_array=processed_img)
            >>> import tempfile
            >>> with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
            ...     io.save_image(f.name)

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

        if not os.path.exists(os.path.dirname(full_path)):
            raise FileNotFoundError(
                f'The directory of the full path {full_path} does not exist.'
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
    """Check the properties of two images to ensure they are compatible.

    The first image can be a SimpleITK image, a numpy array, an ANTsPy image, or an ImageIO object.
    The reference image must be an ImageIO object.

    This function checks the size, spacing, origin, and direction of the first image against the reference image.

    Args:
        first_image (Union[sitk.Image, np.ndarray, ants.ANTsImage, ImageIO]): The first image to check.
        ref_image (ImageIO): The reference image to compare against.

    Raises:
        TypeError: If the reference image is not an ImageIO object.
        ValueError: If the image properties (size, spacing, origin, direction) do not match.
        ValueError: If the image properties (size, spacing, origin, direction) do not match.
    """
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
    """Clone an image getting a deep copy.

    All the image properties are copied, including the image path if `include_path` is True.

    Tip:
        This a useful method to create a copy of an image for processing without modifying the original image.
        Also, after making a clone, you can modify the image properties without affecting the original image.
        The image array representation can be modified, but the original image metadata will remain unchanged,
        however the `update_image_data` method can be used to update the image data while preserving the original metadata.

    Args:
        source (ImageIO): The source image to clone.
        include_path (bool, optional): Whether to include the image path in the clone. Defaults to False.

    Raises:
        TypeError: If the source image is not an ImageIO object.

    Returns:
        ImageIO: The cloned image.
    """
    if not isinstance(source, ImageIO):
        raise TypeError('Source image must be a ImageIO object')

    cloned = copy.deepcopy(source)
    if not include_path:
        cloned._image_path = None

    return cloned


def check_path(path: str):
    """Check if the image path is valid.

    Args:
        path (str): The image path to check.

    Raises:
        ValueError: If the image path is not set.
        FileNotFoundError: If the image file does not exist.
    """
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
        >>> loaded_asldata('pcasl').get_as_numpy().shape
        (8, 7, 5, 35, 35)
    """
    check_path(fullpath)
    return dill.load(open(fullpath, 'rb'))
