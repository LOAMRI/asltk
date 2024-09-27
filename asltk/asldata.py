import os

import numpy as np
import SimpleITK as sitk


class ASLData:
    def __init__(
        self,
        **kwargs,
    ):
        """ASLData constructor

        The basic data need to represent a ASL data is the full path to load
        the image file, the Labeling Duration (LD) array and the Post-labeling
        Delay (PLD) array. Is none of those information is passed, a null
        ASLData object is created, which can be further been fed using the
        get/set methods.

        The constructor is generic for classic ASL data and also for multi-TE
        and Diffusion-Weighted (DW) ASL protocols. There is a specfic get/set
        method for TE/DW data. If TE/DW is not provided, then it is assumed as
        type `None` for those data properties. In order to informs the TE or DW
        values in the object instance, you can use the tags `te_values` or
        `dw_values` in the construction call

        Examples:
            By default, the LD and PLD arrays are indicated (as empty lists)

            >>> data = ASLData()
            >>> data.get_ld()
            []
            >>> data.get_pld()
            []

            >>> data = ASLData(te_values=[13.0,20.2,50.5,90.5,125.2])
            >>> data.get_te()
            [13.0, 20.2, 50.5, 90.5, 125.2]

            >>> data = ASLData(dw_values=[13.0,20.2,50.5,90.5,125.2])
            >>> data.get_dw()
            [13.0, 20.2, 50.5, 90.5, 125.2]

        Other parameters: Set the ASL data parameters
            pcasl (str, optional): The ASL data full path with filename.
            Defaults to ''.
            m0 (str, optional): The M0 data full path with filename. Defaults
            to ''.
            ld_values (list, optional): The LD values. Defaults to [].
            pld_values (list, optional): The PLD values. Defaults to [].
            te_values (list, optional): The TE values. Defaults to None.
            dw_values (list, optional): The DW values. Defaults to None.
        """
        self._asl_image = None
        self._m0_image = None
        self._parameters = {
            'ld': [],
            'pld': [],
            'te': None,
            'dw': None,
        }

        if kwargs.get('pcasl') is not None:
            self.load_image(kwargs.get('pcasl'), spec='pcasl')

        if kwargs.get('m0') is not None:
            self.load_image(kwargs.get('m0'), spec='m0')

        self._parameters['ld'] = (
            [] if kwargs.get('ld_values') is None else kwargs.get('ld_value')
        )
        self._parameters['pld'] = (
            []
            if kwargs.get('pld_values') is None
            else kwargs.get('pld_values')
        )
        if kwargs.get('te_values'):
            self._parameters['te'] = kwargs.get('te_values')
        if kwargs.get('dw_values'):
            self._parameters['dw'] = kwargs.get('dw_values')

    def load_image(self, full_path: str, spec: str):
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
            >>> data=ASLData()
            >>> data.load_image("./tests/files/t1-mri.nrrd", 'm0')
            >>> type(data('m0'))
            <class 'numpy.ndarray'>

        Returns:
            (numpy.array): The loaded image as the determined type
        """
        self._check_input_path(full_path)
        img = sitk.ReadImage(full_path)
        match spec:
            case 'm0':
                self._m0_image = sitk.GetArrayFromImage(img)
            case 'pcasl':
                self._asl_image = sitk.GetArrayFromImage(img)

    def save_image(self, full_path: str):
        """Save image to a file path.

        All the available image formats provided in the SimpleITK API can be
        used here.

        Args:
            full_path (str): Full absolute path with image file name provided.
        """
        sitk_img = sitk.GetImageFromArray(self._asl_image)
        sitk.WriteImage(sitk_img, full_path)

    def get_ld(self):
        """Obtain the LD array values"""
        return self._parameters['ld']

    def set_ld(self, ld_values: list):
        """Set the LD values.

        The proper way to inform the values here is using a list of int or
        float data. The total quantity of values depends on the image
        acquisition protocol.

        The list length for LD must be equal to PLD list length.

        Args:
            ld_values (list): The values to be adjusted for LD array
        """
        self._check_input_parameter(ld_values, 'LD')
        self._parameters['ld'] = ld_values

    def get_pld(self):
        """Obtain the PLD array values"""
        return self._parameters['pld']

    def set_pld(self, pld_values: list):
        """Set the PLD values.

        The proper way to inform the values here is using a list of int or
        float data. The total quantity of values depends on the image
        acquisition protocol.

        The list length for PLD must be equal to LD list length.

        Args:
            pld_values (list): The values to be adjusted for PLD array
        """
        self._check_input_parameter(pld_values, 'PLD')
        self._parameters['pld'] = pld_values

    def get_te(self):
        """Obtain the TE array values"""
        return self._parameters['te']

    def set_te(self, te_values: list):
        """Set the TE values.

        The proper way to inform the values here is using a list of int or
        float data. The total quantity of values depends on the image
        acquisition protocol.

        Args:
            te_values (list): The values to be adjusted for TE array
        """
        self._check_input_parameter(te_values, 'TE')
        self._parameters['te'] = te_values

    def get_dw(self):
        """Obtain the PLD array values"""
        return self._parameters['dw']

    def set_dw(self, dw_values: list):
        """Set the DW values.

        The proper way to inform the values here is using a list of int or
        float data. The total quantity of values depends on the image
        acquisition protocol.

        Args:
            dw_values (list): The values to be adjusted for DW array
        """
        self._check_input_parameter(dw_values, 'DW')
        self._parameters['dw'] = dw_values

    def __call__(self, spec: str):
        """Object caller to expose the image data.

        Examples:
            >>> data = ASLData(pcasl='./tests/files/t1-mri.nrrd')
            >>> type(data('pcasl'))
            <class 'numpy.ndarray'>

            >>> np.min(data('pcasl'))
            np.uint8(0)

        Returns:
            (numpy.ndarray): The data placed in the ASLData object
        """
        match spec:
            case 'pcasl':
                return self._asl_image
            case 'm0':
                return self._m0_image

    def _check_input_parameter(self, values, param_type):
        for v in values:
            if not isinstance(v, int) and not isinstance(v, float):
                raise ValueError(
                    f'{param_type} values is not a list of valid numbers.'
                )
            if v <= 0:
                raise ValueError(
                    f'{param_type} values must be postive non zero numbers.'
                )

    def _check_input_path(self, path):
        if not os.path.exists(path):
            raise ValueError('Image path is not valid or image not found.')
