import numpy as np
import SimpleITK as sitk


class asldata:
    image = None
    parameters = {
        'ld': [],
        'pld': [],
        'te': None,
        'dw': None,
    }

    def load_image(self, full_path: str):
        """Load an image file path using the standard SimpleITK API.

        For more details about the image formatts accepted, check the official documentation at: https://simpleitk.org/

        Args:
            full_path (str): Absolute path for the image file

        Returns:
            numpy.array: The loaded image
        """
        img = sitk.ReadImage(full_path)
        self.image = sitk.GetArrayFromImage(img)

    def save_image(self, full_path: str):
        sitk_img = sitk.GetImageFromArray(self.image)
        sitk.WriteImage(sitk_img, full_path)

    def __init__(self):
        pass

    def get_ld(self):
        return self.parameters['ld']

    def set_ld(self, ld_values: list):
        self._check_input_parameter(ld_values, 'LD')
        self.parameters['ld'] = ld_values

    def get_pld(self):
        return self.parameters['pld']

    def set_pld(self, pld_values: list):
        self._check_input_parameter(pld_values, 'PLD')
        self.parameters['pld'] = pld_values

    def get_te(self):
        return self.parameters['te']

    def set_te(self, te_values: list):
        self._check_input_parameter(te_values, 'TE')
        self.parameters['te'] = te_values

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
