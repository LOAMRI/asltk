import warnings

import numpy as np
from rich.progress import track
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.mri_parameters import MRIParameters
from asltk.utils import asl_model_buxton

# TODO Opcao para aplicar filtro no mapa de saida (Gauss, etc)
# TODO Brain mask como input ou opção de processamento?


class CBFMapping(MRIParameters):
    def __init__(self, asl_data: ASLData) -> None:
        """Basic CBFMapping constructor.

        Notes:
            The ASLData is the base data used in the object constructor.
            In order to create the CBF map correctly, a proper ASLData must be
            provided. Check whether the ASLData given as input is defined
            correctly

        Examples:
            The default MRIParameters are used as default in the object
            constructor
            >>> asl_data = ASLData(pcasl='./tests/files/pcasl.nii.gz',m0='./tests/files/m0.nii.gz')
            >>> cbf = CBFMapping(asl_data)
            >>> cbf.get_constant('T1csf')
            1400.0

            If the user want to change the MRIParameter value, for a specific
            object, one can change it directly:
            >>> cbf.set_constant(1600.0, 'T1csf')
            >>> cbf.get_constant('T1csf')
            1600.0
            >>> default_param = MRIParameters()
            >>> default_param.get_constant('T1csf')
            1400.0

        Args:
            asl_data (ASLData): The ASL data object (ASLData)
        """
        super().__init__()
        self._asl_data = asl_data
        self._brain_mask = self._create_basic_mask()
        self._cbf_map = np.zeros(self._asl_data('m0').shape)
        self._att_map = np.zeros(self._asl_data('m0').shape)

    def set_brain_mask(self, brain_mask: np.ndarray, label: int = 1):
        """Defines whether a brain a mask is applied to the CBFMapping
        calculation

        A image mask is simply an image that defines the voxels where the ASL
        calculation should be made. Basically any integer value can be used as
        proper label mask.

        A most common approach is to use a binary image (zeros for background
        and 1 for the brain tissues). Anyway, the default behavior of the
        method can transform a integer-pixel values image to a binary mask with
        the `label` parameter provided by the user

        Args:
            brain_mask (np.ndarray): The image representing the brain mask label (int, optional): The label value used to define the foreground tissue (brain). Defaults to 1.
        """
        # TODO Add flag to create new mask using fsl.bet
        self._check_mask_values(brain_mask, label)

        binary_mask = (brain_mask == label).astype(np.uint8) * label
        self._brain_mask = binary_mask

    def create_map(
        self,
        ub: list = [1.0, 5000.0],
        lb: list = [0.0, 0.0],
        par0: list = [1e-5, 1000],
    ):
        """Create the CBF and also ATT maps

        Note:
            By default the ATT map is already calculated using the same Buxton
            formalism. Once the CBFMapping.create_map() method is called, both
            CBF and ATT maps are given in the output.

        Note:
            The CBF maps is given in two formats: the original pixel scale,
            resulted from the non-linear Buxton model fitting, and also
            a normalized version with the correct units of mL/100 g/min. In the
            output dictionary the user can select the 'cbf' and 'cbf_norm'
            options

        Args:
            ub (list, optional): The upper limit values. Defaults to [1.0, 5000.0].
            lb (list, optional): The lower limit values. Defaults to [0.0, 0.0].
            par0 (list, optional): The initial guess parameter for non-linear fitting. Defaults to [1e-5, 1000].

        Returns:
            (dict): A dictionary with 'cbf', 'att' and 'cbf_norm'
        """

        BuxtonX = [
            self._asl_data.get_ld(),
            self._asl_data.get_pld(),
        ]   # x data for the Buxton model

        x_axis = self._asl_data('m0').shape[2]   # height
        y_axis = self._asl_data('m0').shape[1]   # width
        z_axis = self._asl_data('m0').shape[0]   # depth

        for i in track(range(x_axis), description='[green]CBF processing...'):
            for j in range(y_axis):
                for k in range(z_axis):
                    if self._brain_mask[k, j, i] != 0:
                        m0_px = self._asl_data('m0')[k, j, i]

                        def mod_buxton(Xdata, par1, par2):
                            return asl_model_buxton(
                                Xdata[0], Xdata[1], m0_px, par1, par2
                            )

                        # TODO precisa achar uma forma generica para saber estrutura asl_data (TE,PLD,3DVOL)... como o codigo irá puxar de forma direta?
                        Ydata = self._asl_data('pcasl')[
                            0, :, k, j, i
                        ]   # taking the first TE to get less influence in the PLD/Buxton model

                        par_fit, _ = curve_fit(
                            mod_buxton,
                            BuxtonX,
                            Ydata,
                            p0=par0,
                            bounds=(lb, ub),
                        )
                        self._cbf_map[k, j, i] = par_fit[0]
                        self._att_map[k, j, i] = par_fit[1]

        return {
            'cbf': self._cbf_map,
            'cbf_norm': self._rescale_cbf(self._cbf_map),
            'att': self._att_map,
        }

    def _rescale_cbf(self, map):
        return map * (60 * 60 * 1000)

    def _check_mask_values(self, mask, label):
        # Check whether the mask provided is a binary image
        unique_values = np.unique(mask)
        if unique_values.size > 2:
            warnings.warn(
                'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
                UserWarning,
            )

        # Check whether the label value is found in the mask image
        label_ok = False
        for value in unique_values:
            if label == value:
                label_ok = True
                break
        if not label_ok:
            raise ValueError('Label value is not found in the mask provided.')

        # Check whether the dimensions between mask and input volume matches
        mask_shape = mask.shape
        input_vol_shape = self._asl_data('pcasl')[0, 0, :, :, :].shape
        if mask_shape != input_vol_shape:
            raise TypeError(
                f'Image mask dimension does not match with input 3D volume. Mask shape {mask_shape} not equal to {input_vol_shape}'
            )

    def _create_basic_mask(self):
        if self._asl_data('m0') is None:
            raise ValueError(
                'ASLData is incomplete. CBFMapping need pcasl and m0 images.'
            )
        in_shape = self._asl_data('m0').shape
        return np.ones(in_shape)
