import warnings

import numpy as np
import SimpleITK as sitk
from rich import print
from rich.progress import track
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.mri_parameters import MRIParameters
from asltk.utils import asl_model_buxton, asl_model_multi_te

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
        if self._asl_data('m0') is None:
            raise ValueError(
                'ASLData is incomplete. CBFMapping need pcasl and m0 images.'
            )

        self._brain_mask = np.ones(self._asl_data('m0').shape)
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

    def get_brain_mask(self):
        """Get the brain mask image

        Returns:
            (np.ndarray): The brain mask image
        """
        return self._brain_mask

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
        if (
            len(self._asl_data.get_ld()) == 0
            or len(self._asl_data.get_pld()) == 0
        ):
            raise ValueError('LD or PLD list of values must be provided.')

        BuxtonX = [
            self._asl_data.get_ld(),
            self._asl_data.get_pld(),
        ]   # x data for the Buxton model

        x_axis = self._asl_data('m0').shape[2]   # height
        y_axis = self._asl_data('m0').shape[1]   # width
        z_axis = self._asl_data('m0').shape[0]   # depth

        for i in track(
            range(x_axis), description='[green]CBF/ATT processing...'
        ):
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

                        # TODO Investigar mais...quando chama multiTE e ele chama o CBF acusa erro RuntimeError. Dado utilizado LOAMSample/pac2
                        try:
                            par_fit, _ = curve_fit(
                                mod_buxton,
                                BuxtonX,
                                Ydata,
                                p0=par0,
                                bounds=(lb, ub),
                            )
                            self._cbf_map[k, j, i] = par_fit[0]
                            self._att_map[k, j, i] = par_fit[1]
                        except RuntimeError:
                            self._cbf_map[k, j, i] = 0.0
                            self._att_map[k, j, i] = 0.0

        return {
            'cbf': self._cbf_map,
            'cbf_norm': self._cbf_map * (60 * 60 * 1000),
            'att': self._att_map,
        }

    def _check_mask_values(self, mask, label):
        # Check wheter mask input is an numpy array
        if not isinstance(mask, np.ndarray):
            raise TypeError(f'mask is not an numpy array. Type {type(mask)}')

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


class MultiTE_ASLMapping(MRIParameters):
    def __init__(self, asl_data: ASLData) -> None:
        """Basic MultiTE_ASLMapping constructor

        Notes:
            The ASLData is the base data used in the object constructor.
            In order to create the CBF map correctly, a proper ASLData must be
            provided. Check whether the ASLData given as input is defined
            correctly. In particular, it must provide the `te_values` list of
            values in the ASLData object

        Examples:
            The default MRIParameters are used as default in the object
            constructor
            >>> asl_data = ASLData(pcasl='./tests/files/pcasl.nii.gz',m0='./tests/files/m0.nii.gz', te_values=[13.2, 25.7, 50.4])
            >>> mte = MultiTE_ASLMapping(asl_data)
            >>> mte.get_constant('T1csf')
            1400.0

            If the user want to change the MRIParameter value, for a specific
            object, one can change it directly:
            >>> mte.set_constant(1600.0, 'T1csf')
            >>> mte.get_constant('T1csf')
            1600.0
            >>> default_param = MRIParameters()
            >>> default_param.get_constant('T1csf')
            1400.0

        Args:
            asl_data (ASLData): The ASL data object (ASLData)

        Raises:
            ValueError: Raises when an incomplete ASLData object is provided
        """
        super().__init__()
        self._asl_data = asl_data
        self._basic_maps = CBFMapping(asl_data)
        if self._asl_data.get_te() is None:
            raise ValueError(
                'ASLData is incomplete. MultiTE_ASLMapping need a list of TE values.'
            )

        self._brain_mask = np.ones(self._asl_data('m0').shape)
        self._cbf_map = np.zeros(self._asl_data('m0').shape)
        self._att_map = np.zeros(self._asl_data('m0').shape)
        self._t1blgm_map = np.zeros(self._asl_data('m0').shape)

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

    def get_brain_mask(self):
        """Get the brain mask image

        Returns:
            (np.ndarray): The brain mask image
        """
        return self._brain_mask

    def set_cbf_map(self, cbf_map: np.ndarray):
        """Set the CBF map to the MultiTE_ASLMapping object.

        Note:
            The CBF maps must have the original scale in order to calculate the
            T1blGM map correclty. Hence, if the CBF map was made using
            CBFMapping class, one can use the 'cbf' output.

        Args:
            cbf_map (np.ndarray): The CBF map that is set in the MultiTE_ASLMapping object
        """
        self._cbf_map = cbf_map

    def get_cbf_map(self) -> np.ndarray:
        """Get the CBF map storaged at the MultiTE_ASLMapping object

        Returns:
            (np.ndarray): The CBF map that is storaged in the
            MultiTE_ASLMapping object
        """
        return self._cbf_map

    def set_att_map(self, att_map: np.ndarray):
        """Set the ATT map to the MultiTE_ASLMapping object.

        Args:
            att_map (np.ndarray): The ATT map that is set in the MultiTE_ASLMapping object
        """
        self._att_map = att_map

    def get_att_map(self):
        """Get the ATT map storaged at the MultiTE_ASLMapping object

        Returns:
            (np.ndarray): _description_
        """
        return self._att_map

    def _check_mask_values(self, mask, label):
        # Check wheter mask input is an numpy array
        if not isinstance(mask, np.ndarray):
            raise TypeError(f'mask is not an numpy array. Type {type(mask)}')

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

    def create_map(
        self,
        ub: list = [np.inf],
        lb: list = [0.0],
        par0: list = [400],
    ):
        """Create the T1 relaxation exchange between blood and Grey Matter (GM)
        , i.e. the T1blGM map resulted from the multi-compartiment TE ASL model.

        Reference:  Ultra-long-TE arterial spin labeling reveals rapid and
        brain-wide blood-to-CSF water transport in humans, NeuroImage,
        doi: 10.1016/j.neuroimage.2021.118755

        Note:
            The CBF and ATT maps can be provided before calling this method,
            using the proper set/get methods for each map. If the user does not
            provide these maps, a new calculation is automatically made using
            the default execution implemented at CBFMapping class.

        Note:
            The CBF map must be at the original scale to perform the correct
            multiTE-ASL model. Therefore, provide the 'cbf' output.

        The method assumes that the fine tunning map can be approached using
        the adpted initial guess (parameter par0). Hence, the generated T1blGM
        map applies a cut-off using only positive values (`>0`) and upper limit
        of four times the initial guess (`4 * par0`).

        Note:
            It is a good practive to apply a spatial smoothing in the output
            T1blGM map, in order to improve SNR. However, the `create_map`
            method does not applies any image filter as default.

        Args:
            ub (list, optional): The upper limit values. Defaults to [1.0, 5000.0].
            lb (list, optional): The lower limit values. Defaults to [0.0, 0.0].
            par0 (list, optional): The initial guess parameter for non-linear fitting. Defaults to [1e-5, 1000].

        Returns:
            (dict): A dictionary with 'cbf', 'att' and 'cbf_norm'
        """
        # TODO As entradas ub, lb e par0 não são aplicadas para CBF. Pensar se precisa ter essa flexibilidade para acertar o CBF interno à chamada
        self._basic_maps.set_brain_mask(self._brain_mask)

        basic_maps = {'cbf': self._cbf_map, 'att': self._att_map}
        if np.mean(self._cbf_map) == 0 or np.mean(self._att_map) == 0:
            # If the CBF/ATT maps are zero (empty), then a new one is created
            print(
                '[blue][INFO] The CBF/ATT map were not provided. Creating these maps before next step...'
            )
            basic_maps = self._basic_maps.create_map()
            self._cbf_map = basic_maps['cbf']
            self._att_map = basic_maps['att']

        x_axis = self._asl_data('m0').shape[2]   # height
        y_axis = self._asl_data('m0').shape[1]   # width
        z_axis = self._asl_data('m0').shape[0]   # depth

        for i in track(
            range(x_axis), description='[green]multiTE-ASL processing...'
        ):
            for j in range(y_axis):
                for k in range(z_axis):
                    if self._brain_mask[k, j, i] != 0:
                        m0_px = self._asl_data('m0')[k, j, i]

                        def mod_2comp(Xdata, par1):
                            return asl_model_multi_te(
                                Xdata[:, 0],
                                Xdata[:, 1],
                                Xdata[:, 2],
                                m0_px,
                                basic_maps['cbf'][k, j, i],
                                basic_maps['att'][k, j, i],
                                par1,
                                self.T2bl,
                                self.T2gm,
                            )

                        # Ydata = np.swapaxes(asl_img[:,:,k,j,i],0,1).reshape((len(pld) * len(te), 1), order='F').flatten('F')
                        Ydata = (
                            self._asl_data('pcasl')[:, :, k, j, i]
                            .reshape(
                                (
                                    len(self._asl_data.get_ld())
                                    * len(self._asl_data.get_te()),
                                    1,
                                )
                            )
                            .flatten()
                        )

                        try:
                            Xdata = self._create_x_data(
                                self._asl_data.get_ld(),
                                self._asl_data.get_pld(),
                                self._asl_data.get_te(),
                            )
                            par_fit, _ = curve_fit(
                                mod_2comp,
                                Xdata,
                                Ydata,
                                p0=par0,
                                bounds=(lb, ub),
                            )
                            self._t1blgm_map[k, j, i] = par_fit[0]
                        except RuntimeError:
                            self._t1blgm_map[k, j, i] = 0.0

        # Adjusting output image boundaries
        self._t1blgm_map = self._adjust_image_limits(self._t1blgm_map, par0[0])

        return {
            'cbf': self._cbf_map,
            'cbf_norm': self._cbf_map * (60 * 60 * 1000),
            'att': self._att_map,
            't1blgm': self._t1blgm_map,
        }

    def _create_x_data(self, ld, pld, te):
        # array for the x values, assuming an arbitrary size based on the PLD
        # and TE vector size
        Xdata = np.zeros((len(pld) * len(te), 3))

        count = 0
        for i in range(len(pld)):
            for j in range(len(te)):
                Xdata[count] = [ld[i], pld[i], te[j]]
                count += 1

        return Xdata

    def _adjust_image_limits(self, map, init_guess):
        img = sitk.GetImageFromArray(map)
        thr_filter = sitk.ThresholdImageFilter()
        thr_filter.SetUpper(
            4 * init_guess
        )   # assuming upper to 4x the initial guess
        thr_filter.SetLower(0.0)
        img = thr_filter.Execute(img)

        return sitk.GetArrayFromImage(img)
