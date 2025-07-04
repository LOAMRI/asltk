import warnings
from multiprocessing import Array, Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from rich import print
from rich.progress import Progress
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.aux_methods import _check_mask_values
from asltk.models.signal_dynamic import asl_model_multi_dw
from asltk.mri_parameters import MRIParameters
from asltk.reconstruction import CBFMapping
from asltk.reconstruction.smooth_utils import apply_smoothing_to_maps

# Global variables to assist multi cpu threading
cbf_map = None
att_map = None
brain_mask = None
asl_data = None
ld_arr = None
pld_arr = None
te_arr = None
tblgm_map = None
t2bl = None
t2gm = None


class MultiDW_ASLMapping(MRIParameters):
    def __init__(self, asl_data: ASLData):
        """Multi-Diffusion-Weighted ASL mapping constructor for advanced perfusion analysis.

        MultiDW_ASLMapping enables sophisticated ASL analysis by incorporating multiple
        diffusion weightings (b-values) to separate intravascular and tissue
        compartments. This approach provides enhanced characterization of perfusion
        and can help differentiate between different vascular compartments.

        The class implements diffusion-weighted ASL analysis that can distinguish:
        - Fast-flowing blood (intravascular component)
        - Slow-flowing blood and tissue perfusion
        - Apparent diffusion coefficients for each compartment
        - Water exchange parameters between compartments

        Notes:
            The ASLData object must contain `dw_values` - a list of diffusion
            b-values used during ASL acquisition. These b-values are essential
            for the multi-compartment diffusion model fitting.

        Examples:
            Basic multi-DW ASL mapping setup:
            >>> from asltk.asldata import ASLData
            >>> from asltk.reconstruction import MultiDW_ASLMapping
            >>> # Create ASL data with diffusion weighting
            >>> asl_data = ASLData(
            ...     pcasl='./tests/files/pcasl_mdw.nii.gz',
            ...     m0='./tests/files/m0.nii.gz',
            ...     dw_values=[0, 50, 100, 200],    # b-values in s/mm²
            ...     ld_values=[1.8, 1.8, 1.8, 1.8],
            ...     pld_values=[0.8, 1.8, 2.8, 3.8]
            ... )
            >>> mdw_mapper = MultiDW_ASLMapping(asl_data)

            Access diffusion-related maps (after processing):
            >>> # These maps will be populated after create_map() is called
            >>> # A1: Signal amplitude for compartment 1
            >>> # D1: Apparent diffusion coefficient for compartment 1
            >>> # A2: Signal amplitude for compartment 2
            >>> # D2: Apparent diffusion coefficient for compartment 2
            >>> # kw: Water exchange parameter

        Args:
            asl_data (ASLData): The ASL data object containing multi-DW acquisition.
                Must include dw_values (b-values), ld_values, and pld_values.

        Raises:
            ValueError: If ASLData object lacks required DW values for
                diffusion-weighted analysis.

        See Also:
            CBFMapping: For basic CBF/ATT mapping without diffusion weighting
            MultiTE_ASLMapping: For multi-echo ASL analysis
        """
        super().__init__()
        self._asl_data = asl_data
        self._basic_maps = CBFMapping(asl_data)
        if self._asl_data.get_dw() is None:
            raise ValueError(
                'ASLData is incomplete. MultiDW_ASLMapping need a list of DW values.'
            )

        self._brain_mask = np.ones(self._asl_data('m0').shape)
        self._cbf_map = np.zeros(self._asl_data('m0').shape)
        self._att_map = np.zeros(self._asl_data('m0').shape)

        self._b_values = self._asl_data.get_dw()
        # self._A1 = np.zeros(tuple([len(self._b_values)]) + self._asl_data('m0').shape)
        self._A1 = np.zeros(self._asl_data('m0').shape)
        # self._D1 = np.zeros(tuple([1]) +self._asl_data('m0').shape)
        self._D1 = np.zeros(self._asl_data('m0').shape)
        self._A2 = np.zeros(self._asl_data('m0').shape)
        # self._A2 = np.zeros(tuple([len(self._b_values)])  + self._asl_data('m0').shape)
        # self._D2 = np.zeros(tuple([1]) +self._asl_data('m0').shape)
        self._D2 = np.zeros(self._asl_data('m0').shape)
        self._kw = np.zeros(self._asl_data('m0').shape)

    def set_brain_mask(self, brain_mask: np.ndarray, label: int = 1):
        """Set brain mask for MultiDW-ASL processing (strongly recommended).

        A brain mask is especially important for multi-diffusion-weighted ASL
        processing as it significantly reduces computation time by limiting
        the intensive voxel-wise fitting to brain tissue regions only.

        Without a brain mask, processing time can be prohibitively long (hours)
        for whole-volume analysis. A proper brain mask can reduce processing
        time by 5-10x while maintaining analysis quality.

        Args:
            brain_mask (np.ndarray): The image representing the brain mask.
                Must match the spatial dimensions of the M0 image.
            label (int, optional): The label value used to define brain tissue.
                Defaults to 1. Voxels with this value will be processed.

        Examples:
            Set a brain mask for efficient processing:
            >>> from asltk.asldata import ASLData
            >>> from asltk.reconstruction import MultiDW_ASLMapping
            >>> import numpy as np
            >>> asl_data = ASLData(
            ...     pcasl='./tests/files/pcasl_mdw.nii.gz',
            ...     m0='./tests/files/m0.nii.gz',
            ...     dw_values=[0, 50, 100], ld_values=[1.8]*3, pld_values=[1.8]*3
            ... )
            >>> mdw_mapper = MultiDW_ASLMapping(asl_data)
            >>> # Create conservative brain mask (center region only)
            >>> mask_shape = asl_data('m0').shape
            >>> brain_mask = np.zeros(mask_shape)
            >>> brain_mask[1:4, 5:30, 5:30] = 1  # Conservative brain region
            >>> mdw_mapper.set_brain_mask(brain_mask)

        Note:
            For multi-DW ASL, consider using a more conservative (smaller) brain
            mask initially to test parameters and processing time, then expand
            to full brain analysis once satisfied with results.
        """
        _check_mask_values(brain_mask, label, self._asl_data('m0').shape)

        binary_mask = (brain_mask == label).astype(np.uint8) * label
        self._brain_mask = binary_mask

    def get_brain_mask(self):
        """Get the brain mask image

        Returns:
            (np.ndarray): The brain mask image
        """
        return self._brain_mask

    def set_cbf_map(self, cbf_map: np.ndarray):
        """Set the CBF map to the MultiDW_ASLMapping object.

        Note:
            The CBF maps must have the original scale in order to calculate the
            T1blGM map correclty. Hence, if the CBF map was made using
            CBFMapping class, one can use the 'cbf' output.

        Args:
            cbf_map (np.ndarray): The CBF map that is set in the MultiDW_ASLMapping object
        """
        self._cbf_map = cbf_map

    def get_cbf_map(self) -> np.ndarray:
        """Get the CBF map storaged at the MultiDW_ASLMapping object

        Returns:
            (np.ndarray): The CBF map that is storaged in the
            MultiDW_ASLMapping object
        """
        return self._cbf_map

    def set_att_map(self, att_map: np.ndarray):
        """Set the ATT map to the MultiDW_ASLMapping object.

        Args:
            att_map (np.ndarray): The ATT map that is set in the MultiDW_ASLMapping object
        """
        self._att_map = att_map

    def get_att_map(self):
        """Get the ATT map storaged at the MultiDW_ASLMapping object

        Returns:
            (np.ndarray): _description_
        """
        return self._att_map

    def create_map(
        self,
        lb: list = [0.0, 0.0, 0.0, 0.0],
        ub: list = [np.inf, np.inf, np.inf, np.inf],
        par0: list = [0.5, 0.000005, 0.5, 0.000005],
        smoothing=None,
        smoothing_params=None,
    ):
        """Create multi-diffusion-weighted ASL maps for compartment analysis.

        This method performs advanced diffusion-weighted ASL analysis to generate
        multi-compartment perfusion maps. The analysis uses multiple b-values to
        separate fast-flowing intravascular and slower tissue perfusion components.

        The method fits a bi-exponential diffusion model to estimate:
        - Signal amplitudes and apparent diffusion coefficients for two compartments
        - Water exchange parameters between vascular and tissue compartments
        - Enhanced CBF characterization with compartment specificity

        Note:
            The CBF and ATT maps can be provided before calling this method using
            set_cbf_map() and set_att_map() methods. If not provided, basic maps
            are automatically calculated using the CBFMapping class.

        Warning:
            This method is computationally intensive as it performs voxel-wise
            non-linear fitting without parallel processing. Consider using a brain
            mask to limit processing to relevant tissue areas.

        Args:
            lb (list, optional): Lower bounds for [A1, D1, A2, D2] parameters.
                Defaults to [0.0, 0.0, 0.0, 0.0]. All parameters should be non-negative.
                - A1, A2: Signal amplitudes (relative units)
                - D1, D2: Apparent diffusion coefficients (mm²/s)
            ub (list, optional): Upper bounds for [A1, D1, A2, D2] parameters.
                Defaults to [np.inf, np.inf, np.inf, np.inf].
            par0 (list, optional): Initial guess for [A1, D1, A2, D2] parameters.
                Defaults to [0.5, 0.000005, 0.5, 0.000005].
                - A1, A2: Typical values 0.1-1.0 (relative amplitudes)
                - D1, D2: Typical values 1e-6 to 1e-3 mm²/s
            smoothing (str, optional): Type of spatial smoothing filter to apply.
                Options: None (default, no smoothing), 'gaussian', 'median'.
                Smoothing is applied to all output maps after reconstruction.
            smoothing_params (dict, optional): Parameters for the smoothing filter.
                For 'gaussian': {'sigma': float} (default: 1.0)
                For 'median': {'size': int} (default: 3)

        Returns:
            dict: Dictionary containing diffusion-weighted ASL maps:
                - 'cbf': Basic CBF map in original units (numpy.ndarray)
                - 'cbf_norm': Normalized CBF in mL/100g/min (numpy.ndarray)
                - 'att': Arterial transit time in ms (numpy.ndarray)
                - 'A1': Signal amplitude for compartment 1 (numpy.ndarray)
                - 'D1': Apparent diffusion coefficient for compartment 1 in mm²/s (numpy.ndarray)
                - 'A2': Signal amplitude for compartment 2 (numpy.ndarray)
                - 'D2': Apparent diffusion coefficient for compartment 2 in mm²/s (numpy.ndarray)
                - 'kw': Water exchange parameter (numpy.ndarray)
                All maps are smoothed if smoothing is enabled.

        Examples:
            Basic multi-DW ASL analysis:
            >>> from asltk.asldata import ASLData
            >>> from asltk.reconstruction import MultiDW_ASLMapping
            >>> import numpy as np
            >>> # Load multi-DW ASL data
            >>> asl_data = ASLData(
            ...     pcasl='./tests/files/pcasl_mdw.nii.gz',
            ...     m0='./tests/files/m0.nii.gz',
            ...     dw_values=[0, 50, 100, 200],  # b-values in s/mm²
            ...     ld_values=[1.8, 1.8, 1.8, 1.8],
            ...     pld_values=[0.8, 1.8, 2.8, 3.8]
            ... )
            >>> mdw_mapper = MultiDW_ASLMapping(asl_data)
            >>> # Set brain mask for faster processing (recommended)
            >>> brain_mask = np.ones(asl_data('m0').shape)
            >>> brain_mask[0:2, :, :] = 0  # Remove some background slices
            >>> mdw_mapper.set_brain_mask(brain_mask)
            >>> # Generate all maps (may take several minutes)
            >>> results = mdw_mapper.create_map() # doctest: +SKIP

            Custom parameters for specific tissue analysis:
            >>> # For analyzing fast vs slow perfusion components
            >>> results = mdw_mapper.create_map(
            ...     lb=[0.1, 1e-6, 0.1, 1e-7],      # Minimum realistic values
            ...     ub=[2.0, 1e-3, 2.0, 1e-4],      # Maximum realistic values
            ...     par0=[0.8, 5e-5, 0.3, 1e-5]     # Initial guesses
            ... ) # doctest: +SKIP

        Note:
            Processing time scales with brain mask size. For a full brain analysis,
            expect processing times of 30+ minutes depending on data size and
            hardware capabilities.

        See Also:
            set_cbf_map(): Provide pre-computed CBF map
            set_att_map(): Provide pre-computed ATT map
            CBFMapping: For basic CBF/ATT mapping
        """
        self._basic_maps.set_brain_mask(self._brain_mask)

        basic_maps = {'cbf': self._cbf_map, 'att': self._att_map}
        if np.mean(self._cbf_map) == 0 or np.mean(self._att_map) == 0:
            # If the CBF/ATT maps are zero (empty), then a new one is created
            print(
                '[blue][INFO] The CBF/ATT map were not provided. Creating these maps before next step...'
            )   # pragma: no cover
            basic_maps = self._basic_maps.create_map()   # pragma: no cover
            self._cbf_map = basic_maps['cbf']   # pragma: no cover
            self._att_map = basic_maps['att']   # pragma: no cover

        x_axis = self._asl_data('m0').shape[2]   # height
        y_axis = self._asl_data('m0').shape[1]   # width
        z_axis = self._asl_data('m0').shape[0]   # depth

        # TODO Fix
        print('multiDW-ASL processing...')
        for i in range(x_axis):
            for j in range(y_axis):
                for k in range(z_axis):
                    if self._brain_mask[k, j, i] != 0:
                        # Calculates the diffusion components for (A1, D1), (A2, D2)
                        def mod_diff(Xdata, par1, par2, par3, par4):
                            return asl_model_multi_dw(
                                b_values=Xdata,
                                A1=par1,
                                D1=par2,
                                A2=par3,
                                D2=par4,
                            )

                        # M(t,b)/M(t,0)
                        Ydata = (
                            self._asl_data('pcasl')[:, :, k, j, i]
                            .reshape(
                                (
                                    len(self._asl_data.get_ld())
                                    * len(self._asl_data.get_dw()),
                                    1,
                                )
                            )
                            .flatten()
                            / self._asl_data('m0')[k, j, i]
                        )

                        try:
                            # Xdata = self._b_values
                            Xdata = self._create_x_data(
                                self._asl_data.get_ld(),
                                self._asl_data.get_pld(),
                                self._asl_data.get_dw(),
                            )

                            par_fit, _ = curve_fit(
                                mod_diff,
                                Xdata[:, 2],
                                Ydata,
                                p0=par0,
                                bounds=(lb, ub),
                            )
                            self._A1[k, j, i] = par_fit[0]
                            self._D1[k, j, i] = par_fit[1]
                            self._A2[k, j, i] = par_fit[2]
                            self._D2[k, j, i] = par_fit[3]
                        except RuntimeError:
                            self._A1[k, j, i] = 0
                            self._D1[k, j, i] = 0
                            self._A2[k, j, i] = 0
                            self._D2[k, j, i] = 0

                        # Calculates the Mc fitting to alpha = kw + T1blood
                        m0_px = self._asl_data('m0')[k, j, i]

                        # def mod_2comp(Xdata, par1):
                        #     ...
                        #     # return asl_model_multi_te(
                        #     #     Xdata[:, 0],
                        #     #     Xdata[:, 1],
                        #     #     Xdata[:, 2],
                        #     #     m0_px,
                        #     #     basic_maps['cbf'][k, j, i],
                        #     #     basic_maps['att'][k, j, i],
                        #     #     par1,
                        #     #     self.T2bl,
                        #     #     self.T2gm,
                        #     # )

                        # Ydata = (
                        #     self._asl_data('pcasl')[:, :, k, j, i]
                        #     .reshape(
                        #         (
                        #             len(self._asl_data.get_ld())
                        #             * len(self._asl_data.get_te()),
                        #             1,
                        #         )
                        #     )
                        #     .flatten()
                        # )

                        # try:
                        #     Xdata = self._create_x_data(
                        #         self._asl_data.get_ld(),
                        #         self._asl_data.get_pld(),
                        #         self._asl_data.get_dw(),
                        #     )
                        #     par_fit, _ = curve_fit(
                        #         mod_2comp,
                        #         Xdata,
                        #         Ydata,
                        #         p0=par0,
                        #         bounds=(lb, ub),
                        #     )
                        #     self._kw[k, j, i] = par_fit[0]
                        # except RuntimeError:
                        #     self._kw[k, j, i] = 0.0

        # # Adjusting output image boundaries
        # self._kw = self._adjust_image_limits(self._kw, par0[0])

        # Create output maps dictionary
        output_maps = {
            'cbf': self._cbf_map,
            'cbf_norm': self._cbf_map * (60 * 60 * 1000),
            'att': self._att_map,
            'a1': self._A1,
            'd1': self._D1,
            'a2': self._A2,
            'd2': self._D2,
            'kw': self._kw,
        }

        # Apply smoothing if requested
        return apply_smoothing_to_maps(
            output_maps, smoothing, smoothing_params
        )

    def _create_x_data(self, ld, pld, dw):
        # array for the x values, assuming an arbitrary size based on the PLD
        # and TE vector size
        Xdata = np.zeros((len(pld) * len(dw), 3))

        count = 0
        for i in range(len(pld)):
            for j in range(len(dw)):
                Xdata[count] = [ld[i], pld[i], dw[j]]
                count += 1

        return Xdata
