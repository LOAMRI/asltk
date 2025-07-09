import warnings
from multiprocessing import Array, Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from rich.progress import Progress
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.aux_methods import _check_mask_values
from asltk.logging_config import get_logger, log_data_info, log_processing_step
from asltk.models.signal_dynamic import asl_model_buxton
from asltk.mri_parameters import MRIParameters
from asltk.reconstruction.smooth_utils import apply_smoothing_to_maps

# Global variables to assist multi cpu threading
cbf_map = None
att_map = None
brain_mask = None
asl_data = None


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
            >>> asl_data = ASLData(pcasl='./tests/files/pcasl_mte.nii.gz',m0='./tests/files/m0.nii.gz')
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
        """Defines a brain mask to limit CBF mapping calculations to specific regions.

        A brain mask significantly improves processing speed by limiting calculations
        to brain tissue voxels and excluding background regions. It also improves
        the quality of results by focusing the fitting algorithm on relevant tissue.

        A image mask is simply an image that defines the voxels where the ASL
        calculation should be made. The mask should have the same spatial dimensions
        as the M0 reference image.

        A most common approach is to use a binary image (zeros for background
        and 1 for brain tissues). However, the method can also handle multi-label
        masks by specifying which label value represents brain tissue.

        Args:
            brain_mask (np.ndarray): The image representing the brain mask.
                Must match the spatial dimensions of the M0 image.
            label (int, optional): The label value used to define the foreground
                tissue (brain). Defaults to 1. Voxels with this value will be
                included in processing.

        Examples:
            Use a binary brain mask:
            >>> from asltk.asldata import ASLData
            >>> from asltk.reconstruction import CBFMapping
            >>> import numpy as np
            >>> asl_data = ASLData(
            ...     pcasl='./tests/files/pcasl_mte.nii.gz',
            ...     m0='./tests/files/m0.nii.gz',
            ...     ld_values=[1.8], pld_values=[1.8]
            ... )
            >>> cbf_mapper = CBFMapping(asl_data)
            >>> # Create a simple brain mask (center region only)
            >>> mask_shape = asl_data('m0').shape  # Get M0 dimensions
            >>> brain_mask = np.zeros(mask_shape)
            >>> brain_mask[2:6, 10:25, 10:25] = 1  # Define brain region
            >>> cbf_mapper.set_brain_mask(brain_mask)

            Load and use an existing brain mask:
            >>> # Load pre-computed brain mask
            >>> from asltk.utils import load_image
            >>> brain_mask = load_image('./tests/files/m0_brain_mask.nii.gz')
            >>> cbf_mapper.set_brain_mask(brain_mask)

            Use multi-label mask (select specific region):
            >>> # Assuming a segmentation mask with different tissue labels
            >>> segmentation_mask = np.random.randint(0, 4, mask_shape)  # Example
            >>> # Use only label 2 (e.g., grey matter)
            >>> cbf_mapper.set_brain_mask(segmentation_mask, label=2)

            Automatic thresholding of M0 image as mask:
            >>> # Use M0 intensity to create brain mask
            >>> m0_data = asl_data('m0')
            >>> threshold = np.percentile(m0_data, 20)  # Bottom 20% as background
            >>> auto_mask = (m0_data > threshold).astype(np.uint8)
            >>> cbf_mapper.set_brain_mask(auto_mask)

        Raises:
            ValueError: If brain_mask dimensions don't match M0 image dimensions.
        """
        logger = get_logger('cbf_mapping')
        logger.info(f'Setting brain mask with label {label}')

        _check_mask_values(brain_mask, label, self._asl_data('m0').shape)

        binary_mask = (brain_mask == label).astype(np.uint8) * label
        self._brain_mask = binary_mask

        mask_volume = np.sum(binary_mask > 0)
        logger.info(f'Brain mask set successfully: {mask_volume} voxels')

    def get_brain_mask(self):
        """Get the current brain mask image being used for CBF calculations.

        Returns:
            np.ndarray: The brain mask image as a binary array where 1 indicates
                brain tissue voxels that will be processed, and 0 indicates
                background voxels that will be skipped.

        Examples:
            Check if a brain mask has been set:
            >>> from asltk.asldata import ASLData
            >>> from asltk.reconstruction import CBFMapping
            >>> import numpy as np
            >>> asl_data = ASLData(
            ...     pcasl='./tests/files/pcasl_mte.nii.gz',
            ...     m0='./tests/files/m0.nii.gz',
            ...     ld_values=[1.8], pld_values=[1.8]
            ... )
            >>> cbf_mapper = CBFMapping(asl_data)
            >>> # Initially, mask covers entire volume
            >>> current_mask = cbf_mapper.get_brain_mask()

            Verify brain mask after setting:
            >>> brain_mask = np.ones(asl_data('m0').shape)
            >>> brain_mask[0:4, :, :] = 0  # Remove some slices
            >>> cbf_mapper.set_brain_mask(brain_mask)
            >>> updated_mask = cbf_mapper.get_brain_mask()
        """
        return self._brain_mask

    def create_map(
        self,
        ub=[1.0, 5000.0],
        lb=[0.0, 0.0],
        par0=[1e-5, 1000],
        cores: int = cpu_count(),
        smoothing=None,
        smoothing_params=None,
    ):
        """Create the CBF and also ATT maps using the Buxton ASL model.

        This method performs voxel-wise non-linear fitting of the Buxton ASL model
        to generate Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) maps.
        The fitting is performed in parallel using multiple CPU cores for efficiency.

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
            ub (list, optional): The upper bounds for [CBF, ATT] fitting parameters.
                Defaults to [1.0, 5000.0]. CBF in relative units, ATT in ms.
            lb (list, optional): The lower bounds for [CBF, ATT] fitting parameters.
                Defaults to [0.0, 0.0]. Both parameters must be non-negative.
            par0 (list, optional): The initial guess for [CBF, ATT] parameters.
                Defaults to [1e-5, 1000]. Good starting values help convergence.
            cores (int, optional): Number of CPU threads to use for parallel processing.
                Defaults to using all available threads. Use fewer cores to preserve
                system resources.
            smoothing (str, optional): Type of spatial smoothing filter to apply.
                Options: None (default, no smoothing), 'gaussian', 'median'.
                Smoothing is applied to all output maps after reconstruction.
            smoothing_params (dict, optional): Parameters for the smoothing filter.
                For 'gaussian': {'sigma': float} (default: 1.0)
                For 'median': {'size': int} (default: 3)

        Returns:
            dict: A dictionary containing:
                - 'cbf': Raw CBF map in model units (numpy.ndarray)
                - 'cbf_norm': Normalized CBF map in mL/100g/min (numpy.ndarray)
                - 'att': ATT map in milliseconds (numpy.ndarray)
                All maps are smoothed if smoothing is enabled.

        Examples:  # doctest: +SKIP
            Basic CBF mapping with default parameters:
            >>> from asltk.asldata import ASLData
            >>> from asltk.reconstruction import CBFMapping
            >>> import numpy as np
            >>> # Load ASL data with LD/PLD values
            >>> asl_data = ASLData(
            ...     pcasl='./tests/files/pcasl_mte.nii.gz',
            ...     m0='./tests/files/m0.nii.gz',
            ...     ld_values=[1.8, 1.8, 1.8],
            ...     pld_values=[0.8, 1.8, 2.8]
            ... )
            >>> cbf_mapper = CBFMapping(asl_data)
            >>> # Set brain mask (recommended for faster processing)
            >>> brain_mask = np.ones((5, 35, 35))  # Example mask
            >>> cbf_mapper.set_brain_mask(brain_mask)
            >>> # Generate maps
            >>> results = cbf_mapper.create_map() # doctest: +SKIP

            Custom parameter bounds for specific tissue properties:
            >>> # For grey matter regions (higher CBF expected)
            >>> results_gm = cbf_mapper.create_map(
            ...     ub=[2.0, 3000.0],      # Higher CBF upper bound
            ...     lb=[0.1, 500.0],       # Reasonable lower bounds
            ...     par0=[0.5, 1200.0]     # Good initial guess for GM
            ... ) # doctest: +SKIP

            Apply spatial smoothing to reduce noise:
            >>> # Gaussian smoothing with default sigma=1.0
            >>> results_smooth = cbf_mapper.create_map(
            ...     smoothing='gaussian'
            ... ) # doctest: +SKIP
            >>> # Custom smoothing parameters
            >>> results_custom = cbf_mapper.create_map(
            ...     smoothing='gaussian',
            ...     smoothing_params={'sigma': 1.5}
            ... ) # doctest: +SKIP
            >>> # Median filtering for edge-preserving smoothing
            >>> results_median = cbf_mapper.create_map(
            ...     smoothing='median',
            ...     smoothing_params={'size': 5}
            ... ) # doctest: +SKIP

            Memory-efficient processing with limited cores:
            >>> # Use only 4 cores to preserve system resources
            >>> results = cbf_mapper.create_map(cores=4) # doctest: +SKIP

        Raises:
            ValueError: If cores parameter is invalid, or if LD/PLD values are missing.
        """
        logger = get_logger('cbf_mapping')
        logger.info('Starting CBF map creation')

        if (cores < 0) or (cores > cpu_count()) or not isinstance(cores, int):
            error_msg = 'Number of proecess must be at least 1 and less than maximum cores availble.'
            logger.error(
                f'{error_msg} Requested: {cores}, Available: {cpu_count()}'
            )
            raise ValueError(error_msg)

        if (
            len(self._asl_data.get_ld()) == 0
            or len(self._asl_data.get_pld()) == 0
        ):
            error_msg = 'LD or PLD list of values must be provided.'
            logger.error(error_msg)
            raise ValueError(error_msg)
        # TODO Testar se retirando esse if do LD PLD sizes, continua rodando... isso é erro do ASLData

        logger.info(f'Using {cores} CPU cores for parallel processing')
        log_processing_step('Initializing CBF mapping computation')

        global asl_data, brain_mask
        asl_data = self._asl_data
        brain_mask = self._brain_mask

        BuxtonX = [self._asl_data.get_ld(), self._asl_data.get_pld()]

        x_axis, y_axis, z_axis = (
            self._asl_data('m0').shape[2],
            self._asl_data('m0').shape[1],
            self._asl_data('m0').shape[0],
        )

        logger.info(
            f'Processing volume dimensions: {z_axis}x{y_axis}x{x_axis}'
        )

        cbf_map_shared = Array('d', z_axis * y_axis * x_axis, lock=False)
        att_map_shared = Array('d', z_axis * y_axis * x_axis, lock=False)

        log_processing_step(
            'Running voxel-wise CBF fitting', 'this may take several minutes'
        )
        with Pool(
            processes=cores,
            initializer=_cbf_init_globals,
            initargs=(cbf_map_shared, att_map_shared, brain_mask, asl_data),
        ) as pool:
            with Progress() as progress:
                task = progress.add_task('CBF/ATT processing...', total=x_axis)
                results = [
                    pool.apply_async(
                        _cbf_process_slice,
                        args=(
                            i,
                            x_axis,
                            y_axis,
                            z_axis,
                            BuxtonX,
                            par0,
                            lb,
                            ub,
                        ),
                        callback=lambda _: progress.update(task, advance=1),
                    )
                    for i in range(x_axis)
                ]
                for result in results:
                    result.wait()

        self._cbf_map = np.frombuffer(cbf_map_shared).reshape(
            z_axis, y_axis, x_axis
        )
        self._att_map = np.frombuffer(att_map_shared).reshape(
            z_axis, y_axis, x_axis
        )

        # Log completion statistics
        cbf_values = self._cbf_map[brain_mask > 0]
        att_values = self._att_map[brain_mask > 0]

        logger.info(f'CBF mapping completed successfully')
        logger.info(
            f'CBF statistics - Mean: {np.mean(cbf_values):.4f}, Std: {np.std(cbf_values):.4f}'
        )
        logger.info(
            f'ATT statistics - Mean: {np.mean(att_values):.4f}, Std: {np.std(att_values):.4f}'
        )

        output_maps = {
            'cbf': self._cbf_map,
            'cbf_norm': self._cbf_map * (60 * 60 * 1000),
            'att': self._att_map,
        }

        # Apply smoothing if requested
        return apply_smoothing_to_maps(
            output_maps, smoothing, smoothing_params
        )


def _cbf_init_globals(
    cbf_map_, att_map_, brain_mask_, asl_data_
):   # pragma: no cover
    # indirect call method by CBFMapping().create_map()
    global cbf_map, att_map, brain_mask, asl_data
    cbf_map = cbf_map_
    att_map = att_map_
    brain_mask = brain_mask_
    asl_data = asl_data_


def _cbf_process_slice(
    i, x_axis, y_axis, z_axis, BuxtonX, par0, lb, ub
):   # pragma: no cover
    # indirect call method by CBFMapping().create_map()
    for j in range(y_axis):
        for k in range(z_axis):
            if brain_mask[k, j, i] != 0:
                m0_px = asl_data('m0')[k, j, i]

                def mod_buxton(Xdata, par1, par2):
                    return asl_model_buxton(
                        Xdata[0], Xdata[1], m0_px, par1, par2
                    )

                Ydata = asl_data('pcasl')[0, :, k, j, i]

                # Calculate the processing index for the 3D space
                index = k * (y_axis * x_axis) + j * x_axis + i

                try:
                    par_fit, _ = curve_fit(
                        mod_buxton, BuxtonX, Ydata, p0=par0, bounds=(lb, ub)
                    )
                    cbf_map[index] = par_fit[0]
                    att_map[index] = par_fit[1]
                except RuntimeError:
                    cbf_map[index] = 0.0
                    att_map[index] = 0.0
