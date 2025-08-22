import warnings
from multiprocessing import Array, Pool, cpu_count

import numpy as np
from rich import print
from rich.progress import Progress
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.aux_methods import _apply_smoothing_to_maps, _check_mask_values
from asltk.logging_config import get_logger, log_processing_step
from asltk.mri_parameters import MRIParameters
from asltk.utils.io import ImageIO

# Global variables for multiprocessing
t2_map_shared = None
brain_mask = None
data = None
TEs = None


class T2Scalar_ASLMapping(MRIParameters):
    """
    Class for voxel-wise T2 mapping from multi-echo ASL data.

    This class provides methods to calculate T2 relaxation maps from multi-echo ASL MRI data.
    It supports brain masking, multiprocessing for fast computation, and optional smoothing.

    Main methods:
        - set_brain_mask: Set a binary mask to restrict T2 fitting to brain voxels.
        - create_map: Compute T2 maps using multiprocessing (output shape: (N_PLDS, Z, Y, X)).
        - get_t2_maps: Retrieve the computed T2 maps.
        - get_mean_t2s: Retrieve mean T2 values per PLD.
    """

    def __init__(self, asl_data: ASLData) -> None:
        super().__init__()
        self._asl_data = asl_data
        self._te_values = self._asl_data.get_te()
        self._pld_values = self._asl_data.get_pld()

        # Check if the ASLData has TE and PLD values
        if self._te_values is None or not self._pld_values:
            raise ValueError('ASLData must provide TE and PLD values.')

        # Check if the ASLData has DW values (not allowed for T2 mapping)
        if self._asl_data.get_dw() is not None:
            raise ValueError('ASLData must not include DW values.')

        self._brain_mask = ImageIO(
            image_array=np.ones(self._asl_data('m0').get_as_numpy().shape)
        )
        self._t2_maps = None  # Will be 4D: (N_PLDS, Z, Y, X)
        self._mean_t2s = None

    def set_brain_mask(self, brain_mask: ImageIO, label: int = 1):
        """
        Set a brain mask to restrict T2 fitting to specific voxels.

        Args:
            brain_mask (np.ndarray): Binary or integer mask with the same shape as the M0 image. Nonzero values indicate voxels to include.
            label (int, optional): The label value to use as foreground (default: 1).

        The mask should be a 3D numpy array matching the spatial dimensions of the ASL data.
        """
        _check_mask_values(
            brain_mask, label, self._asl_data('m0').get_as_numpy().shape
        )

        binary_mask = ImageIO(
            image_array=(brain_mask.get_as_numpy() == label).astype(np.uint8)
            * label
        )
        self._brain_mask = binary_mask

    def get_t2_maps(self):
        """Get the T2 maps storaged at the T2Scalar_ASLMapping object

        Returns:
            (np.ndarray): The T2 maps that is storaged in the
            T2Scalar_ASLMapping object
        """
        return self._t2_maps

    def get_mean_t2s(self):
        """Get the mean T2 values calculated from the T2 maps

        Returns:
            (list): The mean T2 values for each PLD
        """
        return self._mean_t2s

    def create_map(
        self,
        cores=cpu_count(),
        smoothing=None,
        smoothing_params=None,
        suppress_warnings=False,
    ):
        """
        Compute T2 maps using multi-echo ASL data and a brain mask, with multiprocessing.

        This method uses multiprocessing to accelerate voxel-wise T2 fitting. The output is a 4D array with shape (N_PLDS, Z, Y, X).

        Warning:
            For large datasets, memory usage can be significant due to parallel processing and storage of intermediate arrays.

        Args:
            cores (int, optional): Number of CPU cores for processing. Defaults to all available.
            smoothing (str, optional): Smoothing type ('gaussian', 'median', or None).
            smoothing_params (dict, optional): Smoothing parameters.

        Returns:
            dict: Dictionary with T2 maps ('t2', shape (N_PLDS, Z, Y, X)) and mean T2 values ('mean_t2').
        """
        logger = get_logger('t2_mapping')
        logger.info('Starting T2 map creation')

        # Optionally suppress warnings
        if suppress_warnings:
            warnings_context = warnings.catch_warnings()
            warnings_context.__enter__()
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            logger.info('Warnings suppressed during T2 mapping')

        try:
            data = self._asl_data('pcasl').get_as_numpy()
            mask = self._brain_mask.get_as_numpy()
            TEs = np.array(self._te_values)
            PLDs = np.array(self._pld_values)
            n_tes, n_plds, z_axis, y_axis, x_axis = data.shape

            t2_maps_all = []
            mean_t2s = []

            for pld_idx in range(n_plds):
                logger.info(
                    f'Processing PLD index {pld_idx} ({PLDs[pld_idx]} ms)'
                )
                t2_map_shared = Array(
                    'd', z_axis * y_axis * x_axis, lock=False
                )
                log_processing_step(
                    'Running voxel-wise T2 fitting',
                    'this may take several minutes',
                )
                with Pool(
                    processes=cores,
                    initializer=_t2_init_globals,
                    initargs=(t2_map_shared, mask, data, TEs),
                ) as pool:
                    with Progress() as progress:
                        task = progress.add_task(
                            f'T2 fitting (PLD {PLDs[pld_idx]} ms)...',
                            total=x_axis,
                        )
                        results = [
                            pool.apply_async(
                                _t2_process_slice,
                                args=(i, x_axis, y_axis, z_axis, pld_idx),
                                callback=lambda _: progress.update(
                                    task, advance=1
                                ),
                            )
                            for i in range(x_axis)
                        ]
                        for result in results:
                            result.wait()

                t2_map = np.frombuffer(t2_map_shared).reshape(
                    z_axis, y_axis, x_axis
                )
                t2_maps_all.append(t2_map)
                mean_t2s.append(np.nanmean(t2_map))

            t2_maps_stacked = np.array(t2_maps_all)  # shape: (N_PLDS, Z, Y, X)
            self._t2_maps = t2_maps_stacked
            self._mean_t2s = mean_t2s

            logger.info('T2 mapping completed successfully')
            logger.info(
                f'T2 statistics - Mean: {np.mean(self._t2_maps):.4f}, Std: {np.std(self._t2_maps):.4f}'
            )

            # Prepare output maps
            # TODO At the moment, the T2 maps and mean T2 maps are as ImageIO object, however, the Spacing, Dimension are not given as a 4D array. Ceck if can be imported from the m0 image is 3D.
            t2_maps_image = ImageIO(
                image_array=np.array(
                    [
                        self._asl_data('m0').get_as_numpy()
                        for _ in range(len(t2_maps_all))
                    ]
                )
            )
            t2_maps_image.update_image_data(self._t2_maps)

            # Update the _t2_maps attribute to be an ImageIO object
            self._t2_maps = t2_maps_image

            output_maps = {
                't2': t2_maps_image,
                'mean_t2': self._mean_t2s,
            }

            return _apply_smoothing_to_maps(
                output_maps, smoothing, smoothing_params
            )
        finally:
            # Ensure warnings are restored if suppressed
            if suppress_warnings:
                warnings_context.__exit__(None, None, None)


def _fit_voxel(signal, TEs):  # pragma: no cover
    """
    Fits a monoexponential decay model to the signal across TEs to estimate T2.

    Args:
        signal (np.ndarray): Signal intensities for different TEs.
        TEs (np.ndarray): Echo times (ms).

    Returns:
        float: Estimated T2 value (ms), or 0 if fitting fails.
    """

    def monoexp(te, S0, T2):
        return S0 * np.exp(-te / T2)

    # Check for valid signal
    if np.any(np.isnan(signal)) or np.max(signal) < 1:
        return 0

    try:
        popt, _ = curve_fit(
            monoexp, TEs, signal, p0=(np.max(signal), 80), bounds=(0, np.inf)
        )
        T2 = popt[1]
        if T2 <= 0 or np.isnan(T2):
            return 0
        return T2
    except Exception:
        return 0


def _t2_init_globals(t2_map_, brain_mask_, data_, TEs_):   # pragma: no cover
    global t2_map_shared, brain_mask, data, TEs
    t2_map_shared = t2_map_
    brain_mask = brain_mask_
    data = data_
    TEs = TEs_


def _t2_process_slice(i, x_axis, y_axis, z_axis, pld_idx):   # pragma: no cover
    for j in range(y_axis):
        for k in range(z_axis):
            if brain_mask[k, j, i]:
                signal = data[:, pld_idx, k, j, i]
                t2_value = _fit_voxel(signal, TEs)
                index = k * (y_axis * x_axis) + j * x_axis + i
                t2_map_shared[index] = t2_value
            else:
                index = k * (y_axis * x_axis) + j * x_axis + i
                t2_map_shared[index] = 0
