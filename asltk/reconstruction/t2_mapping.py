from multiprocessing import Array, Pool, cpu_count

import numpy as np
from rich import print
from rich.progress import Progress
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.aux_methods import _apply_smoothing_to_maps, _check_mask_values
from asltk.logging_config import get_logger, log_processing_step
from asltk.mri_parameters import MRIParameters

# Global variables for multiprocessing
t2_map_shared = None
brain_mask = None
data = None
TEs = None


class T2Scalar_ASLMapping(MRIParameters):
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

        self._brain_mask = np.ones(self._asl_data('m0').shape)
        self._t2_maps = None  # Will be 4D: (Z, Y, X, N_PLDS)
        self._mean_t2s = None

    def set_brain_mask(self, brain_mask: np.ndarray, label: int = 1):
        """Defines whether a brain a mask is applied to the T2 scalar ASL
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
        _check_mask_values(brain_mask, label, self._asl_data('m0').shape)

        binary_mask = (brain_mask == label).astype(np.uint8) * label
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
        self, cores=cpu_count(), smoothing=None, smoothing_params=None
    ):
        """
        Creates the T2 maps using the ASL data and the provided brain mask
        (Multiprocessing version, following CBFMapping strategy)
        """
        logger = get_logger('t2_mapping')
        logger.info('Starting T2 map creation')

        data = self._asl_data('pcasl')
        mask = self._brain_mask
        TEs = np.array(self._te_values)
        PLDs = np.array(self._pld_values)
        n_tes, n_plds, z_axis, y_axis, x_axis = data.shape

        t2_maps_all = []
        mean_t2s = []

        for pld_idx in range(n_plds):
            logger.info(f'Processing PLD index {pld_idx} ({PLDs[pld_idx]} ms)')
            t2_map_shared = Array('d', z_axis * y_axis * x_axis, lock=False)
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
                        f'T2 fitting (PLD {PLDs[pld_idx]} ms)...', total=x_axis
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

        t2_maps_stacked = np.stack(
            t2_maps_all, axis=0
        )  # shape: (N_PLDS, Z, Y, X)
        self._t2_maps = t2_maps_stacked
        self._mean_t2s = mean_t2s

        logger.info('T2 mapping completed successfully')
        logger.info(
            f'T2 statistics - Mean: {np.mean(self._t2_maps):.4f}, Std: {np.std(self._t2_maps):.4f}'
        )

        output_maps = {
            't2': self._t2_maps,
            'mean_t2': self._mean_t2s,
        }

        return _apply_smoothing_to_maps(
            output_maps, smoothing, smoothing_params
        )


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
