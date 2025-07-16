from multiprocessing import Array, Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from rich import print
from rich.progress import Progress
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.aux_methods import _apply_smoothing_to_maps, _check_mask_values
from asltk.models.signal_dynamic import asl_t2_scalar_multi_te
from asltk.mri_parameters import MRIParameters
from asltk.reconstruction import CBFMapping

# Global variables to assist multi cpu threading
# brain_mask = None
# asl_data = None
# ld_arr = None
# pld_arr = None
# te_arr = None
# tblgm_map = None
# t2bl = None
# t2gm = None


def monoexp(TE, A, T2):
    return A * np.exp(-TE / T2)


def fit_voxel(signal, TEs):
    if np.any(np.isnan(signal)) or np.max(signal) < 1:
        return np.nan
    try:
        A0 = float(np.clip(np.max(signal), 1, 1e5))
        T20 = 100
        popt, _ = curve_fit(
            monoexp,
            TEs,
            signal,
            p0=(A0, T20),
            bounds=([0, 5], [1e5, 200]),
        )
        return popt[1]
    except Exception:
        return np.nan


def _t2scalar_process_slice(
    i, x_axis, y_axis, z_axis, mask, data, TEs, pld_idx, t2_map_shared
):  # pragma: no cover
    # For slice i, fit T2 for each voxel at PLD index pld_idx
    for j in range(y_axis):
        for k in range(z_axis):
            if mask[k, j, i]:
                signal = data[k, j, i, pld_idx, :]
                t2_value = fit_voxel(signal, TEs)
                index = k * (y_axis * x_axis) + j * x_axis + i
                t2_map_shared[index] = t2_value
            else:
                index = k * (y_axis * x_axis) + j * x_axis + i
                t2_map_shared[index] = np.nan


class T2Scalar_ASLMapping(MRIParameters):
    def __init__(self, asl_data: ASLData) -> None:
        super().__init__()
        self._asl_data = asl_data
        self._te_values = self._asl_data.get_te()
        self._pld_values = self._asl_data.get_pld()
        if self._te_values is None or not self._pld_values:
            raise ValueError('ASLData must provide TE and PLD values.')

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
        """Creates the T2 maps using the ASL data and the provided brain mask

        Args:
            cores (int, optional): Number of CPU cores for processing.
            smoothing (str, optional): Smoothing type ('gaussian', 'median', or None).
            smoothing_params (dict, optional): Smoothing parameters.

        Returns:
            dict: Dictionary with T2 maps and mean T2 values.
        """
        # Data shape: (Z, Y, X, N_PLDS, N_TEs)
        data = self._asl_data('pcasl')
        mask = self._brain_mask
        TEs = np.array(self._te_values)
        PLDs = np.array(self._pld_values)
        n_tes, n_plds, z_axis, y_axis, x_axis = data.shape

        t2_maps_all = []
        mean_t2s = []

        for pld_idx in range(n_plds):
            t2_map_shared = Array('d', z_axis * y_axis * x_axis, lock=False)
            with Pool(processes=cores) as pool:
                with Progress() as progress:
                    task = progress.add_task(
                        f'T2 fitting (PLD {PLDs[pld_idx]} ms)...', total=x_axis
                    )
                    results = [
                        pool.apply_async(
                            _t2scalar_process_slice,
                            args=(
                                i,
                                x_axis,
                                y_axis,
                                z_axis,
                                mask,
                                data,
                                TEs,
                                pld_idx,
                                t2_map_shared,
                            ),
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

        self._t2_maps = np.stack(
            t2_maps_all, axis=-1
        )  # shape: (Z, Y, X, N_PLDS)
        self._mean_t2s = mean_t2s

        output_maps = {
            't2': self._t2_maps,
            'mean_t2': self._mean_t2s,
        }

        # Apply smoothing if requested
        return _apply_smoothing_to_maps(
            output_maps, smoothing, smoothing_params
        )
