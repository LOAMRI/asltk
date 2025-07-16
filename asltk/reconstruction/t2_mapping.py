from multiprocessing import Array, Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from rich import print
from rich.progress import Progress
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.aux_methods import _apply_smoothing_to_maps, _check_mask_values
from asltk.models.signal_dynamic import asl_model_multi_te
from asltk.mri_parameters import MRIParameters
from asltk.reconstruction import CBFMapping

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


class T2Scalar_ASLMapping(MRIParameters):
    def __init__(self, asl_data: ASLData) -> None:
        super().__init__()
        self._asl_data = asl_data
        if self._asl_data.get_te() is None:
            raise ValueError(
                'ASLData is incomplete. T2Scalar_ASLMapping need a list of TE values.'
            )

        self._brain_mask = np.ones(self._asl_data('m0').shape)
        self._pld_indexes = self._asl_data.get_pld() if self._asl_data.get_pld() is not None else []
        self._t2_map = np.zeros(self._asl_data('m0').shape)

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

    def get_brain_mask(self):
        """Get the brain mask image

        Returns:
            (np.ndarray): The brain mask image
        """
        return self._brain_mask

    def get_t2_map(self):
        """Get the T2 map storaged at the T2Scalar_ASLMapping object

        Returns:
            (np.ndarray): The T2 map that is storaged in the
            T2Scalar_ASLMapping object
        """
        return self._t2_map

    def create_map(
        self,
        cores=cpu_count(),
        smoothing=None,
        smoothing_params=None,
    ):

        # basic_maps = {'cbf': self._cbf_map, 'att': self._att_map}
        # if np.mean(self._cbf_map) == 0 or np.mean(self._att_map) == 0:
        #     # If the CBF/ATT maps are zero (empty), then a new one is created
        #     print(
        #         '[blue][INFO] The CBF/ATT map were not provided. Creating these maps before next step...'
        #     )
        #     basic_maps = self._basic_maps.create_map()
        #     self._cbf_map = basic_maps['cbf']
        #     self._att_map = basic_maps['att']

        # global asl_data, brain_mask, cbf_map, att_map, t2bl, t2gm
        # asl_data = self._asl_data
        # brain_mask = self._brain_mask
        # cbf_map = self._cbf_map
        # att_map = self._att_map
        # ld_arr = self._asl_data.get_ld()
        # pld_arr = self._asl_data.get_pld()
        # te_arr = self._asl_data.get_te()
        # t2bl = self.T2bl
        # t2gm = self.T2gm

        # x_axis = self._asl_data('m0').shape[2]   # height
        # y_axis = self._asl_data('m0').shape[1]   # width
        # z_axis = self._asl_data('m0').shape[0]   # depth

        # tblgm_map_shared = Array('d', z_axis * y_axis * x_axis, lock=False)

        # with Pool(
        #     processes=cores,
        #     initializer=_multite_init_globals,
        #     initargs=(
        #         cbf_map,
        #         att_map,
        #         brain_mask,
        #         asl_data,
        #         ld_arr,
        #         pld_arr,
        #         te_arr,
        #         tblgm_map_shared,
        #         t2bl,
        #         t2gm,
        #     ),
        # ) as pool:
        #     with Progress() as progress:
        #         task = progress.add_task(
        #             'multiTE-ASL processing...', total=x_axis
        #         )
        #         results = [
        #             pool.apply_async(
        #                 _tblgm_multite_process_slice,
        #                 args=(i, x_axis, y_axis, z_axis, par0, lb, ub),
        #                 callback=lambda _: progress.update(task, advance=1),
        #             )
        #             for i in range(x_axis)
        #         ]
        #         for result in results:
        #             result.wait()

        # self._t1blgm_map = np.frombuffer(tblgm_map_shared).reshape(
        #     z_axis, y_axis, x_axis
        # )

        # # Adjusting output image boundaries
        # self._t1blgm_map = self._adjust_image_limits(self._t1blgm_map, par0[0])

        # Create output maps dictionary
        output_maps = {
            'cbf': self._cbf_map,
            'cbf_norm': self._cbf_map * (60 * 60 * 1000),
            'att': self._att_map,
            't1blgm': self._t1blgm_map,
        }

        # Apply smoothing if requested
        return _apply_smoothing_to_maps(
            output_maps, smoothing, smoothing_params
        )

    def _adjust_image_limits(self, map, init_guess):
        img = sitk.GetImageFromArray(map)
        thr_filter = sitk.ThresholdImageFilter()
        thr_filter.SetUpper(
            4 * init_guess
        )   # assuming upper to 4x the initial guess
        thr_filter.SetLower(0.0)
        img = thr_filter.Execute(img)

        return sitk.GetArrayFromImage(img)


def _multite_init_globals(
    cbf_map_,
    att_map_,
    brain_mask_,
    asl_data_,
    ld_arr_,
    pld_arr_,
    te_arr_,
    tblgm_map_,
    t2bl_,
    t2gm_,
):   # pragma: no cover
    # indirect call method by CBFMapping().create_map()
    global cbf_map, att_map, brain_mask, asl_data, ld_arr, te_arr, pld_arr, tblgm_map, t2bl, t2gm
    cbf_map = cbf_map_
    att_map = att_map_
    brain_mask = brain_mask_
    asl_data = asl_data_
    ld_arr = ld_arr_
    pld_arr = pld_arr_
    te_arr = te_arr_
    tblgm_map = tblgm_map_
    t2bl = t2bl_
    t2gm = t2gm_


def _tblgm_multite_process_slice(
    i, x_axis, y_axis, z_axis, par0, lb, ub
):   # pragma: no cover
    # indirect call method by CBFMapping().create_map()
    for j in range(y_axis):
        for k in range(z_axis):
            if brain_mask[k, j, i] != 0:
                m0_px = asl_data('m0')[k, j, i]

                def mod_2comp(Xdata, par1):
                    return asl_model_multi_te(
                        Xdata[:, 0],
                        Xdata[:, 1],
                        Xdata[:, 2],
                        m0_px,
                        cbf_map[k, j, i],
                        att_map[k, j, i],
                        par1,
                        t2bl,
                        t2gm,
                    )

                Ydata = (
                    asl_data('pcasl')[:, :, k, j, i]
                    .reshape(
                        (
                            len(ld_arr) * len(te_arr),
                            1,
                        )
                    )
                    .flatten()
                )

                # Calculate the processing index for the 3D space
                index = k * (y_axis * x_axis) + j * x_axis + i

                try:
                    Xdata = _multite_create_x_data(
                        ld_arr,
                        pld_arr,
                        te_arr,
                    )
                    par_fit, _ = curve_fit(
                        mod_2comp,
                        Xdata,
                        Ydata,
                        p0=par0,
                        bounds=(lb, ub),
                    )
                    tblgm_map[index] = par_fit[0]
                except RuntimeError:   # pragma: no cover
                    tblgm_map[index] = 0.0


def _multite_create_x_data(ld, pld, te):   # pragma: no cover
    # array for the x values, assuming an arbitrary size based on the PLD
    # and TE vector size
    Xdata = np.zeros((len(pld) * len(te), 3))

    count = 0
    for i in range(len(pld)):
        for j in range(len(te)):
            Xdata[count] = [ld[i], pld[i], te[j]]
            count += 1

    return Xdata
