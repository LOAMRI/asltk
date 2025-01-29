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
        """Defines whether a brain a mask is applied to the MultiDW_ASLMapping
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
    ):
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

        return {
            'cbf': self._cbf_map,
            'cbf_norm': self._cbf_map * (60 * 60 * 1000),
            'att': self._att_map,
            'a1': self._A1,
            'd1': self._D1,
            'a2': self._A2,
            'd2': self._D2,
            'kw': self._kw,
        }

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
