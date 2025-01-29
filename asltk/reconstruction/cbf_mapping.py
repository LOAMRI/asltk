import warnings
from multiprocessing import Array, Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from rich.progress import Progress
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.aux_methods import _check_mask_values
from asltk.models.signal_dynamic import asl_model_buxton
from asltk.mri_parameters import MRIParameters

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
        _check_mask_values(brain_mask, label, self._asl_data('m0').shape)

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
        ub=[1.0, 5000.0],
        lb=[0.0, 0.0],
        par0=[1e-5, 1000],
        cores: int = cpu_count(),
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
            cores (int, optional): Defines how many CPU threads can be used for the class. Defaults is using all the availble threads.

        Returns:
            (dict): A dictionary with 'cbf', 'att' and 'cbf_norm'
        """
        if (cores < 0) or (cores > cpu_count()) or not isinstance(cores, int):
            raise ValueError(
                'Number of proecess must be at least 1 and less than maximum cores availble.'
            )
        if (
            len(self._asl_data.get_ld()) == 0
            or len(self._asl_data.get_pld()) == 0
        ):
            raise ValueError('LD or PLD list of values must be provided.')
        # TODO Testar se retirando esse if do LD PLD sizes, continua rodando... isso é erro do ASLData

        global asl_data, brain_mask
        asl_data = self._asl_data
        brain_mask = self._brain_mask

        BuxtonX = [self._asl_data.get_ld(), self._asl_data.get_pld()]

        x_axis, y_axis, z_axis = (
            self._asl_data('m0').shape[2],
            self._asl_data('m0').shape[1],
            self._asl_data('m0').shape[0],
        )

        cbf_map_shared = Array('d', z_axis * y_axis * x_axis, lock=False)
        att_map_shared = Array('d', z_axis * y_axis * x_axis, lock=False)

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

        return {
            'cbf': self._cbf_map,
            'cbf_norm': self._cbf_map * (60 * 60 * 1000),
            'att': self._att_map,
        }


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
