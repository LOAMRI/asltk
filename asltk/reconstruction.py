import warnings

import numpy as np
from rich.progress import track

from asltk.asldata import ASLData
from asltk.mri_parameters import MRIParameters

# TODO Opcao para aplicar filtro no mapa de saida (Gauss, etc)
# TODO Brain mask como input ou opção de processamento?


class CBFMapping(MRIParameters):
    def __init__(self, asl_data: ASLData) -> None:
        super().__init__()
        self._asl_data = asl_data

    def set_brain_mask(self, brain_mask: np.ndarray, label: int = 1):
        # TODO Add flag to create new mask using fsl.bet
        self._check_mask_values(brain_mask, label)

        binary_mask = (brain_mask == label).astype(np.uint8) * label
        self._brain_mask = binary_mask

    def _check_mask_values(self, mask, label):
        unique_values = np.unique(mask)
        if unique_values.size > 2:
            warnings.warn(
                'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
                UserWarning,
            )

        label_ok = False
        for value in unique_values:
            if label == value:
                label_ok = True
                break
        if not label_ok:
            raise ValueError('Label value is not found in the mask provided.')
