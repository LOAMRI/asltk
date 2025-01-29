import warnings

import numpy as np

from asltk.asldata import ASLData
from asltk.registration.rigid import rigid_body_registration
from asltk.utils import collect_data_volumes


def head_movement_correction(
    asl_data: ASLData, ref_vol: int = 0, verbose: bool = False
):
    # Check if the input is a valid ASLData object.
    if not isinstance(asl_data, ASLData):
        raise TypeError('Input must be an ASLData object.')

    # Collect all the volumes in the pcasl image
    total_vols, orig_shape = collect_data_volumes(asl_data('pcasl'))

    # Check if the reference volume is a valid integer based on the ASLData number of volumes.
    if not isinstance(ref_vol, int) or ref_vol >= len(total_vols):
        raise ValueError(
            'ref_vol must be an positive integer based on the total asl data volumes.'
        )

    # Apply the rigid body registration to each volume (considering the ref_vol)
    corrected_vols = []
    trans_mtx = []
    ref_volume = total_vols[ref_vol]

    for idx, vol in enumerate(total_vols):
        if verbose:
            print(f'Correcting volume {idx}...', end='')
        try:
            corrected_vol, trans_m = rigid_body_registration(vol, ref_volume)
        except Exception as e:
            warnings.warn(
                f'Volume movement no handle by: {e}. Assuming the original data.'
            )
            corrected_vol, trans_m = vol, np.eye(4)

        if verbose:
            print('...finished.')
        corrected_vols.append(corrected_vol)
        trans_mtx.append(trans_m)

    # Rebuild the original ASLData object with the corrected volumes
    corrected_vols = np.stack(corrected_vols).reshape(orig_shape)

    # # Update the ASLData object with the corrected volumes
    # asl_data.set_image(corrected_vols, 'pcasl')

    return corrected_vols, trans_mtx
