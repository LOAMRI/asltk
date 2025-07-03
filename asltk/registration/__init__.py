import warnings

import numpy as np

from asltk.asldata import ASLData
from asltk.logging_config import (
    get_logger,
    log_processing_step,
    log_warning_with_context,
)
from asltk.registration.rigid import rigid_body_registration
from asltk.utils import collect_data_volumes


def head_movement_correction(
    asl_data: ASLData, ref_vol: int = 0, verbose: bool = False
):
    logger = get_logger('registration')
    logger.info('Starting head movement correction')

    # Check if the input is a valid ASLData object.
    if not isinstance(asl_data, ASLData):
        error_msg = 'Input must be an ASLData object.'
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Collect all the volumes in the pcasl image
    log_processing_step('Collecting data volumes')
    total_vols, orig_shape = collect_data_volumes(asl_data('pcasl'))
    logger.info(f'Collected {len(total_vols)} volumes for registration')

    # Check if the reference volume is a valid integer based on the ASLData number of volumes.
    if not isinstance(ref_vol, int) or ref_vol >= len(total_vols):
        error_msg = 'ref_vol must be an positive integer based on the total asl data volumes.'
        logger.error(
            f'{error_msg} ref_vol={ref_vol}, total_volumes={len(total_vols)}'
        )
        raise ValueError(error_msg)

    # Apply the rigid body registration to each volume (considering the ref_vol)
    log_processing_step(
        'Applying rigid body registration',
        f'using volume {ref_vol} as reference',
    )
    corrected_vols = []
    trans_mtx = []
    ref_volume = total_vols[ref_vol]

    for idx, vol in enumerate(total_vols):
        logger.debug(f'Correcting volume {idx}')
        if verbose:
            print(f'Correcting volume {idx}...', end='')
        try:
            corrected_vol, trans_m = rigid_body_registration(vol, ref_volume)
            logger.debug(f'Volume {idx} registration successful')
        except Exception as e:
            warning_msg = f'Volume movement no handle by: {e}. Assuming the original data.'
            log_warning_with_context(warning_msg, f'volume {idx}')
            warnings.warn(warning_msg)
            corrected_vol, trans_m = vol, np.eye(4)

        if verbose:
            print('...finished.')
        corrected_vols.append(corrected_vol)
        trans_mtx.append(trans_m)

    # Rebuild the original ASLData object with the corrected volumes
    log_processing_step('Rebuilding corrected volume data')
    corrected_vols = np.stack(corrected_vols).reshape(orig_shape)

    logger.info(
        f'Head movement correction completed successfully for {len(total_vols)} volumes'
    )

    # # Update the ASLData object with the corrected volumes
    # asl_data.set_image(corrected_vols, 'pcasl')

    return corrected_vols, trans_mtx
