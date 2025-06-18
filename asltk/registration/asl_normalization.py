import numpy as np
from rich import print

from asltk.asldata import ASLData
from asltk.data.brain_atlas import BrainAtlas
from asltk.registration import rigid_body_registration, space_normalization
from asltk.utils import collect_data_volumes


def asl_template_registration(
    asl_data: ASLData,
    ref_vol: int = 0,
    asl_data_mask: np.ndarray = None,
    atlas_name: str = 'MNI2009',
    verbose: bool = False,
):
    """
    Register ASL data to MNI space using rigid body registration.

    This function applies rigid body registration to correct head movement
    in ASL data. It registers each volume in the ASL data to a reference volume.

    Args:
        asl_data: ASLData
            The ASLData object containing the pcasl and/or m0 image to be corrected.
        ref_vol: (int, optional)
            The index of the reference volume to which all other volumes will be registered.
            Defaults to 0.
        verbose: (bool, optional)
            If True, prints progress messages. Defaults to False.

    Raises:
        TypeError: If the input is not an ASLData object.
        ValueError: If ref_vol is not a valid index.
        RuntimeError: If an error occurs during registration.

    Returns:
        tuple: ASLData object with corrected volumes and a list of transformation matrices.
    """
    if not isinstance(asl_data, ASLData):
        raise TypeError('Input must be an ASLData object.')

    if not isinstance(ref_vol, int) or ref_vol < 0:
        raise ValueError('ref_vol must be a non-negative integer.')

    total_vols, orig_shape = collect_data_volumes(asl_data('pcasl'))

    if ref_vol >= len(total_vols):
        raise ValueError(
            'ref_vol must be a valid index based on the total ASL data volumes.'
        )

    atlas = BrainAtlas(atlas_name=atlas_name)

    def norm_function(vol, _):
        return space_normalization(
            moving_image=vol,
            template_image=atlas_name,
            moving_mask=asl_data_mask,
            template_mask=None,
            transform_type='SyNBoldAff',
        )

    corrected_vols, trans_mtx = __apply_array_normalization(
        total_vols, ref_vol, orig_shape, norm_function, verbose
    )

    # TODO Make the verbose output more informative
    if asl_data('m0') is not None:
        corrected_m0_vols, _ = __apply_array_normalization(
            asl_data('m0'), ref_vol, orig_shape, norm_function, verbose
        )
        asl_data.set_image(corrected_m0_vols, 'm0')
    # Update the ASLData object with the corrected volumes
    asl_data.set_image(corrected_vols, 'pcasl')

    return asl_data, trans_mtx


def head_movement_correction(
    asl_data: ASLData, ref_vol: int = 0, verbose: bool = False
):
    """
    Correct head movement in ASL data using rigid body registration.

    This function applies rigid body registration to correct head movement
    in ASL data. It registers each volume in the ASL data to a reference volume.

    Hence, it can be helpfull to correct for head movements that may have
    occurred during the acquisition of ASL data.
    Note:
        The reference volume is selected based on the `ref_vol` parameter,
        which should be a valid index of the total number of volumes in the ASL data.
        The `ref_vol` value for 0 means that the first volume will be used as the reference.

    Args:
        asl_data: ASLData)
            The ASLData object containing the pcasl image to be corrected.
        ref_vol: (int, optional)
            The index of the reference volume to which all other volumes will be registered.
            Defaults to 0.
        verbose: (bool, optional)
            If True, prints progress messages. Defaults to False.

    Raises:
        TypeError: _description_
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        tuple: ASLData object with corrected volumes and a list of transformation matrices.
    """

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

    def norm_function(vol, ref_volume):
        return rigid_body_registration(vol, ref_volume)

    corrected_vols, trans_mtx = __apply_array_normalization(
        total_vols, ref_vol, orig_shape, norm_function, verbose
    )

    # TODO The corrected volumes should be set in the ASLData object.
    # # Update the ASLData object with the corrected volumes
    asl_data.set_image(corrected_vols, 'pcasl')

    return asl_data, trans_mtx


def __apply_array_normalization(
    total_vols, ref_vol, orig_shape, normalization_function, verbose=False
):
    # Apply the rigid body registration to each volume (considering the ref_vol)
    corrected_vols = []
    trans_mtx = []
    ref_volume = (
        total_vols[ref_vol] if isinstance(total_vols, list) else total_vols
    )

    # TODO build a progress bar using rich
    for idx, vol in enumerate(total_vols):
        if verbose:
            print(f'[b green]Correcting volume {idx}...[/]', end='')
        try:
            corrected_vol, trans_m = normalization_function(vol, ref_volume)
        except Exception as e:
            raise RuntimeError(
                f'[red]Error during registration of volume {idx}: {e}[/red]'
            ) from e

        if verbose:
            print('[b green]...finished.[/]')
        corrected_vols.append(corrected_vol)
        trans_mtx.append(trans_m)

    # Rebuild the original ASLData object with the corrected volumes
    corrected_vols = np.stack(corrected_vols).reshape(orig_shape)

    return corrected_vols, trans_mtx
