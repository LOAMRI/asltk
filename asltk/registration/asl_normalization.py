from typing import List, Union

import ants
import numpy as np
from rich.progress import Progress

from asltk.asldata import ASLData
from asltk.data.brain_atlas import BrainAtlas
from asltk.registration import (
    apply_transformation,
    rigid_body_registration,
    space_normalization,
)
from asltk.utils.image_manipulation import (
    collect_data_volumes,
    select_reference_volume,
)
from asltk.utils.image_statistics import (
    calculate_mean_intensity,
    calculate_snr,
)
from asltk.utils.io import ImageIO, clone_image


def asl_template_registration(
    asl_data: ASLData,
    atlas_reference: Union[str, BrainAtlas] = 'MNI2009',
    additional_maps: List[ImageIO] = None,
    asl_data_mask: ImageIO = None,
    verbose: bool = False,
):
    """
    Register ASL data to common atlas space.

    This function applies a elastic normalization to fit the subject head
    space into the atlas template space.


    Note:
        This method takes in consideration the ASLData object, which contains
        the pcasl and/or m0 image. The registration is performed using primarily
        the `m0`image if available, otherwise it uses the `pcasl` image.
        Therefore, choose wisely the `ref_vol` parameter, which should be a valid index
        for the best `pcasl`volume reference to be registered to the atlas.

    Args:
        asl_data: ASLData
            The ASLData object containing the pcasl and/or m0 image to be corrected.
        ref_vol: (int, optional)
            The index of the reference volume to which all other volumes will be registered.
            Defaults to 0.
        asl_data_mask: np.ndarray
            A single volume image mask. This can assist the normalization method to converge
            into the atlas space. If not provided, the full image is adopted.
        atlas_name: str
            The atlas type to be considered. The BrainAtlas class is applied, then choose
            the `atlas_name` based on the ASLtk brain atlas list.
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

    if asl_data('m0') is None:
        raise ValueError(
            'M0 image is required for normalization. Please provide an ASLData with a valid M0 image.'
        )

    if not (
        isinstance(atlas_reference, BrainAtlas)
        or isinstance(atlas_reference, str)
    ):
        raise TypeError(
            'atlas_reference must be a BrainAtlas object or a string.'
        )
    if (
        isinstance(atlas_reference, str)
        and atlas_reference not in BrainAtlas('MNI2009').list_atlas()
    ):
        raise ValueError(
            f"atlas_reference '{atlas_reference}' is not a valid atlas name. "
            f"Available atlases: {BrainAtlas('MNI2009').list_atlas()}"
        )

    if additional_maps is not None:
        if not all(
            [
                isinstance(additional_map, ImageIO)
                and additional_map.get_as_numpy().shape
                == asl_data('m0').get_as_numpy().shape
                for additional_map in additional_maps
            ]
        ):
            raise TypeError(
                'All additional_maps must be ImageIO objects and have the same shape as the M0 image.'
            )
    else:
        additional_maps = []

    if isinstance(atlas_reference, BrainAtlas):
        atlas = atlas_reference
    else:
        atlas = BrainAtlas(atlas_reference)

    atlas_img = ImageIO(atlas.get_atlas()['t1_data'])

    def norm_function(vol, _):
        return space_normalization(
            moving_image=vol,
            template_image=atlas,
            moving_mask=asl_data_mask,
            template_mask=None,
            transform_type='SyN',
            check_orientation=True,
            verbose=verbose,
        )

    # Create a new ASLData to allocate the normalized image
    new_asl = asl_data.copy()

    tmp_vol_list = [asl_data('m0')]

    # Apply the normalization transformation to the M0 volume and update the new ASLData
    m0_vol_corrected, trans_m0_mtx = __apply_array_normalization(
        tmp_vol_list, 0, norm_function, None
    )
    new_asl.set_image(m0_vol_corrected[0], 'm0')

    # Apply the normalization transformation to all chosen volumes
    raw_volumes, _ = collect_data_volumes(asl_data('pcasl'))

    additional_maps_normalized = []
    raw_volumes_normalized = []
    with Progress() as progress:
        task = progress.add_task(
            '[green]Applying normalization to chosen volumes...',
            total=len(raw_volumes) + len(additional_maps),
        )
        for raw in raw_volumes:
            norm_vol = apply_transformation(
                moving_image=raw,
                reference_image=atlas_img,
                transforms=trans_m0_mtx,
            )
            raw_volumes_normalized.append(norm_vol)
            progress.update(task, advance=1)

        for additional_map in additional_maps:
            norm_additional_map = apply_transformation(
                moving_image=additional_map,
                reference_image=atlas_img,
                transforms=trans_m0_mtx,
            )
            additional_maps_normalized.append(norm_additional_map)
            progress.update(task, advance=1)

    # Update the new ASLData with the normalized volumes
    norm_array = np.array(
        [vol.get_as_numpy() for vol in raw_volumes_normalized]
    )
    new_asl.set_image(norm_array, 'pcasl')

    return new_asl, trans_m0_mtx, additional_maps_normalized


def head_movement_correction(
    asl_data: ASLData,
    ref_vol: ImageIO = None,
    method: str = 'snr',
    roi: ImageIO = None,
    verbose: bool = False,
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
        ref_vol: (np.ndarray, optional)
            The reference volume to which all other volumes will be registered.
            If not defined, the `m0` volume will be used.
            In case the `m0` volume is not available, the volume is defined by the method parameter.
        method: (str, optional)
            The method to select the reference volume. Options are 'snr' or 'mean'.
            If 'snr', the volume with the highest SNR is selected.
            If 'mean', the volume with the highest mean signal is selected.
        verbose: (bool, optional)
            If True, prints progress messages. Defaults to False.

    Raises:
        TypeError: If the input is not an ASLData object.
        ValueError: If no valid reference volume is provided.
        RuntimeError: If the normalization fails.

    Returns:
        tuple: ASLData object with corrected volumes and a list of transformation matrices.
    """

    # Check if the input is a valid ASLData object.
    if not isinstance(asl_data, ASLData):
        raise TypeError('Input must be an ASLData object.')

    # Collect all the volumes in the pcasl image
    total_vols, _ = collect_data_volumes(asl_data('pcasl'))
    trans_proportions = _collect_transformation_proportions(
        total_vols, method, roi
    )

    # If ref_vol is not provided, use the m0 volume or the first pcasl volume
    ref_volume = None
    if ref_vol is None:
        if asl_data('m0') is not None:
            ref_volume = asl_data('m0')
        elif total_vols:
            vol_from_method, _ = select_reference_volume(
                asl_data, ref_vol, method=method
            )
            ref_volume = vol_from_method
        else:
            raise ValueError(
                'No valid reference volume provided. Please provide a valid m0 or ASLData volume.'
            )
    else:
        ref_volume = ref_vol

    # Check if the reference volume is a valid volume.
    if (
        not isinstance(ref_volume, ImageIO)
        or ref_volume.get_as_numpy().shape
        != total_vols[0].get_as_numpy().shape
    ):
        raise ValueError(
            'ref_vol must be a valid volume from the total asl data volumes.'
        )

    def norm_function(vol, ref_volume):
        return rigid_body_registration(vol, ref_volume)

    corrected_vols, trans_mtx = __apply_array_normalization(
        total_vols, ref_volume, norm_function, trans_proportions
    )

    new_asl_data = asl_data.copy()
    # Create the new ASLData object with the corrected volumes
    corrected_vols_array = np.array(
        [vol.get_as_numpy() for vol in corrected_vols]
    ).reshape(asl_data('pcasl').get_as_numpy().shape)

    adjusted_pcasl = clone_image(asl_data('pcasl'))
    adjusted_pcasl.update_image_data(corrected_vols_array)
    new_asl_data.set_image(adjusted_pcasl, 'pcasl')

    return new_asl_data, trans_mtx


def __apply_array_normalization(
    total_vols, ref_vol, normalization_function, trans_proportions
):
    corrected_vols = []
    trans_mtx = []
    with Progress() as progress:
        task = progress.add_task(
            '[green]Registering volumes...', total=len(total_vols)
        )
        for idx, vol in enumerate(total_vols):
            try:
                single_correction_vol, trans_m = normalization_function(
                    vol, ref_vol
                )

                trans_path = trans_m[-1]
                t_matrix = ants.read_transform(trans_path)
                if trans_proportions is None:
                    params = t_matrix.parameters
                else:
                    params = t_matrix.parameters * trans_proportions[idx]

                t_matrix.set_parameters(params)
                ants.write_transform(t_matrix, trans_m[-1])

                if isinstance(ref_vol, ImageIO):
                    # Then the normalization is doing by rigid body registration
                    corrected_vol = apply_transformation(vol, ref_vol, trans_m)
                else:
                    # Then the normalization is doing by asl_template_normalization
                    corrected_vol = apply_transformation(
                        vol, single_correction_vol, trans_m
                    )

            except Exception as e:
                raise RuntimeError(
                    f'[red on white]Error during registration of volume {idx}: {e}[/]'
                )

            corrected_vols.append(corrected_vol)
            trans_mtx.append(trans_m)
            progress.update(task, advance=1)

    if isinstance(trans_mtx[0], list):
        # If the transformation list has a inner list, then take the first one
        trans_mtx = trans_mtx[0]

    return corrected_vols, trans_mtx


def _collect_transformation_proportions(total_vols, method, roi):
    """
    Collect method values to be used for matrix transformation balancing.

    Args:
        total_vols (list): List of ASL volumes.
        method (str): Method to use (in accordance to the `select_reference_volume`).
        roi (np.ndarray): Region of interest mask.

    Returns:
        list: List of calculated values based on the method.
    """
    if roi is None:
        # Making a full mask if no ROI is provided
        roi = np.ones_like(total_vols[0].get_as_numpy())

    method_values = []
    for vol in total_vols:
        if method == 'snr':
            value = calculate_snr(vol, roi=ImageIO(image_array=roi))
        elif method == 'mean':
            value = calculate_mean_intensity(vol, roi=ImageIO(image_array=roi))
        else:
            raise ValueError(f'Unknown method: {method}')
        method_values.append(value)

    min_val = np.min(method_values)
    max_val = np.max(method_values)
    if max_val == min_val:
        trans_proportions = np.ones_like(method_values)
    else:
        trans_proportions = (np.array(method_values) - min_val) / (
            max_val - min_val
        )

    return trans_proportions
