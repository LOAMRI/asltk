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
from asltk.utils.io import load_image


def asl_template_registration(
    asl_data: ASLData,
    asl_data_mask: np.ndarray = None,
    atlas_name: str = 'MNI2009',
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

    # if not isinstance(ref_vol, int) or ref_vol < 0:
    #     raise ValueError('ref_vol must be a non-negative integer.')

    total_vols, orig_shape = collect_data_volumes(asl_data('pcasl'))
    # if ref_vol >= len(total_vols):
    #     raise ValueError(
    #         'ref_vol must be a valid index based on the total ASL data volumes.'
    #     )

    if asl_data('m0') is None:
        raise ValueError(
            'M0 image is required for normalization. Please provide an ASLData with a valid M0 image.'
        )

    atlas = BrainAtlas(atlas_name)
    # atlas_img = ants.image_read(atlas.get_atlas()['t1_data']).numpy()
    atlas_img = load_image(atlas.get_atlas()['t1_data'])

    def norm_function(vol, _):
        return space_normalization(
            moving_image=vol,
            template_image=atlas,
            moving_mask=asl_data_mask,
            template_mask=None,
            transform_type='Affine',
            check_orientation=True,
        )

    # Create a new ASLData to allocate the normalized image
    new_asl = asl_data.copy()

    tmp_vol_list = [asl_data('m0')]
    orig_shape = asl_data('m0').shape

    m0_vol_corrected, trans_m0_mtx = __apply_array_normalization(
        tmp_vol_list, 0, norm_function
    )
    new_asl.set_image(m0_vol_corrected[0], 'm0')

    # Apply the normalization transformation to all pcasl volumes
    pcasl_vols, _ = collect_data_volumes(asl_data('pcasl'))
    normalized_pcasl_vols = []
    with Progress() as progress:
        task = progress.add_task(
            '[green]Applying normalization to pcasl volumes...',
            total=len(pcasl_vols),
        )
        for vol in pcasl_vols:
            norm_vol = apply_transformation(
                moving_image=vol,
                reference_image=atlas_img,
                transforms=trans_m0_mtx,
            )
            normalized_pcasl_vols.append(norm_vol)
            progress.update(task, advance=1)

    new_asl.set_image(normalized_pcasl_vols, 'pcasl')

    return new_asl, trans_m0_mtx


def asl_template_registration(
    asl_data: ASLData,
    asl_data_mask: np.ndarray = None,
    atlas_name: str = 'MNI2009',
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

    # if not isinstance(ref_vol, int) or ref_vol < 0:
    #     raise ValueError('ref_vol must be a non-negative integer.')

    total_vols, orig_shape = collect_data_volumes(asl_data('pcasl'))
    # if ref_vol >= len(total_vols):
    #     raise ValueError(
    #         'ref_vol must be a valid index based on the total ASL data volumes.'
    #     )

    if asl_data('m0') is None:
        raise ValueError(
            'M0 image is required for normalization. Please provide an ASLData with a valid M0 image.'
        )

    atlas = BrainAtlas(atlas_name)
    # atlas_img = ants.image_read(atlas.get_atlas()['t1_data']).numpy()
    atlas_img = load_image(atlas.get_atlas()['t1_data'])

    def norm_function(vol, _):
        return space_normalization(
            moving_image=vol,
            template_image=atlas,
            moving_mask=asl_data_mask,
            template_mask=None,
            transform_type='Affine',
            check_orientation=True,
            orientation_verbose=verbose,
        )

    # Create a new ASLData to allocate the normalized image
    new_asl = asl_data.copy()

    tmp_vol_list = [asl_data('m0')]
    orig_shape = asl_data('m0').shape

    m0_vol_corrected, trans_m0_mtx = __apply_array_normalization(
        tmp_vol_list, 0, orig_shape, norm_function, verbose
    )
    new_asl.set_image(m0_vol_corrected[0], 'm0')

    # Apply the normalization transformation to all pcasl volumes
    pcasl_vols, _ = collect_data_volumes(asl_data('pcasl'))
    normalized_pcasl_vols = []
    with Progress() as progress:
        task = progress.add_task(
            '[green]Applying normalization to pcasl volumes...',
            total=len(pcasl_vols),
        )
        for vol in pcasl_vols:
            norm_vol = apply_transformation(
                moving_image=vol,
                reference_image=atlas_img,
                transforms=trans_m0_mtx,
            )
            normalized_pcasl_vols.append(norm_vol)
            progress.update(task, advance=1)

    new_asl.set_image(normalized_pcasl_vols, 'pcasl')

    return new_asl, trans_m0_mtx


def head_movement_correction(
    asl_data: ASLData,
    ref_vol: np.ndarray = None,
    method: str = 'snr',
    roi: np.ndarray = None,
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
        not isinstance(ref_volume, np.ndarray)
        or ref_volume.shape != total_vols[0].shape
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
    corrected_vols_array = np.array(corrected_vols).reshape(
        asl_data('pcasl').shape
    )
    new_asl_data.set_image(corrected_vols_array, 'pcasl')

    return new_asl_data, trans_mtx


# TODO Provavel que tenha que separar esse metodo para o asl_template_registration... revisar depois
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
                _, trans_m = normalization_function(vol, ref_vol)

                # Adjust the transformation matrix
                trans_path = trans_m[0]
                t_matrix = ants.read_transform(trans_path)
                params = t_matrix.parameters * trans_proportions[idx]
                t_matrix.set_parameters(params)
                ants.write_transform(t_matrix, trans_m[0])

                corrected_vol = apply_transformation(vol, ref_vol, trans_m)
            except Exception as e:
                raise RuntimeError(
                    f'[red on white]Error during registration of volume {idx}: {e}[/]'
                )

            corrected_vols.append(corrected_vol)
            trans_mtx.append(trans_m)
            progress.update(task, advance=1)

    # Rebuild the original ASLData object with the corrected volumes
    # orig_shape = orig_shape[1:4]
    # corrected_vols = np.stack(corrected_vols).reshape(orig_shape)

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
    method_values = []
    for vol in total_vols:
        if method == 'snr':
            value = calculate_snr(vol, roi=roi)
        elif method == 'mean':
            value = calculate_mean_intensity(vol, roi=roi)
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
