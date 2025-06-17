import ants
import numpy as np
from rich.console import Console

from asltk.data.brain_atlas import BrainAtlas


def space_normalization(
    moving_image: np.ndarray,
    template_image: BrainAtlas,
    moving_mask: np.ndarray = None,
    template_mask: np.ndarray = None,
    transform_type: str = 'SyN',
):
    """
    Perform brain normalization to register the moving image into the
    template image space.

    This function uses ANTsPy to register a moving image to a template
    image. Optional masks can be provided for both images. The
    registration process supports different interpolation methods,
    transformation types, and a configurable number of iterations.

    Parameters
    ----------
    moving_image : np.ndarray
        Path to the moving image.
    template_image : np.ndarray
        Path to the template image.
    moving_mask : np.ndarray, optional
        Path to the moving mask.
    template_mask : np.ndarray, optional
        Path to the template mask.
    interpolation : str, optional
        Interpolation method ('linear', 'nearestNeighbor', etc.). Default is 'linear'.
    transform_type : str, optional
        Type of transformation ('SyN', 'BSpline', etc.). Default is 'SyN'.
    num_iterations : int, optional
        Number of iterations for the registration. Default is 1000.

    Returns
    -------
    normalized_image : np.ndarray
        The moving image transformed into the template image space.
    transform : object
        The transformation mapping from moving to template space.
    inverse_transform : object
        The inverse transformation mapping from template to moving space.
    """

    # Load images
    moving = ants.from_numpy(moving_image)

    # Get template image from BrainAtlas
    if isinstance(template_image, BrainAtlas):
        template_image = template_image.get_atlas()['t1_data']
    elif isinstance(template_image, str):
        template_image = BrainAtlas(template_image).get_atlas()['t1_data']
    template = ants.image_read(template_image)

    # Load masks if provided
    if moving_mask:
        moving_mask = ants.image_read(moving_mask)
    if template_mask:
        template_mask = ants.image_read(template_mask)

    # Perform registration
    console = Console()
    with console.status(
        '[bold green]Calculating registration...', spinner='dots'
    ):
        registration = ants.registration(
            fixed=template,
            moving=moving,
            type_of_transform=transform_type,
            mask=moving_mask,
            mask_fixed=template_mask,
        )

    # Passing the warped image and forward transforms
    console.log('[bold green]Registration completed successfully.')
    return registration['warpedmovout'].numpy(), registration['fwdtransforms']
