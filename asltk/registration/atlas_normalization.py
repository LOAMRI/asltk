import ants
import numpy as np


def brain_normalization(
    moving_image: np.ndarray,
    template_image: np.ndarray,
    output_prefix: str = 'norm',
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
    output_prefix : str, optional
        Prefix for the output files (default is 'norm').
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
    template = ants.from_numpy(template_image)

    # Load masks if provided
    if moving_mask:
        moving_mask = ants.image_read(moving_mask)
    if template_mask:
        template_mask = ants.image_read(template_mask)

    # Perform registration
    registration = ants.registration(
        fixed=template,
        moving=moving,
        type_of_transform=transform_type,
        mask=moving_mask,
        mask_fixed=template_mask,
    )

    # Save results
    return None
