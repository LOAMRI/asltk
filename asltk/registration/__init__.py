import ants
import numpy as np
import SimpleITK as sitk

from asltk.asldata import ASLData
from asltk.data.brain_atlas import BrainAtlas
from asltk.logging_config import get_logger

# from asltk.utils.image_manipulation import check_and_fix_orientation
from asltk.utils.io import ImageIO, clone_image


def space_normalization(
    moving_image: ImageIO,
    template_image: BrainAtlas,
    moving_mask: ImageIO = None,
    template_mask: ImageIO = None,
    transform_type: str = 'SyNBoldAff',
    **kwargs,
):
    """
    Perform brain normalization to register the moving image into the
    template image space.

    This function uses ANTsPy to register a moving image to a template
    image. Optional masks can be provided for both images. The
    registration process supports different transformation types.

    This is the base method for space normalization, which can be used
    for different types of images, such as M0, T1w, and ASL images.
    The method is designed to be flexible and can be used for different
    types of images, as long as the moving image and template image are
    provided in the correct format.

    Note:
        For more specific cases, such as ASL data normalization, one can
        use other methods, such as in `asl_normalization` module.

    Note:
        Usually the space normalization is performed between the M0 and T1w
        images. The M0 image is one of the images obtained in the ASL
        acquisition and the T1w image is the anatomical image template.

    Important:
        The `transform_type` parameter allows for different types of
        transformations, such as 'SyN', 'BSpline', etc. The default is 'SyNBoldAff',
        which is suitable for registering ASL images to a T1-weighted template.
        All the definitions of the transformation types can be found in the
        ANTsPy documentation: https://antspy.readthedocs.io/en/latest/registration.html

    Important:
        This method always assumes a template image as a BrainAtlas object.
        One may pass a string with the name of the atlas, and the method will
        automatically load the atlas and use the T1-weighted image as the
        template image. If a different template image is needed, it should be
        passed as a BrainAtlas object, however, it depends on the ASLtk
        Kaggle dataset structure, so it is not recommended to raise an issue
        in the official ASLtk repository if the template image is not presented
        in the BrainAtlas format.

    Parameters
    ----------
    moving_image : np.ndarray
        The moving image.
    template_image : BrainAtlas or str or np.ndarray
        The template image as BrainAtlas object, string with the atlas name or
        a numpy array.
    moving_mask : np.ndarray, optional
        The moving mask in the same space as the moving image. If not provided,
        no mask is used.
    template_mask : np.ndarray, optional
        The template mask in the same space as the template image. If not provided,
        no mask is used.
    transform_type : str, optional
        Type of transformation ('SyN', 'BSpline', etc.). Default is 'SyNBoldAff'.
    verbose : bool, optional
        Whether to print detailed orientation analysis. Default is False.

    Returns
    -------
    normalized_image : np.ndarray
        The moving image transformed into the template image space.
    transform : list
        A list of transformation mapping from moving to template space.
    """
    if not isinstance(moving_image, ImageIO) or not isinstance(
        template_image, (BrainAtlas, str, ImageIO)
    ):
        raise TypeError(
            'moving_image must be an ImageIO object and template_image must be a BrainAtlas object, a string with the atlas name, or an ImageIO object.'
        )

    # Load template image first
    template_array = None
    if isinstance(template_image, BrainAtlas):
        template_file = template_image.get_atlas()['t1_data']
        template_array = ImageIO(template_file)
    elif isinstance(template_image, str):
        template_file = BrainAtlas(template_image).get_atlas()['t1_data']
        template_array = ImageIO(template_file)
        # template_array = ants.image_read('/home/antonio/Imagens/loamri-samples/20240909/mni_2mm.nii.gz')
    elif isinstance(template_image, ImageIO):
        template_array = template_image
    else:
        raise TypeError(
            'template_image must be a BrainAtlas object, a string with the atlas name, or an ImageIO object.'
        )

    if (
        moving_image.get_as_numpy().ndim != 3
        or template_array.get_as_numpy().ndim != 3
    ):
        raise ValueError(
            'Both moving_image and template_image must be 3D arrays.'
        )

    corrected_moving_image = clone_image(moving_image)

    # Load masks if provided
    if isinstance(moving_mask, ImageIO):
        moving_mask = moving_mask.get_as_ants()
    if isinstance(template_mask, ImageIO):
        template_mask = template_mask.get_as_ants()

    # Perform registration
    registration = ants.registration(
        fixed=template_array.get_as_ants(),
        moving=corrected_moving_image.get_as_ants(),
        type_of_transform=transform_type,
        mask=moving_mask,
        mask_fixed=template_mask,
        **kwargs,  # Additional parameters for ants.registration
    )

    # Passing the warped image and forward transforms
    out_warped = clone_image(template_array)
    ants_numpy = registration['warpedmovout'].numpy()
    out_warped.update_image_data(np.transpose(ants_numpy, (2, 1, 0)))

    return out_warped, registration['fwdtransforms']


def rigid_body_registration(
    fixed_image: ImageIO,
    moving_image: ImageIO,
    moving_mask: ImageIO = None,
    template_mask: ImageIO = None,
):
    """
    Register two images using a rigid body transformation. This methods applies
    a Euler 3D transformation in order to register the moving image to the
    fixed image.

    Note:
        The registration assumes that the moving image can be adjusted using
        only rotation and translation, without any scaling or shearing. This
        is suitable for cases in algiment among temporal volumes, such as in
        ASL data, where the images are acquired in the same space and only
        small movements are expected.

    Args:
        fixed_image: np.ndarray
            The fixed image as the reference space.
        moving_image: np.ndarray
            The moving image to be registered.
        moving_mask: np.ndarray, optional
            The mask of the moving image. If not provided, the moving image
            will be used as the mask.
        template_mask: np.ndarray, optional
            The mask of the fixed image. If not provided, the fixed image
            will be used as the mask.

    Raises:
        Exception: fixed_image and moving_image must be a numpy array.
        Exception: moving_mask must be a numpy array.
        Exception: template_mask must be a numpy array.

    Returns
    -------
    normalized_image : np.ndarray
        The moving image transformed into the template image space.
    transforms : list
        A list of transformation mapping from moving to template space.
    """
    if not isinstance(fixed_image, ImageIO) or not isinstance(
        moving_image, ImageIO
    ):
        raise Exception(
            'fixed_image and moving_image must be an ImageIO object.'
        )

    if moving_mask is not None and not isinstance(moving_mask, ImageIO):
        raise Exception('moving_mask must be an ImageIO object.')
    if template_mask is not None and not isinstance(template_mask, ImageIO):
        raise Exception('template_mask must be an ImageIO object.')

    normalized_image, trans_maps = space_normalization(
        moving_image,
        fixed_image,
        transform_type='Rigid',
        moving_mask=moving_mask,
        template_mask=template_mask,
    )

    return normalized_image, trans_maps


def affine_registration(
    fixed_image: ImageIO,
    moving_image: ImageIO,
    moving_mask: ImageIO = None,
    template_mask: ImageIO = None,
    fast_method: bool = True,
):
    """
    Register two images using an affine transformation. This method applies
    a 3D affine transformation in order to register the moving image to the
    fixed image.

    Args:
        fixed_image: np.ndarray
            The fixed image as the reference space.
        moving_image: np.ndarray
            The moving image to be registered.
        moving_mask: np.ndarray, optional
            The mask of the moving image. If not provided, the moving image
            will be used as the mask.
        template_mask: np.ndarray, optional
            The mask of the fixed image. If not provided, the fixed image
            will be used as the mask.

    Raises:
        Exception: fixed_image and moving_image must be a numpy array.

    Returns
    -------
    resampled_image : np.ndarray
        The moving image transformed into the template image space.
    transformation_matrix : np.ndarray
        The transformation matrix mapping from moving to template space.
    """
    if not isinstance(fixed_image, ImageIO) or not isinstance(
        moving_image, ImageIO
    ):
        raise Exception(
            'fixed_image and moving_image must be an ImageIO object.'
        )
    if moving_mask is not None and not isinstance(moving_mask, ImageIO):
        raise Exception('moving_mask must be an ImageIO object.')
    if template_mask is not None and not isinstance(template_mask, ImageIO):
        raise Exception('template_mask must be an ImageIO object.')

    affine_type = 'AffineFast' if fast_method else 'Affine'
    warped_image, transformation_matrix = space_normalization(
        moving_image,
        fixed_image,
        transform_type=affine_type,
        moving_mask=moving_mask,
        template_mask=template_mask,
    )

    return warped_image, transformation_matrix


def apply_transformation(
    moving_image: ImageIO,
    reference_image: ImageIO,
    transforms: list,
    **kwargs,
):
    """
    Apply a transformation list set to an image.

    This method applies a list of transformations to a moving image
    to align it with a reference image. The transformations are typically
    obtained from a registration process, such as rigid or affine
    registration.

    Note:
        The `transforms` parameter should be a list of transformation matrices
        obtained from a registration process. The transformations are applied
        in the order they are provided in the list.

    Tip:
        Additional parameters can be passed to the `ants.apply_transforms`
        function using the `kwargs` parameter. This allows for customization of
        the transformation process, such as specifying interpolation methods,
        handling of missing data, etc. See more in the ANTsPy documentation:
        https://antspy.readthedocs.io/en/latest/registration.html#ants.apply_transforms

    Args:
        image: np.ndarray
            The image to be transformed.
        reference_image: np.ndarray
            The reference image to which the transformed image will be aligned.
            If not provided, the original image will be used as the reference.
        transforms: list
            The transformation matrix list.

    Returns:
        transformed_image: np.ndarray
            The transformed image.
    """
    if not isinstance(moving_image, ImageIO):
        raise TypeError('moving image must be an ImageIO object.')

    if not isinstance(reference_image, (ImageIO, BrainAtlas)):
        raise TypeError(
            'reference_image must be an ImageIO object or a BrainAtlas object.'
        )

    if isinstance(reference_image, BrainAtlas):
        reference_image = ImageIO(reference_image.get_atlas()['t1_data'])

    if not isinstance(transforms, list):
        raise TypeError(
            'transforms must be a list of transformation matrices.'
        )

    corr_image = ants.apply_transforms(
        fixed=reference_image.get_as_ants(),
        moving=moving_image.get_as_ants(),
        transformlist=transforms,
        **kwargs,  # Additional parameters for ants.apply_transforms
    )

    out_image = clone_image(reference_image)
    out_image.update_image_data(np.transpose(corr_image.numpy(), (2, 1, 0)))

    return out_image
