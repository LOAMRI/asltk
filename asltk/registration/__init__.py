import ants
import numpy as np
import SimpleITK as sitk

from asltk.asldata import ASLData
from asltk.data.brain_atlas import BrainAtlas
from asltk.logging_config import get_logger
from asltk.utils.image_manipulation import check_and_fix_orientation
from asltk.utils.io import load_image


# TODO Montar classe para fazer o coregistro de ASL
class ASLRegistration:

    # Pipeline
    # inputs: ASLData (com m0 e pcasl), BrainAtlas, resolution (1 or 2 mm)
    # Tomar m0 e comparar orientação com o template
    # Se necessário, corrigir orientação do template para estar coerente com o m0 (salvar a transformação e aplicar para os labels)
    # Realizar o registro do m0 no template
    # com a transformação do m0, deixar salvo como parametro do objeto da classe
    # Ter metodos para aplicar transformação para o pcasl, ou mapas gerados pelo CBFMapping, MultiTE, etc.

    def __init__(self):
        pass


def space_normalization(
    moving_image: np.ndarray,
    template_image: BrainAtlas,
    moving_mask: np.ndarray = None,
    template_mask: np.ndarray = None,
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
        For more specfiic cases, such as ASL data normalization, one can
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
    check_orientation : bool, optional
        Whether to automatically check and fix orientation mismatches between
        moving and template images. Default is True.
    verbose : bool, optional
        Whether to print detailed orientation analysis. Default is False.

    Returns
    -------
    normalized_image : np.ndarray
        The moving image transformed into the template image space.
    transform : list
        A list of transformation mapping from moving to template space.
    """
    if not isinstance(moving_image, np.ndarray) or not isinstance(
        template_image, (BrainAtlas, str, np.ndarray)
    ):
        raise TypeError(
            'moving_image must be a numpy array and template_image must be a BrainAtlas object, a string with the atlas name, or a numpy array.'
        )

    # Take optional parameters
    check_orientation = kwargs.get('check_orientation', True)
    verbose = kwargs.get('verbose', False)

    logger = get_logger('registration')
    logger.info('Starting space normalization')

    # Load template image first
    # TODO PROBLEMA PRINCIPAL: A leitura de imagens para numpy faz a perda da origen e spacing, para fazer o corregistro é preciso acertar a orientação da imagem com relação a origem (flip pela origem) para que ambas estejam na mesma orientação visual
    # TODO Pensar em como será a utilização do corregistro para o ASLTK (assume que já está alinhado? ou tenta alinhar imagens check_orientation?)
    template_array = None
    if isinstance(template_image, BrainAtlas):
        template_file = template_image.get_atlas()['t1_data']
        template_array = load_image(template_file)
    elif isinstance(template_image, str):
        template_file = BrainAtlas(template_image).get_atlas()['t1_data']
        template_array = load_image(template_file)
        # template_array = ants.image_read('/home/antonio/Imagens/loamri-samples/20240909/mni_2mm.nii.gz')
    elif isinstance(template_image, np.ndarray):
        template_array = template_image
    else:
        raise TypeError(
            'template_image must be a BrainAtlas object, a string with the atlas name, or a numpy array.'
        )

    if moving_image.ndim != 3 or template_array.ndim != 3:
        raise ValueError(
            'Both moving_image and template_image must be 3D arrays.'
        )

    corrected_moving_image = moving_image
    orientation_transform = None

    # TODO VERIICAR SE CHECK_ORIENTATION ESTA CERTO... USAR sitk.FlipImageFilter usando a Origen da image (Slicer da certo assim)
    if check_orientation:
        (
            corrected_moving_image,
            orientation_transform,
        ) = check_and_fix_orientation(
            moving_image, template_array, verbose=verbose
        )
        if verbose and orientation_transform:
            print(f'Applied orientation correction: {orientation_transform}')

    # Convert to ANTs images

    moving = ants.from_numpy(corrected_moving_image)
    template = ants.from_numpy(template_array)

    # Load masks if provided
    if isinstance(moving_mask, np.ndarray):
        moving_mask = ants.from_numpy(moving_mask)
    if isinstance(template_mask, np.ndarray):
        template_mask = ants.from_numpy(template_mask)

    # TODO Vericicar se ants.registration consegue colocar o TransformInit como Centro de Massa!'
    # Perform registration
    registration = ants.registration(
        fixed=template,
        moving=moving,
        type_of_transform=transform_type,
        mask=moving_mask,
        mask_fixed=template_mask,
        **kwargs,  # Additional parameters for ants.registration
    )

    # Passing the warped image and forward transforms
    return registration['warpedmovout'].numpy(), registration['fwdtransforms']


def rigid_body_registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    moving_mask: np.ndarray = None,
    template_mask: np.ndarray = None,
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
    if not isinstance(fixed_image, np.ndarray) or not isinstance(
        moving_image, np.ndarray
    ):
        raise Exception('fixed_image and moving_image must be a numpy array.')

    if moving_mask is not None and not isinstance(moving_mask, np.ndarray):
        raise Exception('moving_mask must be a numpy array.')
    if template_mask is not None and not isinstance(template_mask, np.ndarray):
        raise Exception('template_mask must be a numpy array.')

    normalized_image, trans_maps = space_normalization(
        moving_image,
        fixed_image,
        transform_type='Rigid',
        moving_mask=moving_mask,
        template_mask=template_mask,
    )

    return normalized_image, trans_maps


def affine_registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    moving_mask: np.ndarray = None,
    template_mask: np.ndarray = None,
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
    if not isinstance(fixed_image, np.ndarray) or not isinstance(
        moving_image, np.ndarray
    ):
        raise Exception('fixed_image and moving_image must be a numpy array.')
    if moving_mask is not None and not isinstance(moving_mask, np.ndarray):
        raise Exception('moving_mask must be a numpy array.')
    if template_mask is not None and not isinstance(template_mask, np.ndarray):
        raise Exception('template_mask must be a numpy array.')

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
    moving_image: np.ndarray,
    reference_image: np.ndarray,
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
    # TODO handle kwargs for additional parameters based on ants.apply_transforms
    if not isinstance(moving_image, np.ndarray):
        raise TypeError('moving image must be numpy array.')

    if not isinstance(reference_image, (np.ndarray, BrainAtlas)):
        raise TypeError(
            'reference_image must be a numpy array or a BrainAtlas object.'
        )
    elif isinstance(reference_image, BrainAtlas):
        reference_image = load_image(reference_image.get_atlas()['t1_data'])

    if not isinstance(transforms, list):
        raise TypeError(
            'transforms must be a list of transformation matrices.'
        )

    corr_image = ants.apply_transforms(
        fixed=ants.from_numpy(reference_image),
        moving=ants.from_numpy(moving_image),
        transformlist=transforms,
    )

    return corr_image.numpy()
