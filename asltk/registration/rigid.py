import numpy as np
import SimpleITK as sitk


def rigid_body_registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    interpolator=sitk.sitkLinear,
    iterations: int = 5000,
    converge_min: float = 1e-8,
):
    """
    Register two images using a rigid body transformation. This methods applies
    a Euler 3D transformation in order to register the moving image to the
    fixed image.

    The optimization method used is the Gradient Descent.

    Note:
        The registration process is based on the SimpleITK library. More details
        on how the registration process works can be found at: [Registration Overview](https://simpleitk.readthedocs.io/en/master/registrationOverview.html)

    Args:
        fixed_image (np.ndarray): The fixed image as the reference space.
        moving_image (np.ndarray): The moving image to be registered.
        interpolator (sitk.Interpolator, optional): The interpolation method used in the registration process. Defaults to sitk.sitkLinear.

    Raises:
        Exception: fixed_image and moving_image must be a numpy array.

    Returns:
        numpy.ndarray: The resampled image.
        numpy.ndarray: The transformation matrix.
    """

    # Check if the fixed_image is a numpy array.
    if not isinstance(fixed_image, np.ndarray) or not isinstance(
        moving_image, np.ndarray
    ):
        raise Exception('fixed_image and moving_image must be a numpy array.')

    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    # Create the registration method.
    registration_method = sitk.ImageRegistrationMethod()

    # Initialize the registration method.
    registration_transform = sitk.Euler3DTransform()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        registration_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method.SetInitialTransform(initial_transform)

    # Set the metric.
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Set the optimizer.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=converge_min,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Set the interpolator.
    registration_method.SetInterpolator(interpolator)

    # Execute the registration.
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Convert the final transform to a numpy array.
    transform_matrix = np.array(final_transform.GetMatrix()).reshape(3, 3)

    # Create a 4x4 transformation matrix.
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = transform_matrix
    transformation_matrix[:3, 3] = final_transform.GetTranslation()

    # Resample the moving image.
    resampled_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        interpolator,
        0.0,
        moving_image.GetPixelID(),
    )

    resampled_image = sitk.GetArrayFromImage(resampled_image)
    return resampled_image, transformation_matrix


# def affine_registration(fixed_image: np.ndarray, moving_image: np.ndarray, interpolator=sitk.sitkLinear, iterations: int = 5000, converge_min: float = 1e-8):

#     # Check if the fixed_image is a numpy array.
#     if not isinstance(fixed_image, np.ndarray) or not isinstance(moving_image, np.ndarray):
#         raise Exception('fixed_image and moving_image must be a numpy array.')

#     fixed_image = sitk.GetImageFromArray(fixed_image)
#     moving_image = sitk.GetImageFromArray(moving_image)

#     # Create the registration method.
#     registration_method = sitk.ImageRegistrationMethod()

#     # Initialize the registration method.
#     registration_transform = sitk.AffineTransform(3)
#     initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, registration_transform,
#                                                           sitk.CenteredTransformInitializerFilter.GEOMETRY)
#     registration_method.SetInitialTransform(initial_transform)

#     # Set the metric.
#     registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#     registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
#     registration_method.SetMetricSamplingPercentage(0.01)

#     # Set the optimizer.
#     registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=iterations,
#                                                       convergenceMinimumValue=converge_min, convergenceWindowSize=10)
#     registration_method.SetOptimizerScalesFromPhysicalShift()

#     # Set the interpolator.
#     registration_method.SetInterpolator(interpolator)

#     # Execute the registration.
#     final_transform = registration_method.Execute(fixed_image, moving_image)

#     # Convert the final transform to a numpy array.
#     transformation_matrix = np.array(final_transform.GetMatrix()).reshape(3, 3)

#     # Resample the moving image.
#     resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, interpolator, 0.0,
#                                     moving_image.GetPixelID())

#     resampled_image = sitk.GetArrayFromImage(resampled_image)
#     return resampled_image, transformation_matrix
