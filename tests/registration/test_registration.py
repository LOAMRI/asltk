import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.data.brain_atlas import BrainAtlas
from asltk.registration import (
    affine_registration,
    apply_transformation,
    rigid_body_registration,
    space_normalization,
)
from asltk.utils import load_image

SEP = os.sep
M0_ORIG = (
    f'tests' + SEP + 'files' + SEP + 'registration' + SEP + 'm0_mean.nii.gz'
)
M0_RIGID = (
    f'tests'
    + SEP
    + 'files'
    + SEP
    + 'registration'
    + SEP
    + 'm0_mean-rigid-25degrees.nrrd'
)
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'


def test_rigid_body_registration_run_sucess():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    resampled_image, _ = rigid_body_registration(img_orig, img_rot)

    assert resampled_image.shape == img_orig.shape


@pytest.mark.parametrize(
    'img_orig', [('invalid_image'), ([1, 2, 3]), (['a', 1, 5.23])]
)
def test_rigid_body_registration_error_fixed_image_is_not_numpy_array(
    img_orig,
):
    img_rot = load_image(M0_RIGID)

    with pytest.raises(Exception) as e:
        rigid_body_registration(img_orig, img_rot)

    assert (
        str(e.value) == 'fixed_image and moving_image must be a numpy array.'
    )


@pytest.mark.parametrize(
    'img_rot', [('invalid_image'), ([1, 2, 3]), (['a', 1, 5.23])]
)
def test_rigid_body_registration_error_fixed_image_is_not_numpy_array(img_rot):
    img_orig = load_image(M0_ORIG)

    with pytest.raises(Exception) as e:
        rigid_body_registration(img_orig, img_rot)

    assert (
        str(e.value) == 'fixed_image and moving_image must be a numpy array.'
    )


def test_rigid_body_registration_output_registration_matrix_success():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    _, trans_matrix = rigid_body_registration(img_orig, img_rot)

    assert isinstance(trans_matrix[0], str)


def test_rigid_body_registration_raise_exception_if_moving_mask_not_numpy():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    with pytest.raises(Exception) as e:
        rigid_body_registration(img_orig, img_rot, moving_mask='invalid_mask')

    assert str(e.value) == 'moving_mask must be a numpy array.'


def test_rigid_body_registration_raise_exception_if_template_mask_not_numpy():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    with pytest.raises(Exception) as e:
        rigid_body_registration(
            img_orig, img_rot, template_mask='invalid_mask'
        )

    assert str(e.value) == 'template_mask must be a numpy array.'


def test_space_normalization_success():
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)

    # Use the ASLData object directly
    normalized_image, transform = space_normalization(
        pcasl_orig('m0'), template_image='MNI2009'
    )

    assert isinstance(normalized_image, np.ndarray)
    assert normalized_image.shape == (182, 218, 182)
    assert len(transform) == 2


def test_space_normalization_raise_exception_if_fixed_image_not_numpy():
    with pytest.raises(Exception) as e:
        space_normalization('invalid_image', template_image='MNI2009')

    assert (
        'moving_image must be a numpy array and template_image must be a BrainAtlas object'
        in str(e.value)
    )


def test_space_normalization_raise_exception_if_template_image_not_a_valid_BrainAtlas_option():
    img_orig = load_image(M0_ORIG)

    with pytest.raises(Exception) as e:
        space_normalization(img_orig, template_image='invalid_image')

    assert 'Template image invalid_image is not a valid' in str(e.value)


def test_space_normalization_success_passing_template_image_as_BrainAtlas_option():
    img_orig = load_image(M0_ORIG)

    # Use the BrainAtlas object directly
    normalized_image, transform = space_normalization(
        img_orig, template_image='MNI2009'
    )

    assert isinstance(normalized_image, np.ndarray)
    assert normalized_image.shape == (182, 218, 182)
    assert len(transform) == 2


def test_space_normalization_success_passing_template_image_as_BrainAtlas_object():
    img_orig = load_image(M0_ORIG)
    atlas = BrainAtlas(atlas_name='MNI2009')

    # Use the BrainAtlas object directly
    normalized_image, transform = space_normalization(
        img_orig, template_image=atlas
    )

    assert isinstance(normalized_image, np.ndarray)
    assert normalized_image.shape == (182, 218, 182)
    assert len(transform) == 2


def test_affine_registration_success():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    resampled_image, _ = affine_registration(img_orig, img_rot)

    assert (
        np.mean(np.subtract(img_orig, resampled_image))
        < np.mean(img_orig) * 0.5
    )


def test_affine_registration_raise_exception_if_fixed_image_not_numpy():
    img_rot = load_image(M0_RIGID)

    with pytest.raises(Exception) as e:
        affine_registration('invalid_image', img_rot)

    assert (
        str(e.value) == 'fixed_image and moving_image must be a numpy array.'
    )


def test_affine_registration_raise_exception_if_moving_image_not_numpy():
    img_orig = load_image(M0_ORIG)

    with pytest.raises(Exception) as e:
        affine_registration(img_orig, 'invalid_image')

    assert (
        str(e.value) == 'fixed_image and moving_image must be a numpy array.'
    )


def test_affine_registration_raise_exception_if_moving_mask_not_numpy():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    with pytest.raises(Exception) as e:
        affine_registration(img_orig, img_rot, moving_mask='invalid_mask')

    assert str(e.value) == 'moving_mask must be a numpy array.'


def test_affine_registration_raise_exception_if_template_mask_not_numpy():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    with pytest.raises(Exception) as e:
        affine_registration(img_orig, img_rot, template_mask='invalid_mask')

    assert str(e.value) == 'template_mask must be a numpy array.'


def test_affine_registration_fast_method():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    resampled_image, _ = affine_registration(
        img_orig, img_rot, fast_method=True
    )

    assert isinstance(resampled_image, np.ndarray)
    assert resampled_image.shape == img_rot.shape
    assert np.mean(np.abs(img_orig - resampled_image)) < 0.5 * np.mean(
        img_orig
    )


def test_affine_registration_slow_method():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)

    resampled_image, _ = affine_registration(
        img_orig, img_rot, fast_method=False
    )

    assert isinstance(resampled_image, np.ndarray)
    assert resampled_image.shape == img_rot.shape
    assert np.mean(np.abs(img_orig - resampled_image)) < 0.5 * np.mean(
        img_orig
    )


def test_apply_transformation_success():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)
    # Get transformation matrix from rigid registration
    _, trans_matrix = rigid_body_registration(img_orig, img_rot)
    # Apply transformation
    transformed_img = apply_transformation(img_rot, img_orig, trans_matrix)
    assert isinstance(transformed_img, np.ndarray)
    assert transformed_img.shape == img_rot.shape
    assert np.mean(np.abs(transformed_img - img_rot)) < 0.5 * np.mean(img_rot)


def test_apply_transformation_invalid_fixed_image():
    img_rot = load_image(M0_RIGID)
    _, trans_matrix = rigid_body_registration(img_rot, img_rot)
    with pytest.raises(Exception) as e:
        apply_transformation('invalid_image', img_rot, trans_matrix)
    assert 'moving image must be numpy array' in str(e.value)


def test_apply_transformation_invalid_moving_image():
    img_orig = load_image(M0_ORIG)
    _, trans_matrix = rigid_body_registration(img_orig, img_orig)
    with pytest.raises(Exception) as e:
        apply_transformation(img_orig, 'invalid_image', trans_matrix)
    assert 'reference_image must be a numpy array' in str(e.value)


def test_apply_transformation_invalid_transformation_matrix():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)
    with pytest.raises(Exception) as e:
        apply_transformation(img_orig, img_rot, 'invalid_matrix')
    assert 'transforms must be a list of transformation matrices' in str(
        e.value
    )


def test_apply_transformation_with_mask():
    img_orig = load_image(M0_ORIG)
    img_rot = load_image(M0_RIGID)
    mask = np.ones_like(img_orig, dtype=bool)
    _, trans_matrix = rigid_body_registration(img_orig, img_rot)
    transformed_img = apply_transformation(
        img_orig, img_rot, trans_matrix, mask=mask
    )
    assert isinstance(transformed_img, np.ndarray)
    assert transformed_img.shape == img_rot.shape


def test_apply_transformation_with_BrainAtlas_reference_input_error():
    img_rot = load_image(M0_RIGID)
    img_orig = load_image(M0_ORIG)
    _, trans_matrix = rigid_body_registration(img_orig, img_rot)
    with pytest.raises(Exception) as e:
        apply_transformation(img_rot, 'invalid atlas', trans_matrix)

    assert (
        'reference_image must be a numpy array or a BrainAtlas object'
        in str(e.value)
    )


def test_apply_transformation_with_BrainAtlas_reference_input_sucess():
    img_rot = load_image(M0_RIGID)
    img_orig = load_image(M0_ORIG)
    _, trans_matrix = rigid_body_registration(img_orig, img_rot)
    atlas = BrainAtlas(atlas_name='MNI2009')
    atlas_img = load_image(atlas.get_atlas()['t1_data'])
    corr_img = apply_transformation(img_rot, atlas, trans_matrix)

    assert isinstance(corr_img, np.ndarray)
    assert corr_img.shape == atlas_img.shape
