import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.registration import rigid_body_registration
from asltk.registration.asl_normalization import head_movement_correction
from asltk.utils.io import load_image

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

    assert isinstance(trans_matrix, list)


def test_head_movement_correction_build_asldata_success():
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)

    asldata, _ = head_movement_correction(pcasl_orig)

    assert asldata('pcasl').shape == pcasl_orig('pcasl').shape


def test_head_movement_correction_error_input_is_not_ASLData_object():
    with pytest.raises(TypeError) as e:
        head_movement_correction('invalid_input')

    assert str(e.value) == 'Input must be an ASLData object.'


def test_head_movement_correction_error_ref_vol_is_not_int():
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)

    with pytest.raises(Exception) as e:
        head_movement_correction(pcasl_orig, ref_vol='invalid_ref_vol')

    assert (
        str(e.value)
        == 'ref_vol must be an positive integer based on the total asl data volumes.'
    )


def test_head_movement_correction_success():
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)

    pcasl_corrected, trans_mtxs = head_movement_correction(
        pcasl_orig, verbose=True
    )

    assert pcasl_corrected('pcasl').shape == pcasl_orig('pcasl').shape
    assert any(not np.array_equal(mtx, np.eye(4)) for mtx in trans_mtxs)
