import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.registration.asl_normalization import (
    asl_template_registration,
    head_movement_correction,
)

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
    # assert (
    #     np.abs(
    #         np.mean(np.subtract(pcasl_corrected('pcasl'), pcasl_orig('pcasl')))
    #     )
    #     > np.abs(np.mean(pcasl_orig('pcasl')) * 0.01)
    # )
    assert any(not np.array_equal(mtx, np.eye(4)) for mtx in trans_mtxs)


def test_head_movement_correction_returns_asl_data_corrected():
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)

    asl_data_corrected, _ = head_movement_correction(pcasl_orig)

    assert isinstance(asl_data_corrected, ASLData)
    assert asl_data_corrected('pcasl').shape == pcasl_orig('pcasl').shape
    assert asl_data_corrected('pcasl').dtype == pcasl_orig('pcasl').dtype


# TODO Arrumar o path do arquivo de template
# def test_asl_template_registration_success():
#     pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)
#     # pcasl_orig = ASLData(
#     #     pcasl='/home/antonio/Imagens/loamri-samples/20240909/pcasl.nii.gz',
#     #     m0='/home/antonio/Imagens/loamri-samples/20240909/m0.nii.gz',
#     # )
#     # asl_data_mask = np.ones_like(pcasl_orig('m0'), dtype=bool)

#     asl_data_registered, trans_mtxs = asl_template_registration(
#         pcasl_orig,
#         atlas_name='MNI2009',
#         verbose=True,
#     )

#     assert isinstance(asl_data_registered, ASLData)
#     assert asl_data_registered('pcasl').shape == pcasl_orig('pcasl').shape
#     assert isinstance(trans_mtxs, list)
#     assert len(trans_mtxs) == pcasl_orig('pcasl').shape[0]


def test_asl_template_registration_invalid_input_type():
    with pytest.raises(TypeError) as e:
        asl_template_registration('not_asldata')
    assert str(e.value) == 'Input must be an ASLData object.'


# def test_asl_template_registration_invalid_ref_vol_type():
#     pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)
#     with pytest.raises(ValueError) as e:
#         asl_template_registration(pcasl_orig, ref_vol='invalid')
#     assert str(e.value) == 'ref_vol must be a non-negative integer.'


# def test_asl_template_registration_invalid_ref_vol_type_with_negative_volume():
#     pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)
#     with pytest.raises(ValueError) as e:
#         asl_template_registration(pcasl_orig, ref_vol=-1)
#     assert str(e.value) == 'ref_vol must be a non-negative integer.'


# def test_asl_template_registration_invalid_ref_vol_index():
#     pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)
#     n_vols = 1000000
#     with pytest.raises(ValueError) as e:
#         asl_template_registration(pcasl_orig, ref_vol=n_vols)
#     assert 'ref_vol must be a valid index' in str(e.value)


# def test_asl_template_registration_create_another_asldata_object():
#     pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)

#     asl_data_registered, _ = asl_template_registration(
#         pcasl_orig,
#         ref_vol=0,
#         atlas_name='MNI2009',
#         verbose=True,
#     )

#     assert isinstance(asl_data_registered, ASLData)
#     assert asl_data_registered('pcasl').shape == pcasl_orig('pcasl').shape
#     assert asl_data_registered('m0').shape == pcasl_orig('m0').shape
#     assert asl_data_registered is not pcasl_orig


# def test_asl_template_registration_returns_transforms():
#     pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)
#     asl_data_mask = np.ones_like(pcasl_orig('pcasl')[0], dtype=bool)

#     asl_data_registered, trans_mtxs = asl_template_registration(
#         pcasl_orig, ref_vol=0, asl_data_mask=asl_data_mask
#     )

#     assert isinstance(trans_mtxs, list)
#     assert all(isinstance(mtx, np.ndarray) for mtx in trans_mtxs)
