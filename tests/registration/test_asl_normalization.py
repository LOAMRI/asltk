import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.registration.asl_normalization import head_movement_correction
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
    assert (
        np.abs(
            np.mean(np.subtract(pcasl_corrected('pcasl'), pcasl_orig('pcasl')))
        )
        > np.mean(pcasl_orig('pcasl')) * 0.1
    )
    assert any(not np.array_equal(mtx, np.eye(4)) for mtx in trans_mtxs)


def test_head_movement_correction_returns_asl_data_corrected():
    pcasl_orig = ASLData(pcasl=PCASL_MTE, m0=M0)

    asl_data_corrected, _ = head_movement_correction(pcasl_orig)

    assert isinstance(asl_data_corrected, ASLData)
    assert asl_data_corrected('pcasl').shape == pcasl_orig('pcasl').shape
    assert asl_data_corrected('pcasl').dtype == pcasl_orig('pcasl').dtype
