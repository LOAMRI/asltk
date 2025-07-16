import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping, MultiTE_ASLMapping

SEP = os.sep

PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'

asldata_te = ASLData(
    pcasl=PCASL_MTE,
    m0=M0,
    ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    te_values=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.100, 393.36],
)


# Test smoothing functionality
def test_cbf_object_create_map_with_gaussian_smoothing():
    cbf = CBFMapping(asldata_te)
    out_no_smooth = cbf.create_map()
    out_smooth = cbf.create_map(smoothing='gaussian')

    # Check that output has same keys and shapes
    assert set(out_no_smooth.keys()) == set(out_smooth.keys())
    for key in out_no_smooth.keys():
        assert out_no_smooth[key].shape == out_smooth[key].shape

    # Check that smoothing changed the values (reduced noise)
    assert np.std(out_smooth['cbf']) <= np.std(out_no_smooth['cbf'])
    assert np.std(out_smooth['att']) <= np.std(out_no_smooth['att'])


def test_cbf_object_create_map_with_median_smoothing():
    cbf = CBFMapping(asldata_te)
    out_no_smooth = cbf.create_map()
    out_smooth = cbf.create_map(smoothing='median')

    # Check that output has same keys and shapes
    assert set(out_no_smooth.keys()) == set(out_smooth.keys())
    for key in out_no_smooth.keys():
        assert out_no_smooth[key].shape == out_smooth[key].shape

    # Check that smoothing changed the values (reduced noise)
    assert np.std(out_smooth['cbf']) <= np.std(out_no_smooth['cbf'])
    assert np.std(out_smooth['att']) <= np.std(out_no_smooth['att'])


def test_cbf_object_create_map_with_custom_smoothing_params():
    cbf = CBFMapping(asldata_te)
    out_default = cbf.create_map(smoothing='gaussian')
    out_custom = cbf.create_map(
        smoothing='gaussian', smoothing_params={'sigma': 2.0}
    )

    # Check that different parameters produce different results
    assert not np.array_equal(out_default['cbf'], out_custom['cbf'])

    # Custom higher sigma should produce more smoothing
    assert np.std(out_custom['cbf']) <= np.std(out_default['cbf'])


def test_cbf_object_create_map_invalid_smoothing_type():
    cbf = CBFMapping(asldata_te)
    with pytest.raises(ValueError) as e:
        cbf.create_map(smoothing='invalid')
    assert 'Unsupported smoothing type: invalid' in str(e.value)


def test_multite_asl_object_create_map_with_gaussian_smoothing():
    mte = MultiTE_ASLMapping(asldata_te)
    out_no_smooth = mte.create_map()
    out_smooth = mte.create_map(smoothing='gaussian')

    # Check that output has same keys and shapes
    assert set(out_no_smooth.keys()) == set(out_smooth.keys())
    for key in out_no_smooth.keys():
        assert out_no_smooth[key].shape == out_smooth[key].shape

    # Check that smoothing changed the values for t1blgm map
    assert np.std(out_smooth['t1blgm']) <= np.std(out_no_smooth['t1blgm'])


def test_multite_asl_object_create_map_with_median_smoothing():
    mte = MultiTE_ASLMapping(asldata_te)
    out_no_smooth = mte.create_map()
    out_smooth = mte.create_map(
        smoothing='median', smoothing_params={'size': 5}
    )

    # Check that output has same keys and shapes
    assert set(out_no_smooth.keys()) == set(out_smooth.keys())
    for key in out_no_smooth.keys():
        assert out_no_smooth[key].shape == out_smooth[key].shape

    # Check that smoothing changed the values
    assert np.std(out_smooth['t1blgm']) <= np.std(out_no_smooth['t1blgm'])
