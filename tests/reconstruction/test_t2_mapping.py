import os

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction.t2_mapping import T2Scalar_ASLMapping

SEP = os.sep

T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
PCASL_MDW = f'tests' + SEP + 'files' + SEP + 'pcasl_mdw.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'

asldata_te = ASLData(
    pcasl=PCASL_MTE,
    m0=M0,
    ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    te_values=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.100, 393.36],
)


def test_t2_scalar_asl_mapping_initialization():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)

    assert isinstance(t2_mapping, T2Scalar_ASLMapping)
    assert isinstance(t2_mapping._asl_data, ASLData)
    assert isinstance(t2_mapping._brain_mask, np.ndarray)
    assert t2_mapping._t2_maps is None
    assert t2_mapping._mean_t2s is None


def test_t2_scalar_mapping_raise_error_if_asl_data_do_not_has_te_values():
    asldata = ASLData(pcasl=PCASL_MTE, m0=M0)
    with pytest.raises(ValueError) as error:
        T2Scalar_ASLMapping(asldata)
    assert str(error.value) == 'ASLData must provide TE and PLD values.'


def test_t2_scalar_mapping_raise_error_if_asl_data_do_not_has_pld_values():
    asldata = ASLData(pcasl=PCASL_MTE, m0=M0, te_values=asldata_te.get_te())
    with pytest.raises(ValueError) as error:
        T2Scalar_ASLMapping(asldata)
    assert str(error.value) == 'ASLData must provide TE and PLD values.'


def test_t2_scalar_mapping_success_construction_t2_map():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)

    out = t2_mapping.create_map()

    assert isinstance(out['t2'], np.ndarray)
    assert out['t2'].ndim == 4  # Expecting a 4D array
    assert out['mean_t2'] is not None
    assert len(out['mean_t2']) == len(
        asldata_te.get_pld()
    )  # One mean T2 per PLD
    assert np.mean(out['t2']) > 0  # Ensure T2 values are positive


def test_t2_scalar_mapping_raise_error_with_dw_in_asldata():
    asldata = ASLData(
        pcasl=PCASL_MTE,
        m0=M0,
        ld_values=asldata_te.get_ld(),
        pld_values=asldata_te.get_pld(),
        te_values=asldata_te.get_te(),
        dw_values=[1000, 2000, 3000],
    )
    with pytest.raises(ValueError) as error:
        T2Scalar_ASLMapping(asldata)
    assert str(error.value) == 'ASLData must not include DW values.'


def test_t2_scalar_mapping_get_t2_maps_and_mean_t2s_before_and_after_create_map():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)

    # Before map creation, should be None
    assert t2_mapping.get_t2_maps() is None
    assert t2_mapping.get_mean_t2s() is None

    # After map creation, should return correct types and shapes
    t2_mapping.create_map()
    t2_maps = t2_mapping.get_t2_maps()
    mean_t2s = t2_mapping.get_mean_t2s()

    assert isinstance(t2_maps, np.ndarray)
    assert t2_maps.ndim == 4  # (N_PLDS, Z, Y, X)
    assert isinstance(mean_t2s, list)
    assert len(mean_t2s) == len(asldata_te.get_pld())
    assert all(
        isinstance(val, float) or isinstance(val, np.floating)
        for val in mean_t2s
    )
    assert np.all(t2_maps >= 0)


def test_set_brain_mask_binary_and_label():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)
    shape = t2_mapping._asl_data('m0').shape

    # Binary mask (all ones)
    binary_mask = np.ones(shape, dtype=np.uint8)
    t2_mapping.set_brain_mask(binary_mask)
    assert np.all(t2_mapping._brain_mask == 1)
    assert t2_mapping._brain_mask.shape == shape

    # Mask with different label
    label = 2
    mask_with_label = np.zeros(shape, dtype=np.uint8)
    mask_with_label[0, 0, 0] = label
    t2_mapping.set_brain_mask(mask_with_label, label=label)
    assert t2_mapping._brain_mask[0, 0, 0] == label
    assert np.sum(t2_mapping._brain_mask == label) == 1


def test_set_brain_mask_invalid_shape_raises():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)
    wrong_shape_mask = np.ones((2, 2, 2), dtype=np.uint8)
    with pytest.raises(Exception) as error:
        t2_mapping.set_brain_mask(wrong_shape_mask)

    assert 'Image mask dimension does not match with input 3D volume.' in str(
        error.value
    )


def test_set_brain_mask_noninteger_label():
    t2_mapping = T2Scalar_ASLMapping(asldata_te)
    shape = t2_mapping._asl_data('m0').shape
    mask = np.ones(shape, dtype=np.float32)
    # Should still work, as mask == label will be True for 1.0 == 1
    t2_mapping.set_brain_mask(mask, label=1)
    assert np.all(t2_mapping._brain_mask == 1)
