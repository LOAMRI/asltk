import os
import re
import warnings

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping, MultiTE_ASLMapping
from asltk.utils import load_image

SEP = os.sep

T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL = f'tests' + SEP + 'files' + SEP + 'pcasl.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'

asldata = ASLData(
    pcasl=PCASL,
    m0=M0,
    ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    te_values=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.100, 393.36],
)
incomplete_asldata = ASLData(pcasl=PCASL)


def test_cbf_object_raises_error_if_asldata_does_not_have_pcasl_or_m0_image():
    with pytest.raises(Exception) as error:
        cbf = CBFMapping(incomplete_asldata)

    assert (
        error.value.args[0]
        == 'ASLData is incomplete. CBFMapping need pcasl and m0 images.'
    )


@pytest.mark.parametrize(
    'value,param',
    [
        (100, 'T1bl'),
        (151, 'T1csf'),
        (200.2, 'T2bl'),
        (110, 'T2gm'),
        (5600, 'T2csf'),
        (0.99, 'Alpha'),
        (0.69, 'Lambda'),
    ],
)
def test_cbf_object_set_mri_parameters_values(value, param):
    cbf = CBFMapping(asldata)
    mri_default = cbf.get_constant(param)
    cbf.set_constant(value, param)
    assert cbf.get_constant(param) != mri_default


def test_cbf_add_brain_mask_success():
    cbf = CBFMapping(asldata)
    mask = load_image(M0_BRAIN_MASK)
    cbf.set_brain_mask(mask)
    assert isinstance(cbf._brain_mask, np.ndarray)


def test_cbf_object_create_map_raise_error_if_ld_or_pld_are_not_provided():
    data = ASLData(pcasl=PCASL, m0=M0)
    cbf = CBFMapping(data)
    with pytest.raises(Exception) as e:
        cbf.create_map()
    assert e.value.args[0] == 'LD or PLD list of values must be provided.'


def test_set_brain_mask_verify_if_input_is_a_label_mask():
    cbf = CBFMapping(asldata)
    not_mask = load_image(T1_MRI)
    with pytest.warns(UserWarning):
        warnings.warn(
            'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
            UserWarning,
        )


def test_set_brain_mask_set_label_value():
    cbf = CBFMapping(asldata)
    mask = load_image(M0_BRAIN_MASK)
    cbf.set_brain_mask(mask, label=1)
    assert np.unique(cbf._brain_mask).size == 2
    assert np.max(cbf._brain_mask) == np.int8(1)


@pytest.mark.parametrize('label', [(3), (-1), (1000000), (-1.1), (2.1)])
def test_set_brain_mask_set_label_value_raise_error_value_not_found_in_mask(
    label,
):
    cbf = CBFMapping(asldata)
    mask = load_image(M0_BRAIN_MASK)
    with pytest.raises(Exception) as e:
        cbf.set_brain_mask(mask, label=label)
    assert e.value.args[0] == 'Label value is not found in the mask provided.'


def test_set_brain_mask_gives_binary_image_using_correct_label_value():
    cbf = CBFMapping(asldata)
    img = np.zeros((5, 35, 35))
    img[1, 16:30, 16:30] = 250
    img[1, 0:15, 0:15] = 1
    cbf.set_brain_mask(img, label=250)
    assert np.unique(cbf._brain_mask).size == 2
    assert np.max(cbf._brain_mask) == np.uint8(250)
    assert np.min(cbf._brain_mask) == np.uint8(0)


# def test_ TODO Teste se mask tem mesma dimensao que 3D asl
def test_set_brain_mask_raise_error_if_image_dimension_is_different_from_3d_volume():
    cbf = CBFMapping(asldata)
    pcasl_3d_vol = load_image(PCASL)[0, 0, :, :, :]
    fake_mask = np.array(((1, 1, 1), (0, 1, 0)))
    with pytest.raises(Exception) as error:
        cbf.set_brain_mask(fake_mask)
    assert (
        error.value.args[0]
        == f'Image mask dimension does not match with input 3D volume. Mask shape {fake_mask.shape} not equal to {pcasl_3d_vol.shape}'
    )


def test_set_brain_mask_creates_3d_volume_of_ones_if_not_set_in_cbf_object():
    cbf = CBFMapping(asldata)
    vol_shape = asldata('m0').shape
    mask_shape = cbf._brain_mask.shape
    assert vol_shape == mask_shape


def test_set_brain_mask_raise_error_mask_is_not_an_numpy_array():
    cbf = CBFMapping(asldata)
    with pytest.raises(Exception) as e:
        cbf.set_brain_mask(M0_BRAIN_MASK)
    assert (
        e.value.args[0]
        == f'mask is not an numpy array. Type {type(M0_BRAIN_MASK)}'
    )


def test_cbf_object_create_map_success():
    cbf = CBFMapping(asldata)
    out = cbf.create_map()
    assert isinstance(out['cbf'], np.ndarray)
    assert np.mean(out['cbf']) < 0.0001
    assert isinstance(out['att'], np.ndarray)
    assert np.mean(out['att']) > 10


def test_cbf_map_normalized_flag_true_result_cbf_map_rescaled():
    cbf = CBFMapping(asldata)
    out = cbf.create_map()
    out['cbf_norm'][out['cbf_norm'] == 0] = np.nan
    mean_px_value = np.nanmean(out['cbf_norm'])
    assert mean_px_value < 500 and mean_px_value > 50


def test_multite_asl_object_constructor_created_sucessfully():
    mte = MultiTE_ASLMapping(asldata)
    assert isinstance(mte._asl_data, ASLData)
    assert isinstance(mte._basic_maps, CBFMapping)
    assert isinstance(mte._brain_mask, np.ndarray)
    assert isinstance(mte._cbf_map, np.ndarray)
    assert isinstance(mte._att_map, np.ndarray)
    assert isinstance(mte._t1blgm_map, np.ndarray)


def test_multite_asl_set_brain_mask_success():
    mte = MultiTE_ASLMapping(asldata)
    mask = load_image(M0_BRAIN_MASK)
    mte.set_brain_mask(mask)
    assert isinstance(mte._brain_mask, np.ndarray)


def test_multite_asl_set_cbf_map_success():
    mte = MultiTE_ASLMapping(asldata)
    fake_cbf = np.ones((10, 10)) * 20
    mte.set_cbf_map(fake_cbf)
    assert np.mean(mte._cbf_map) == 20


def test_multite_asl_get_cbf_map_success():
    mte = MultiTE_ASLMapping(asldata)
    fake_cbf = np.ones((10, 10)) * 20
    mte.set_cbf_map(fake_cbf)
    assert np.mean(mte.get_cbf_map()) == 20


def test_multite_asl_set_att_map_success():
    mte = MultiTE_ASLMapping(asldata)
    fake_att = np.ones((10, 10)) * 20
    mte.set_att_map(fake_att)
    assert np.mean(mte._att_map) == 20


def test_multite_asl_get_att_map_success():
    mte = MultiTE_ASLMapping(asldata)
    fake_att = np.ones((10, 10)) * 20
    mte.set_att_map(fake_att)
    assert np.mean(mte.get_att_map()) == 20


@pytest.mark.parametrize('label', [(3), (-1), (1000000), (-1.1), (2.1)])
def test_multite_asl_set_brain_mask_set_label_value_raise_error_value_not_found_in_mask(
    label,
):
    mte = MultiTE_ASLMapping(asldata)
    mask = load_image(M0_BRAIN_MASK)
    with pytest.raises(Exception) as e:
        mte.set_brain_mask(mask, label=label)
    assert e.value.args[0] == 'Label value is not found in the mask provided.'


def test_multite_asl_set_brain_mask_verify_if_input_is_a_label_mask():
    mte = MultiTE_ASLMapping(asldata)
    not_mask = load_image(M0)
    with pytest.warns(UserWarning):
        mte.set_brain_mask(not_mask / np.max(not_mask))
        warnings.warn(
            'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
            UserWarning,
        )


def test_multite_asl_set_brain_mask_raise_error_if_image_dimension_is_different_from_3d_volume():
    mte = MultiTE_ASLMapping(asldata)
    pcasl_3d_vol = load_image(PCASL)[0, 0, :, :, :]
    fake_mask = np.array(((1, 1, 1), (0, 1, 0)))
    with pytest.raises(Exception) as error:
        mte.set_brain_mask(fake_mask)
    assert (
        error.value.args[0]
        == f'Image mask dimension does not match with input 3D volume. Mask shape {fake_mask.shape} not equal to {pcasl_3d_vol.shape}'
    )


def test_multite_asl_object_create_map_success():
    mte = MultiTE_ASLMapping(asldata)
    out = mte.create_map()
    assert isinstance(out['cbf'], np.ndarray)
    assert np.mean(out['cbf']) < 0.0001
    assert isinstance(out['att'], np.ndarray)
    assert np.mean(out['att']) > 10
    assert isinstance(out['t1blgm'], np.ndarray)
    assert np.mean(out['t1blgm']) > 50


def test_multite_asl_object_raises_error_if_asldata_does_not_have_pcasl_or_m0_image():
    with pytest.raises(Exception) as error:
        mte = MultiTE_ASLMapping(incomplete_asldata)

    assert (
        error.value.args[0]
        == 'ASLData is incomplete. CBFMapping need pcasl and m0 images.'
    )


def test_multite_asl_object_raises_error_if_asldata_does_not_have_te_values():
    incompleted_asldata = ASLData(
        pcasl=PCASL,
        m0=M0,
        ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
        pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    )
    with pytest.raises(Exception) as error:
        mte = MultiTE_ASLMapping(incompleted_asldata)

    assert (
        error.value.args[0]
        == 'ASLData is incomplete. MultiTE_ASLMapping need a list of TE values.'
    )


def test_multite_asl_object_set_cbf_and_att_maps_before_create_map():
    mte = MultiTE_ASLMapping(asldata)
    assert np.mean(mte.get_brain_mask()) == 1

    mask = load_image(M0_BRAIN_MASK)
    mte.set_brain_mask(mask)
    assert np.mean(mte.get_brain_mask()) < 1

    # Test if CBF/ATT are empty (fresh obj creation)
    assert np.mean(mte.get_att_map()) == 0 and np.mean(mte.get_cbf_map()) == 0

    # Update CBF/ATT maps and test if it changed in the obj
    cbf = np.ones(mask.shape) * 100
    att = np.ones(mask.shape) * 1500
    mte.set_cbf_map(cbf)
    mte.set_att_map(att)
    assert (
        np.mean(mte.get_att_map()) == 1500
        and np.mean(mte.get_cbf_map()) == 100
    )


def test_multite_asl_object_create_map_using_provided_cbf_att_maps(capfd):
    mte = MultiTE_ASLMapping(asldata)
    mask = load_image(M0_BRAIN_MASK)
    cbf = np.ones(mask.shape) * 100
    att = np.ones(mask.shape) * 1500

    mte.set_brain_mask(mask)
    mte.set_cbf_map(cbf)
    mte.set_att_map(att)

    _ = mte.create_map()
    out, err = capfd.readouterr()
    test_pass = False
    if re.search('multiTE-ASL', out):
        test_pass = True
    assert test_pass
