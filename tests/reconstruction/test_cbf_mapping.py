import os
import warnings

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping
from asltk.utils import load_image

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
asldata_dw = ASLData(
    pcasl=PCASL_MDW,
    m0=M0,
    ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    dw_values=[0, 50.0, 100.0, 250.0],
)
incomplete_asldata = ASLData(pcasl=PCASL_MTE)


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
    cbf = CBFMapping(asldata_te)
    mri_default = cbf.get_constant(param)
    cbf.set_constant(value, param)
    assert cbf.get_constant(param) != mri_default


def test_cbf_add_brain_mask_success():
    cbf = CBFMapping(asldata_te)
    mask = load_image(M0_BRAIN_MASK)
    cbf.set_brain_mask(mask)
    assert isinstance(cbf._brain_mask, np.ndarray)


def test_cbf_object_create_map_raise_error_if_ld_or_pld_are_not_provided():
    data = ASLData(pcasl=PCASL_MTE, m0=M0)
    cbf = CBFMapping(data)
    with pytest.raises(Exception) as e:
        cbf.create_map()
    assert e.value.args[0] == 'LD or PLD list of values must be provided.'


def test_set_brain_mask_verify_if_input_is_a_label_mask():
    cbf = CBFMapping(asldata_te)
    not_mask = load_image(T1_MRI)
    with pytest.warns(UserWarning):
        warnings.warn(
            'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
            UserWarning,
        )


def test_set_brain_mask_set_label_value():
    cbf = CBFMapping(asldata_te)
    mask = load_image(M0_BRAIN_MASK)
    cbf.set_brain_mask(mask, label=1)
    assert np.unique(cbf._brain_mask).size == 2
    assert np.max(cbf._brain_mask) == np.int8(1)


@pytest.mark.parametrize('label', [(3), (-1), (1000000), (-1.1), (2.1)])
def test_set_brain_mask_set_label_value_raise_error_value_not_found_in_mask(
    label,
):
    cbf = CBFMapping(asldata_te)
    mask = load_image(M0_BRAIN_MASK)
    with pytest.raises(Exception) as e:
        cbf.set_brain_mask(mask, label=label)
    assert e.value.args[0] == 'Label value is not found in the mask provided.'


def test_set_brain_mask_gives_binary_image_using_correct_label_value():
    cbf = CBFMapping(asldata_te)
    img = np.zeros((5, 35, 35))
    img[1, 16:30, 16:30] = 250
    img[1, 0:15, 0:15] = 1
    cbf.set_brain_mask(img, label=250)
    assert np.unique(cbf._brain_mask).size == 2
    assert np.max(cbf._brain_mask) == np.uint8(250)
    assert np.min(cbf._brain_mask) == np.uint8(0)


# def test_ TODO Teste se mask tem mesma dimensao que 3D asl
def test_set_brain_mask_raise_error_if_image_dimension_is_different_from_3d_volume():
    cbf = CBFMapping(asldata_te)
    pcasl_3d_vol = load_image(PCASL_MTE)[0, 0, :, :, :]
    fake_mask = np.array(((1, 1, 1), (0, 1, 0)))
    with pytest.raises(Exception) as error:
        cbf.set_brain_mask(fake_mask)
    assert (
        error.value.args[0]
        == f'Image mask dimension does not match with input 3D volume. Mask shape {fake_mask.shape} not equal to {pcasl_3d_vol.shape}'
    )


def test_set_brain_mask_creates_3d_volume_of_ones_if_not_set_in_cbf_object():
    cbf = CBFMapping(asldata_te)
    vol_shape = asldata_te('m0').shape
    mask_shape = cbf._brain_mask.shape
    assert vol_shape == mask_shape


def test_set_brain_mask_raise_error_mask_is_not_an_numpy_array():
    cbf = CBFMapping(asldata_te)
    with pytest.raises(Exception) as e:
        cbf.set_brain_mask(M0_BRAIN_MASK)
    assert (
        e.value.args[0]
        == f'mask is not an numpy array. Type {type(M0_BRAIN_MASK)}'
    )


def test_cbf_mapping_get_brain_mask_return_adjusted_brain_mask_image_in_the_object():
    cbf = CBFMapping(asldata_te)
    assert np.mean(cbf.get_brain_mask()) == 1

    mask = load_image(M0_BRAIN_MASK)
    cbf.set_brain_mask(mask)
    assert np.unique(cbf.get_brain_mask()).tolist() == [0, 1]


def test_cbf_object_create_map_success():
    cbf = CBFMapping(asldata_te)
    out = cbf.create_map()
    assert isinstance(out['cbf'], np.ndarray)
    assert np.mean(out['cbf']) < 0.0001
    assert isinstance(out['att'], np.ndarray)
    assert np.mean(out['att']) > 10


def test_cbf_object_create_map_sucess_setting_single_core():
    cbf = CBFMapping(asldata_te)
    out = cbf.create_map(cores=1)
    assert isinstance(out['cbf'], np.ndarray)
    assert np.mean(out['cbf']) < 0.0001
    assert isinstance(out['att'], np.ndarray)
    assert np.mean(out['att']) > 10


@pytest.mark.parametrize('core_value', [(100), (-1), (-10), (1.5), (-1.5)])
def test_cbf_raise_error_cores_not_valid(core_value):
    cbf = CBFMapping(asldata_te)
    with pytest.raises(Exception) as e:
        cbf.create_map(cores=core_value)

    assert (
        e.value.args[0]
        == 'Number of proecess must be at least 1 and less than maximum cores availble.'
    )


def test_cbf_map_normalized_flag_true_result_cbf_map_rescaled():
    cbf = CBFMapping(asldata_te)
    out = cbf.create_map()
    out['cbf_norm'][out['cbf_norm'] == 0] = np.nan
    mean_px_value = np.nanmean(out['cbf_norm'])
    assert mean_px_value < 500 and mean_px_value > 50
