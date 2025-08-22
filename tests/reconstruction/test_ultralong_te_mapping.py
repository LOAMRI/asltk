import os
import re
import warnings

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping, UltraLongTE_ASLMapping
from asltk.utils.io import ImageIO

SEP = os.sep

T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'

asldata_te = ASLData(
    pcasl=PCASL_MTE,
    m0=M0,
    ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    te_values=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.100, 393.36],
)

incomplete_asldata = ASLData(pcasl=PCASL_MTE)


def test_ultralongte_asl_object_constructor_created_sucessfully():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    assert isinstance(ulte._asl_data, ASLData)
    assert isinstance(ulte._basic_maps, CBFMapping)
    assert isinstance(ulte._brain_mask, np.ndarray)
    assert isinstance(ulte._cbf_map, np.ndarray)
    assert isinstance(ulte._att_map, np.ndarray)
    assert isinstance(ulte._t1csfgm_map, np.ndarray)
    assert ulte.get_constant('T2csf') == 1500.0
    assert ulte.get_constant('T2bl') == 100


def test_multite_asl_set_brain_mask_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    mask = ImageIO(M0_BRAIN_MASK)
    ulte.set_brain_mask(mask)
    assert isinstance(ulte._brain_mask, np.ndarray)


def test_multite_asl_set_cbf_map_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    fake_cbf = ImageIO(image_array=np.ones((10, 10)) * 20)
    ulte.set_cbf_map(fake_cbf)
    assert np.mean(ulte._cbf_map) == 20


def test_multite_asl_get_cbf_map_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    fake_cbf = ImageIO(image_array=np.ones((10, 10)) * 20)
    ulte.set_cbf_map(fake_cbf)
    assert np.mean(ulte.get_cbf_map()) == 20


def test_multite_asl_set_att_map_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    fake_att = ImageIO(image_array=np.ones((10, 10)) * 20)
    ulte.set_att_map(fake_att)
    assert np.mean(ulte._att_map) == 20


def test_multite_asl_get_att_map_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    fake_att = ImageIO(image_array=np.ones((10, 10)) * 20)
    ulte.set_att_map(fake_att)
    assert np.mean(ulte.get_att_map()) == 20


def test_multite_asl_get_t1csfgm_map_attribution_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    fake_att = np.ones((10, 10)) * 20
    ulte._t1csfgm_map = fake_att
    assert np.mean(ulte.get_t1csfgm_map()) == 20


def test_multite_asl_get_t1csfgm_map_create_map_update_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    out = ulte.create_map()

    assert isinstance(ulte.get_t1csfgm_map(), np.ndarray)
    assert np.mean(ulte.get_t1csfgm_map()) != 0


@pytest.mark.parametrize('label', [(3), (-1), (1000000), (-1.1), (2.1)])
def test_multite_asl_set_brain_mask_set_label_value_raise_error_value_not_found_in_mask(
    label,
):
    ulte = UltraLongTE_ASLMapping(asldata_te)
    mask = ImageIO(M0_BRAIN_MASK)
    with pytest.raises(Exception) as e:
        ulte.set_brain_mask(mask, label=label)
    assert e.value.args[0] == 'Label value is not found in the mask provided.'


def test_multite_asl_set_brain_mask_verify_if_input_is_a_label_mask():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    not_mask = ImageIO(M0)
    with pytest.warns(UserWarning):
        not_mask_image = ImageIO(
            image_array=not_mask.get_as_numpy()
            / np.max(not_mask.get_as_numpy())
        )
        ulte.set_brain_mask(not_mask_image)
        warnings.warn(
            'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
            UserWarning,
        )


def test_multite_asl_set_brain_mask_raise_error_if_image_dimension_is_different_from_3d_volume():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    pcasl_3d_vol = ImageIO(PCASL_MTE).get_as_numpy()[0, 0, :, :, :]
    fake_mask = ImageIO(image_array=np.array(((1, 1, 1), (0, 1, 0))))
    with pytest.raises(Exception) as error:
        ulte.set_brain_mask(fake_mask)
    assert (
        error.value.args[0]
        == f'Image mask dimension does not match with input 3D volume. Mask shape {fake_mask.get_as_numpy().shape} not equal to {pcasl_3d_vol.shape}'
    )


def test_multite_mapping_get_brain_mask_return_adjusted_brain_mask_image_in_the_object():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    assert np.mean(ulte.get_brain_mask()) == 1

    mask = ImageIO(M0_BRAIN_MASK)
    ulte.set_brain_mask(mask)
    assert np.unique(ulte.get_brain_mask()).tolist() == [0, 1]


def test_multite_asl_object_create_map_success():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    out = ulte.create_map()
    assert isinstance(out['cbf'], ImageIO)
    assert np.mean(out['cbf'].get_as_numpy()) < 0.0001
    assert isinstance(out['att'], ImageIO)
    assert np.mean(out['att'].get_as_numpy()) > 10
    assert isinstance(out['t1csfgm'], ImageIO)
    assert np.mean(out['t1csfgm'].get_as_numpy()) > 50


def test_multite_asl_object_raises_error_if_asldata_does_not_have_pcasl_or_m0_image():
    with pytest.raises(Exception) as error:
        ulte = UltraLongTE_ASLMapping(incomplete_asldata)

    assert (
        error.value.args[0]
        == 'ASLData is incomplete. CBFMapping need pcasl and m0 images.'
    )


def test_multite_asl_object_raises_error_if_asldata_does_not_have_te_values():
    incompleted_asldata = ASLData(
        pcasl=PCASL_MTE,
        m0=M0,
        ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
        pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    )
    with pytest.raises(Exception) as error:
        ulte = UltraLongTE_ASLMapping(incompleted_asldata)

    assert (
        error.value.args[0]
        == 'ASLData is incomplete. UltraLongTE_ASLMapping need a list of TE values.'
    )


def test_multite_asl_object_set_cbf_and_att_maps_before_create_map():
    ulte = UltraLongTE_ASLMapping(asldata_te)
    assert np.mean(ulte.get_brain_mask()) == 1

    mask = ImageIO(M0_BRAIN_MASK)
    ulte.set_brain_mask(mask)
    assert np.mean(ulte.get_brain_mask()) < 1

    # Test if CBF/ATT are empty (fresh obj creation)
    assert (
        np.mean(ulte.get_att_map()) == 0 and np.mean(ulte.get_cbf_map()) == 0
    )

    # Update CBF/ATT maps and test if it changed in the obj
    cbf = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 100)
    att = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 1500)
    ulte.set_cbf_map(cbf)
    ulte.set_att_map(att)
    assert (
        np.mean(ulte.get_att_map()) == 1500
        and np.mean(ulte.get_cbf_map()) == 100
    )


def test_multite_asl_object_create_map_using_provided_cbf_att_maps(capfd):
    ulte = UltraLongTE_ASLMapping(asldata_te)
    mask = ImageIO(M0_BRAIN_MASK)
    cbf = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 100)
    att = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 1500)

    ulte.set_brain_mask(mask)
    ulte.set_cbf_map(cbf)
    ulte.set_att_map(att)

    _ = ulte.create_map()
    out, err = capfd.readouterr()
    test_pass = False
    if re.search('ultralongTE-ASL', out):
        test_pass = True
    assert test_pass
