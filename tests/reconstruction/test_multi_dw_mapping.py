import os
import re
import warnings

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction import MultiDW_ASLMapping
from asltk.utils.io import ImageIO

SEP = os.sep

T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MDW = f'tests' + SEP + 'files' + SEP + 'pcasl_mdw.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'

asldata_dw = ASLData(
    pcasl=PCASL_MDW,
    m0=M0,
    ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    dw_values=[0, 50.0, 100.0, 250.0],
)
incomplete_asldata = ASLData(pcasl=PCASL_MDW)


def test_multi_dw_asl_object_constructor_created_sucessfully():
    mte = MultiDW_ASLMapping(asldata_dw)
    assert isinstance(mte._asl_data, ASLData)
    assert isinstance(mte._brain_mask, np.ndarray)
    assert isinstance(mte._cbf_map, np.ndarray)
    assert isinstance(mte._att_map, np.ndarray)
    assert isinstance(mte._A1, np.ndarray)
    assert isinstance(mte._D1, np.ndarray)
    assert isinstance(mte._A2, np.ndarray)
    assert isinstance(mte._D2, np.ndarray)
    assert isinstance(mte._kw, np.ndarray)


def test_multi_dw_asl_set_brain_mask_success():
    mte = MultiDW_ASLMapping(asldata_dw)
    mask = ImageIO(M0_BRAIN_MASK)
    mte.set_brain_mask(mask)
    assert isinstance(mte._brain_mask, np.ndarray)


def test_multi_dw_asl_set_cbf_map_success():
    mte = MultiDW_ASLMapping(asldata_dw)
    fake_cbf = ImageIO(image_array=np.ones((10, 10)) * 20)
    mte.set_cbf_map(fake_cbf)
    assert np.mean(mte._cbf_map) == 20


def test_multi_dw_asl_get_cbf_map_success():
    mte = MultiDW_ASLMapping(asldata_dw)
    fake_cbf = ImageIO(image_array=np.ones((10, 10)) * 20)
    mte.set_cbf_map(fake_cbf)
    assert np.mean(mte.get_cbf_map()) == 20


def test_multi_dw_asl_set_att_map_success():
    mte = MultiDW_ASLMapping(asldata_dw)
    fake_att = ImageIO(image_array=np.ones((10, 10)) * 20)
    mte.set_att_map(fake_att)
    assert np.mean(mte._att_map) == 20


def test_multi_dw_asl_get_att_map_success():
    mte = MultiDW_ASLMapping(asldata_dw)
    fake_att = ImageIO(image_array=np.ones((10, 10)) * 20)
    mte.set_att_map(fake_att)
    assert np.mean(mte.get_att_map()) == 20


@pytest.mark.parametrize('label', [(3), (-1), (1000000), (-1.1), (2.1)])
def test_multi_dw_asl_set_brain_mask_set_label_value_raise_error_value_not_found_in_mask(
    label,
):
    mte = MultiDW_ASLMapping(asldata_dw)
    mask = ImageIO(M0_BRAIN_MASK)
    with pytest.raises(Exception) as e:
        mte.set_brain_mask(mask, label=label)
    assert e.value.args[0] == 'Label value is not found in the mask provided.'


def test_multi_dw_asl_set_brain_mask_verify_if_input_is_a_label_mask():
    mte = MultiDW_ASLMapping(asldata_dw)
    not_mask = ImageIO(M0)
    with pytest.warns(UserWarning):
        mte.set_brain_mask(
            ImageIO(
                image_array=not_mask.get_as_numpy()
                / np.max(not_mask.get_as_numpy())
            )
        )
        warnings.warn(
            'Mask image is not a binary image. Any value > 0 will be assumed as brain label.',
            UserWarning,
        )


def test_multi_dw_asl_set_brain_mask_raise_error_if_image_dimension_is_different_from_3d_volume():
    mte = MultiDW_ASLMapping(asldata_dw)
    pcasl_3d_vol = ImageIO(
        image_array=ImageIO(PCASL_MDW).get_as_numpy()[0, 0, :, :, :]
    )
    fake_mask = ImageIO(image_array=np.array(((1, 1, 1), (0, 1, 0))))
    with pytest.raises(Exception) as error:
        mte.set_brain_mask(fake_mask)
    assert (
        error.value.args[0]
        == f'Image mask dimension does not match with input 3D volume. Mask shape {fake_mask.get_as_numpy().shape} not equal to {pcasl_3d_vol.get_as_numpy().shape}'
    )


def test_multi_dw_mapping_get_brain_mask_return_adjusted_brain_mask_image_in_the_object():
    mdw = MultiDW_ASLMapping(asldata_dw)
    assert np.mean(mdw.get_brain_mask()) == 1

    mask = ImageIO(M0_BRAIN_MASK)
    mdw.set_brain_mask(mask)
    assert np.unique(mdw.get_brain_mask()).tolist() == [0, 1]


# def test_multi_dw_asl_object_create_map_success():
#     mte = MultiDW_ASLMapping(asldata_dw)
#     out = mte.create_map()
#     assert isinstance(out['cbf'], np.ndarray)
#     assert np.mean(out['cbf']) < 0.0001
#     assert isinstance(out['att'], np.ndarray)
#     assert np.mean(out['att']) > 10
#     assert isinstance(out['t1blgm'], np.ndarray)
#     assert np.mean(out['t1blgm']) > 50


def test_multi_dw_asl_object_raises_error_if_asldata_does_not_have_pcasl_or_m0_image():
    with pytest.raises(Exception) as error:
        mte = MultiDW_ASLMapping(incomplete_asldata)

    assert (
        error.value.args[0]
        == 'ASLData is incomplete. CBFMapping need pcasl and m0 images.'
    )


def test_multi_dw_asl_object_raises_error_if_asldata_does_not_have_te_values():
    incompleted_asldata = ASLData(
        pcasl=PCASL_MDW,
        m0=M0,
        ld_values=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
        pld_values=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    )
    with pytest.raises(Exception) as error:
        mte = MultiDW_ASLMapping(incompleted_asldata)

    assert (
        error.value.args[0]
        == 'ASLData is incomplete. MultiDW_ASLMapping need a list of DW values.'
    )


def test_multi_dw_asl_object_set_cbf_and_att_maps_before_create_map():
    mte = MultiDW_ASLMapping(asldata_dw)
    assert np.mean(mte.get_brain_mask()) == 1

    mask = ImageIO(M0_BRAIN_MASK)
    mte.set_brain_mask(mask)
    assert np.mean(mte.get_brain_mask()) < 1

    # Test if CBF/ATT are empty (fresh obj creation)
    assert np.mean(mte.get_att_map()) == 0 and np.mean(mte.get_cbf_map()) == 0

    # Update CBF/ATT maps and test if it changed in the obj
    cbf = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 100)
    att = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 1500)
    mte.set_cbf_map(cbf)
    mte.set_att_map(att)
    assert (
        np.mean(mte.get_att_map()) == 1500
        and np.mean(mte.get_cbf_map()) == 100
    )


def test_multi_dw_asl_object_create_map_using_provided_cbf_att_maps(capfd):
    mte = MultiDW_ASLMapping(asldata_dw)
    mask = ImageIO(M0_BRAIN_MASK)
    cbf = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 100)
    att = ImageIO(image_array=np.ones(mask.get_as_numpy().shape) * 1500)

    mte.set_brain_mask(mask)
    mte.set_cbf_map(cbf)
    mte.set_att_map(att)

    _ = mte.create_map()
    out, err = capfd.readouterr()
    test_pass = False
    if re.search('multiDW-ASL', out):
        test_pass = True
    assert test_pass
