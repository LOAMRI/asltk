import os
import warnings

import numpy as np
import pytest

from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping
from asltk.utils import load_image

SEP = os.sep

T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL = f'tests' + SEP + 'files' + SEP + 'pcasl.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'

asldata = ASLData(pcasl=T1_MRI, m0=T1_MRI)


def test_create_cbfmapping_object_success():
    cbf = CBFMapping(asldata)
    assert isinstance(cbf, CBFMapping)


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
    img = load_image(T1_MRI)
    cbf.set_brain_mask(img, label=250)
    assert np.unique(cbf._brain_mask).size == 2
    assert np.max(cbf._brain_mask) == np.uint8(250)
    assert np.min(cbf._brain_mask) == np.uint8(0)


# def test_ TODO Teste se mask tem mesma dimensao que 3D asl
