import os

import numpy as np
import pytest
import SimpleITK as sitk

from asltk import asldata, utils

SEP = os.sep
T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL = f'tests' + SEP + 'files' + SEP + 'pcasl.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'


def test_load_image_pcasl_type_update_object_image_reference():
    img = utils.load_image(PCASL)
    assert isinstance(img, np.ndarray)


def test_load_image_m0_type_update_object_image_reference():
    img = utils.load_image(M0)
    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize(
    'input',
    [
        ('/wrong/path'),
        ('not-a-path'),
        (f'tests' + SEP + 'files' + SEP + 'no-image.nrrd'),
    ],
)
def test_load_image_attest_fullpath_is_valid(input):
    with pytest.raises(Exception) as e:
        utils.load_image(input)
    assert e.value.args[0] == 'Image path is not valid or image not found.'


@pytest.mark.parametrize(
    'input', [('out.nrrd'), ('out.nii'), ('out.mha'), ('out.tif')]
)
def test_save_image_success(input, tmp_path):
    img = utils.load_image(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    utils.save_image(img, full_path)
    assert os.path.exists(full_path)
    read_file = sitk.ReadImage(full_path)
    assert read_file.GetSize() == sitk.ReadImage(T1_MRI).GetSize()


@pytest.mark.parametrize(
    'input', [('out.nrr'), ('out.n'), ('out.m'), ('out.zip')]
)
def test_save_image_throw_error_invalid_formatt(input, tmp_path):
    img = utils.load_image(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    with pytest.raises(Exception) as e:
        utils.save_image(img, full_path)


def test_asl_model_buxton_return_sucess_list_of_values():
    buxton_values = utils.asl_model_buxton(
        tau=[1, 2, 3], w=[10, 20, 30], m0=1000, cbf=450, att=1500
    )
    assert len(buxton_values.tolist()) == 3
    assert type(buxton_values) == np.ndarray


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 2]), ([100.1, 200.0, 'text'])]
)
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'tau list must contain float or int values'


@pytest.mark.parametrize('input', [('a'), (2), (100.1)])
def test_asl_model_buxton_tau_raise_errors_with_wrong_inputs(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=input, w=[10, 20, 30], m0=1000, cbf=450, att=1500
        )
    assert (
        e.value.args[0] == 'tau parameter must be a list or tuple of values.'
    )


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 2]), ([100.1, 200.0, np.ndarray])]
)
def test_asl_model_buxton_w_raise_errors_with_wrong_inputs(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=[10, 20, 30], w=input, m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'w list must contain float or int values'


@pytest.mark.parametrize('input', [('a'), (1), (100.1), (np.ndarray)])
def test_asl_model_buxton_w_raise_errors_with_wrong_inputs_not_list(input):
    with pytest.raises(Exception) as e:
        buxton_values = utils.asl_model_buxton(
            tau=[10, 20, 30], w=input, m0=1000, cbf=450, att=1500
        )
    assert e.value.args[0] == 'w parameter must be a list or tuple of values.'


def test_asl_model_multi_te_return_sucess_list_of_values():
    multite_values = utils.asl_model_multi_te(
        tau=[1, 2, 3], w=[10, 20, 30], te=[1, 2, 3], m0=1000, cbf=450, att=1500
    )
    assert len(multite_values) == 3
    assert type(multite_values) == np.ndarray
