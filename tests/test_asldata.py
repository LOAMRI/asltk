import os

import numpy as np
import pytest
import SimpleITK as sitk

from asltk import asldata

SEP = os.sep
T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
TMP_OUTPUT_FOLDER = 'tests' + SEP + 'tmp' + SEP


def test_create_successfuly_asldata_object():
    obj = asldata.asldata()
    assert isinstance(obj, asldata.asldata)


def test_create_object_check_initial_parameters():
    obj = asldata.asldata()
    assert obj.parameters['ld'] == []
    assert obj.parameters['pld'] == []


def test_get_ld_show_empty_list_for_new_object():
    obj = asldata.asldata()
    assert obj.get_ld() == []


@pytest.mark.parametrize(
    'input, expected',
    [([1, 2, 3], [1, 2, 3]), ([1.5, 2.5, 3.5], [1.5, 2.5, 3.5])],
)
def test_set_ld_update_object_ld_parameters(input, expected):
    obj = asldata.asldata()
    obj.set_ld(input)
    assert obj.get_ld() == expected


def test_get_pld_show_empty_list_for_new_object():
    obj = asldata.asldata()
    assert obj.get_pld() == []


@pytest.mark.parametrize(
    'input, expected',
    [([1, 2, 3], [1, 2, 3]), ([1.5, 2.5, 3.5], [1.5, 2.5, 3.5])],
)
def test_set_pld_update_object_pld_parameters(input, expected):
    obj = asldata.asldata()
    obj.set_pld(input)
    assert obj.get_pld() == expected


def test_load_image_update_object_image_reference():
    obj = asldata.asldata()
    assert obj.image == None
    obj.load_image(T1_MRI)
    assert isinstance(obj.image, np.ndarray)


@pytest.mark.parametrize(
    'input', [('out.nrrd'), ('out.nii'), ('out.mha'), ('out.tif')]
)
def test_save_image_success(input, tmp_path):
    obj = asldata.asldata()
    obj.load_image(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    obj.save_image(full_path)
    assert os.path.exists(full_path)
    read_file = sitk.ReadImage(full_path)
    assert read_file.GetSize() == sitk.ReadImage(T1_MRI).GetSize()


@pytest.mark.parametrize(
    'input', [('out.nrr'), ('out.n'), ('out.m'), ('out.zip')]
)
def test_save_image_throw_error_invalid_formatt(input, tmp_path):
    obj = asldata.asldata()
    obj.load_image(T1_MRI)
    full_path = tmp_path.as_posix() + os.sep + input
    with pytest.raises(Exception) as e:
        obj.save_image(full_path)


def test_get_te_is_none_for_new_object():
    obj = asldata.asldata()
    assert obj.get_te() == None


@pytest.mark.parametrize(
    'input, expected',
    [
        ([10, 20, 30, 40], [10, 20, 30, 40]),
        ([10.1, 20.1, 30.1, 40.1], [10.1, 20.1, 30.1, 40.1]),
    ],
)
def test_set_te_update_parameter_from_input_as_list(input, expected):
    obj = asldata.asldata()
    obj.set_te(input)
    assert obj.get_te() == expected


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 1]), (['a', 10, 1.5])]
)
def test_set_te_throw_error_input_not_a_list_of_numbers(input):
    obj = asldata.asldata()
    with pytest.raises(Exception) as e:
        obj.set_te(input)
    assert e.value.args[0] == 'TE values is not a list of valid numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [([-1, -2, -3]), ([-10, -20.1]), ([0, -0, 10.1, 2])]
)
def test_set_te_throw_error_input_is_list_of_negative_or_zero_numbers(input):
    obj = asldata.asldata()
    with pytest.raises(Exception) as e:
        obj.set_te(input)
    assert e.value.args[0] == 'TE values must be postive non zero numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 1]), (['a', 10, 1.5])]
)
def test_set_ld_throw_error_input_not_a_list_of_numbers(input):
    obj = asldata.asldata()
    with pytest.raises(Exception) as e:
        obj.set_ld(input)
    assert e.value.args[0] == 'LD values is not a list of valid numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [([-1, -2, -3]), ([-10, -20.1]), ([0, -0, 10.1, 2])]
)
def test_set_ld_throw_error_input_is_list_of_negative_or_zero_numbers(input):
    obj = asldata.asldata()
    with pytest.raises(Exception) as e:
        obj.set_ld(input)
    assert e.value.args[0] == 'LD values must be postive non zero numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 1]), (['a', 10, 1.5])]
)
def test_set_pld_throw_error_input_not_a_list_of_numbers(input):
    obj = asldata.asldata()
    with pytest.raises(Exception) as e:
        obj.set_pld(input)
    assert e.value.args[0] == 'PLD values is not a list of valid numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [([-1, -2, -3]), ([-10, -20.1]), ([0, -0, 10.1, 2])]
)
def test_set_pld_throw_error_input_is_list_of_negative_or_zero_numbers(input):
    obj = asldata.asldata()
    with pytest.raises(Exception) as e:
        obj.set_pld(input)
    assert e.value.args[0] == 'PLD values must be postive non zero numbers.'
    assert e.type == ValueError
