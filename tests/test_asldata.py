import os

import numpy as np
import pytest
import SimpleITK as sitk

from asltk.utils import load_image

from asltk import asldata

SEP = os.sep
T1_MRI = f'tests' + SEP + 'files' + SEP + 't1-mri.nrrd'
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'
M0_BRAIN_MASK = f'tests' + SEP + 'files' + SEP + 'm0_brain_mask.nii.gz'


def test_create_successfuly_asldata_object():
    obj = asldata.ASLData()
    assert isinstance(obj, asldata.ASLData)


def test_create_successfuly_asldata_object_with_inputs():
    obj_0 = asldata.ASLData(m0=M0)
    assert isinstance(obj_0, asldata.ASLData)
    assert len(obj_0.get_ld()) == 0
    assert len(obj_0.get_pld()) == 0
    assert obj_0.get_te() == None
    assert obj_0.get_dw() == None
    obj_1 = asldata.ASLData(pcasl=PCASL_MTE)
    assert isinstance(obj_1, asldata.ASLData)
    assert len(obj_1.get_ld()) == 0
    assert len(obj_1.get_pld()) == 0
    assert obj_1.get_te() == None
    assert obj_1.get_dw() == None
    obj_3 = asldata.ASLData(
        pcasl=PCASL_MTE, ld_values=[1, 2, 3], pld_values=[1, 2, 3]
    )
    assert isinstance(obj_3, asldata.ASLData)
    assert len(obj_3.get_ld()) == 3
    assert len(obj_3.get_pld()) == 3
    assert obj_3.get_te() == None
    assert obj_3.get_dw() == None
    obj_4 = asldata.ASLData(
        pcasl=PCASL_MTE,
        ld_values=[1, 2, 3],
        pld_values=[1, 2, 3],
        te_values=[1, 2, 3],
    )
    assert isinstance(obj_4, asldata.ASLData)
    assert len(obj_4.get_ld()) == 3
    assert len(obj_4.get_pld()) == 3
    assert len(obj_4.get_te()) == 3
    assert obj_4.get_dw() == None
    obj_5 = asldata.ASLData(
        pcasl=PCASL_MTE,
        ld_values=[1, 2, 3],
        pld_values=[1, 2, 3],
        te_values=[1, 2, 3],
        dw_values=[1, 2, 3],
    )
    assert isinstance(obj_5, asldata.ASLData)
    assert len(obj_5.get_ld()) == 3
    assert len(obj_5.get_pld()) == 3
    assert len(obj_5.get_te()) == 3
    assert len(obj_5.get_dw()) == 3


def test_create_object_with_different_image_formats():
    obj = asldata.ASLData(pcasl=PCASL_MTE)
    assert isinstance(obj, asldata.ASLData)
    obj = asldata.ASLData(m0=M0)
    assert isinstance(obj, asldata.ASLData)


def test_create_object_check_initial_parameters():
    obj = asldata.ASLData()
    assert obj.get_ld() == []
    assert obj.get_pld() == []

def test_create_object_with_m0_as_numpy_array():
    array = load_image(M0)
    obj = asldata.ASLData(m0=array)

    assert obj('m0').shape == array.shape

def test_get_ld_show_empty_list_for_new_object():
    obj = asldata.ASLData()
    assert obj.get_ld() == []


@pytest.mark.parametrize(
    'input, expected',
    [([1, 2, 3], [1, 2, 3]), ([1.5, 2.5, 3.5], [1.5, 2.5, 3.5])],
)
def test_set_ld_update_object_ld_parameters(input, expected):
    obj = asldata.ASLData()
    obj.set_ld(input)
    assert obj.get_ld() == expected


def test_get_pld_show_empty_list_for_new_object():
    obj = asldata.ASLData()
    assert obj.get_pld() == []


@pytest.mark.parametrize(
    'input, expected',
    [([1, 2, 3], [1, 2, 3]), ([1.5, 2.5, 3.5], [1.5, 2.5, 3.5])],
)
def test_set_pld_update_object_pld_parameters(input, expected):
    obj = asldata.ASLData()
    obj.set_pld(input)
    assert obj.get_pld() == expected


def test_get_te_is_none_for_new_object():
    obj = asldata.ASLData()
    assert obj.get_te() == None


@pytest.mark.parametrize(
    'input, expected',
    [
        ([10, 20, 30, 40], [10, 20, 30, 40]),
        ([10.1, 20.1, 30.1, 40.1], [10.1, 20.1, 30.1, 40.1]),
    ],
)
def test_set_te_update_parameter_from_input_as_list(input, expected):
    obj = asldata.ASLData()
    obj.set_te(input)
    assert obj.get_te() == expected


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 1]), (['a', 10, 1.5])]
)
def test_set_te_throw_error_input_not_a_list_of_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_te(input)
    assert e.value.args[0] == 'TE values is not a list of valid numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [([-1, -2, -3]), ([-10, -20.1]), ([0, -0, 10.1, 2])]
)
def test_set_te_throw_error_input_is_list_of_negative_or_zero_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_te(input)
    assert e.value.args[0] == 'TE values must be postive non zero numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 1]), (['a', 10, 1.5])]
)
def test_set_ld_throw_error_input_not_a_list_of_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_ld(input)
    assert e.value.args[0] == 'LD values is not a list of valid numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [([-1, -2, -3]), ([-10, -20.1]), ([0, -0, 10.1, 2])]
)
def test_set_ld_throw_error_input_is_list_of_negative_or_zero_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_ld(input)
    assert e.value.args[0] == 'LD values must be postive non zero numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 1]), (['a', 10, 1.5])]
)
def test_set_pld_throw_error_input_not_a_list_of_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_pld(input)
    assert e.value.args[0] == 'PLD values is not a list of valid numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [([-1, -2, -3]), ([-10, -20.1]), ([0, -0, 10.1, 2])]
)
def test_set_pld_throw_error_input_is_list_of_negative_or_zero_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_pld(input)
    assert e.value.args[0] == 'PLD values must be postive non zero numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input_ld,input_pld',
    [
        ([1, 2, 3], [1, 2]),
        ([1, 2], [1, 2, 3]),
        ([1.0, 2.0, 3.0], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
    ],
)
def test_asl_data_raise_error_if_ld_and_pld_have_different_sizes(
    input_ld, input_pld
):
    with pytest.raises(Exception) as e:
        data = asldata.ASLData(ld_values=input_ld, pld_values=input_pld)
    assert (
        e.value.args[0]
        == f'LD and PLD must have the same array size. LD size is {len(input_ld)} and PLD size is {len(input_pld)}'
    )


@pytest.mark.parametrize(
    'input, expected',
    [
        ([10, 20, 30, 40], [10, 20, 30, 40]),
        ([10.1, 20.1, 30.1, 40.1], [10.1, 20.1, 30.1, 40.1]),
    ],
)
def test_set_dw_update_parameter_from_input_as_list(input, expected):
    obj = asldata.ASLData()
    obj.set_dw(input)
    assert obj.get_dw() == expected


@pytest.mark.parametrize(
    'input', [(['a', 'b', 'c']), (['a', 'b', 1]), (['a', 10, 1.5])]
)
def test_set_dw_throw_error_input_not_a_list_of_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_dw(input)
    assert e.value.args[0] == 'DW values is not a list of valid numbers.'
    assert e.type == ValueError


@pytest.mark.parametrize(
    'input', [([-1, -2, -3]), ([-10, -20.1]), ([0, -0, 10.1, 2])]
)
def test_set_dw_throw_error_input_is_list_of_negative_or_zero_numbers(input):
    obj = asldata.ASLData()
    with pytest.raises(Exception) as e:
        obj.set_dw(input)
    assert e.value.args[0] == 'DW values must be postive non zero numbers.'
    assert e.type == ValueError


def test_asldata_object_call_returns_image():
    obj = asldata.ASLData(pcasl=T1_MRI)
    assert isinstance(obj('pcasl'), np.ndarray)


def test_set_image_sucess_m0():
    obj = asldata.ASLData(pcasl=T1_MRI)
    obj.set_image(M0, 'm0')
    assert isinstance(obj('m0'), np.ndarray)


def test_set_image_sucess_pcasl():
    obj = asldata.ASLData()
    obj.set_image(M0, 'pcasl')
    assert isinstance(obj('pcasl'), np.ndarray)
