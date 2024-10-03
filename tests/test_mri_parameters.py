import numpy as np
import pytest

from asltk.mri_parameters import MRIParameters


@pytest.mark.parametrize(
    'constant,value',
    [
        ('T1bl', 100),
        ('T1csf', 123),
        ('T2bl', 510),
        ('T2gm', 1000),
        ('T2csf', 234),
        ('Alpha', 0.44),
        ('Lambda', 0.51),
    ],
)
def test_set_constant_and_get_constant_success(constant, value):
    mri = MRIParameters()
    mri.set_constant(value, constant)
    assert mri.get_constant(constant) == value


@pytest.mark.parametrize(
    'wrong_constant',
    [
        ('T1blood'),
        ('T3'),
        ('T2blGM'),
        ('Not_T'),
        ('Invalid_text'),
        (10),
        ('abc'),
        (100.5),
        (np.ones((1, 1, 1))),
    ],
)
def test_set_constant_raise_error_invalid_type_of_constant(wrong_constant):
    mri = MRIParameters()
    with pytest.raises(Exception) as error:
        mri.set_constant(1, wrong_constant)
    assert (
        error.value.args[0]
        == f'Constant type {wrong_constant} is not valid. Choose in the list available in the MRIParameter class.'
    )


@pytest.mark.parametrize(
    'wrong_constant',
    [
        ('T1blood'),
        ('T3'),
        ('T2blGM'),
        ('Not_T'),
        ('Invalid_text'),
        (10),
        ('abc'),
        (100.5),
        (np.ones((1, 1, 1))),
    ],
)
def test_get_constant_raise_error_invalid_type_of_constant(wrong_constant):
    mri = MRIParameters()
    with pytest.raises(Exception) as error:
        mri.get_constant(wrong_constant)
    assert (
        error.value.args[0]
        == f'Constant type {wrong_constant} is not valid. Choose in the list available in the MRIParameter class.'
    )
