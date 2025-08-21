import os

import numpy as np
import pytest

from asltk.smooth.gaussian import isotropic_gaussian
from asltk.smooth.median import isotropic_median
from asltk.utils.io import ImageIO

SEP = os.sep
PCASL_MTE = f'tests' + SEP + 'files' + SEP + 'pcasl_mte.nii.gz'
M0 = f'tests' + SEP + 'files' + SEP + 'm0.nii.gz'


@pytest.mark.parametrize(
    'sigma',
    [
        (3),
        (5),
        (10),
        (3.5),
        (11.5),
    ],
)
def test_isotropic_gaussian_smooth(sigma):
    data = ImageIO(PCASL_MTE)
    smoothed = isotropic_gaussian(data, sigma)
    assert smoothed.get_as_numpy().shape == data.get_as_numpy().shape
    assert np.mean(smoothed.get_as_numpy()) != np.mean(data.get_as_numpy())
    assert np.std(smoothed.get_as_numpy()) < np.std(data.get_as_numpy())


@pytest.mark.parametrize(
    'sigma',
    [
        (-3),
        (-5.1),
        (0),
    ],
)
def test_isotropic_gaussian_smooth_wrong_sigma(sigma):
    data = ImageIO(PCASL_MTE)
    with pytest.raises(Exception) as e:
        isotropic_gaussian(data, sigma)
    assert 'sigma must be a positive number.' in e.value.args[0]


@pytest.mark.parametrize(
    'data',
    [
        ('np.array([1, 2, 3])'),
        (-5.1),
        ([1, 2, 3]),
    ],
)
def test_isotropic_gaussian_smooth_wrong_data(data):
    with pytest.raises(Exception) as e:
        isotropic_gaussian(data)
    assert 'data is not an ImageIO object. Type' in e.value.args[0]


def test_isotropic_gaussian_3D_volume_sucess():
    data = ImageIO(M0)
    smoothed = isotropic_gaussian(data)
    assert smoothed.get_as_numpy().shape == data.get_as_numpy().shape
    assert np.mean(smoothed.get_as_numpy()) != np.mean(data.get_as_numpy())
    assert np.std(smoothed.get_as_numpy()) < np.std(data.get_as_numpy())


@pytest.mark.parametrize(
    'size',
    [
        (3),
        (5),
        (7),
    ],
)
def test_isotropic_median_smooth(size):
    data = ImageIO(PCASL_MTE)
    smoothed = isotropic_median(data, size)
    assert smoothed.get_as_numpy().shape == data.get_as_numpy().shape
    assert np.mean(smoothed.get_as_numpy()) != np.mean(data.get_as_numpy())
    assert np.std(smoothed.get_as_numpy()) < np.std(data.get_as_numpy())


@pytest.mark.parametrize(
    'size',
    [
        (-3),
        (-5),
        (0),
        (2.5),
        ('5'),
    ],
)
def test_isotropic_median_smooth_wrong_size(size):
    data = ImageIO(PCASL_MTE)
    with pytest.raises(Exception) as e:
        isotropic_median(data, size)
    assert 'size must be a positive integer.' in e.value.args[0]


@pytest.mark.parametrize(
    'data',
    [
        ('np.array([1, 2, 3])'),
        (-5.1),
        ([1, 2, 3]),
    ],
)
def test_isotropic_median_smooth_wrong_data(data):
    with pytest.raises(Exception) as e:
        isotropic_median(data)
    assert 'data is not an ImageIO object. Type' in e.value.args[0]


def test_isotropic_median_3D_volume_success():
    data = ImageIO(M0)
    smoothed = isotropic_median(data)
    assert smoothed.get_as_numpy().shape == data.get_as_numpy().shape
    assert np.mean(smoothed.get_as_numpy()) != np.mean(data.get_as_numpy())
    assert np.std(smoothed.get_as_numpy()) < np.std(data.get_as_numpy())


def test_isotropic_median_even_size_warning():
    data = ImageIO(M0)
    with pytest.warns(UserWarning) as warning:
        smoothed = isotropic_median(data, size=4)
    assert 'size was even, using 3 instead' in str(warning[0].message)
    assert smoothed.get_as_numpy().shape == data.get_as_numpy().shape
