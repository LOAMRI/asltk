import os

import numpy as np
import pytest

from asltk.smooth.gaussian import isotropic_gaussian
from asltk.utils import load_image

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
    data = load_image(PCASL_MTE)
    smoothed = isotropic_gaussian(data, sigma)
    assert smoothed.shape == data.shape
    assert np.mean(smoothed) != np.mean(data)
    assert np.std(smoothed) < np.std(data)


@pytest.mark.parametrize(
    'sigma',
    [
        (-3),
        (-5.1),
        (0),
    ],
)
def test_isotropic_gaussian_smooth_wrong_sigma(sigma):
    data = load_image(PCASL_MTE)
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
    assert 'data is not a numpy array. Type' in e.value.args[0]


def test_isotropic_gaussian_3D_volume_sucess():
    data = load_image(M0)
    smoothed = isotropic_gaussian(data)
    assert smoothed.shape == data.shape
    assert np.mean(smoothed) != np.mean(data)
    assert np.std(smoothed) < np.std(data)
