import os

import numpy as np
import pytest

from asltk.utils.image_statistics import (
    analyze_image_properties,
    calculate_mean_intensity,
    calculate_snr,
)
from asltk.utils.io import load_image

SEP = os.sep
T1_MRI = f'tests{SEP}files{SEP}t1-mri.nrrd'
PCASL_MTE = f'tests{SEP}files{SEP}pcasl_mte.nii.gz'
M0 = f'tests{SEP}files{SEP}m0.nii.gz'
M0_BRAIN_MASK = f'tests{SEP}files{SEP}m0_brain_mask.nii.gz'


@pytest.mark.parametrize('image_path', [T1_MRI, PCASL_MTE, M0])
def test_analyze_image_properties_returns_dict(image_path):
    """Test that analyze_image_properties returns a dictionary with expected keys."""
    img = load_image(image_path)
    props = analyze_image_properties(img)
    assert isinstance(props, dict)
    assert 'shape' in props
    assert 'intensity_stats' in props
    assert 'center_of_mass' in props
    assert 'min' in props['intensity_stats']
    assert 'max' in props['intensity_stats']
    assert 'mean' in props['intensity_stats']
    assert 'std' in props['intensity_stats']


@pytest.mark.parametrize(
    'input',
    ['invalid/path/to/image.nii', 1, -2.4, (1, 2), {'wrong': 1, 'input': 2}],
)
def test_analyze_image_properties_invalid_path(input):
    """Test that an invalid path raises an exception."""
    with pytest.raises(Exception) as error:
        analyze_image_properties(input)

    assert len(str(error.value)) > 0


@pytest.mark.parametrize('image_path', [T1_MRI, PCASL_MTE, M0])
def test_calculate_snr_returns_float(image_path):
    """Test that calculate_snr returns a float for valid images."""
    img = load_image(image_path)
    snr = calculate_snr(img)
    assert isinstance(snr, float)
    assert snr >= 0


@pytest.mark.parametrize('image_path', [T1_MRI, PCASL_MTE, M0])
def test_calculate_snr_returns_float_using_valid_roi(image_path):
    """Test that calculate_snr returns a float for valid images."""
    img = load_image(image_path)
    roi = np.ones(img.shape, dtype=bool)  # Create a valid ROI
    snr = calculate_snr(img, roi=roi)
    assert isinstance(snr, float)
    assert snr >= 0


def test_calculate_snr_make_zero_division_with_same_image_input():
    """Test that calculate_snr handles zero division with same image input."""
    img = np.ones((10, 10, 10))  # Create a simple image
    roi = np.ones(img.shape, dtype=bool)  # Create a valid ROI
    snr = calculate_snr(img, roi=roi)

    assert isinstance(snr, float)
    assert snr == float('inf')  # SNR should be infinite for uniform image


@pytest.mark.parametrize(
    'input', [np.zeros((10, 10)), np.ones((5, 5, 5)), np.full((3, 3), 7)]
)
def test_calculate_snr_known_arrays(input):
    """Test calculate_snr with known arrays."""
    snr = calculate_snr(input)
    assert isinstance(snr, float)


def test_calculate_snr_invalid_input():
    """Test that calculate_snr raises an error for invalid input."""
    with pytest.raises(Exception) as error:
        calculate_snr('invalid_input')

    assert len(str(error.value)) > 0


@pytest.mark.parametrize('image_path', [T1_MRI, PCASL_MTE, M0])
def test_calculate_snr_raise_error_roi_different_shape(image_path):
    """Test that calculate_snr raises an error for ROI of different shape."""
    img = load_image(image_path)
    # Add an extra dimension to img and create a mismatched ROI
    img = np.expand_dims(img, axis=0)
    roi = np.ones(
        img.shape[1:], dtype=bool
    )  # ROI shape does not match img shape
    with pytest.raises(ValueError) as error:
        calculate_snr(img, roi=roi)

    assert 'ROI shape must match image shape' in str(error.value)


@pytest.mark.parametrize('image_path', [T1_MRI, PCASL_MTE, M0])
def test_calculate_snr_raise_error_roi_not_numpy_array(image_path):
    """Test that calculate_snr raises an error for ROI not being a numpy array."""
    img = load_image(image_path)
    roi = 'invalid_roi'
    with pytest.raises(ValueError) as error:
        calculate_snr(img, roi=roi)

    assert 'ROI must be a numpy array' in str(error.value)


@pytest.mark.parametrize('image_path', [T1_MRI, PCASL_MTE, M0])
def test_calculate_mean_intensity_returns_float(image_path):
    """Test that calculate_mean_intensity returns a float for valid images."""
    img = load_image(image_path)
    mean_intensity = calculate_mean_intensity(img)
    assert isinstance(mean_intensity, float)
    assert mean_intensity >= 0


@pytest.mark.parametrize('image_path', [T1_MRI, PCASL_MTE, M0])
def test_calculate_mean_intensity_with_valid_roi(image_path):
    """Test that calculate_mean_intensity returns a float for valid ROI."""
    img = load_image(image_path)
    roi = np.ones(img.shape, dtype=bool)
    mean_intensity = calculate_mean_intensity(img, roi=roi)
    assert isinstance(mean_intensity, float)
    assert mean_intensity >= 0


def test_calculate_mean_intensity_known_arrays():
    """Test calculate_mean_intensity with known arrays."""
    arr = np.ones((5, 5, 5))
    mean_intensity = calculate_mean_intensity(arr)
    assert mean_intensity == 1.0

    arr = np.full((3, 3), 7)
    mean_intensity = calculate_mean_intensity(arr)
    assert mean_intensity == 7.0

    arr = np.array([[1, 2], [3, 4]])
    mean_intensity = calculate_mean_intensity(arr)
    assert mean_intensity == 2.5


def test_calculate_mean_intensity_with_roi_mask():
    """Test calculate_mean_intensity with ROI mask."""
    arr = np.array([[1, 2], [3, 4]])
    roi = np.array([[0, 1], [1, 0]])
    mean_intensity = calculate_mean_intensity(arr, roi=roi)
    assert mean_intensity == 2.5  # mean of [2, 3]


def test_calculate_mean_intensity_invalid_input():
    """Test that calculate_mean_intensity raises an error for invalid input."""
    with pytest.raises(ValueError) as error:
        calculate_mean_intensity('invalid_input')
    assert 'Input must be a numpy array' in str(error.value)


def test_calculate_mean_intensity_roi_not_numpy_array():
    """Test that calculate_mean_intensity raises an error for ROI not being a numpy array."""
    arr = np.ones((5, 5))
    roi = 'invalid_roi'
    with pytest.raises(ValueError) as error:
        calculate_mean_intensity(arr, roi=roi)
    assert 'ROI must be a numpy array' in str(error.value)


def test_calculate_mean_intensity_roi_shape_mismatch():
    """Test that calculate_mean_intensity raises an error for ROI shape mismatch."""
    arr = np.ones((5, 5))
    roi = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError) as error:
        calculate_mean_intensity(arr, roi=roi)
    assert 'ROI shape must match image shape' in str(error.value)
