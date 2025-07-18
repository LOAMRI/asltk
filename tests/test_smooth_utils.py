import numpy as np
import pytest

from asltk.aux_methods import _apply_smoothing_to_maps


def test_apply_smoothing_to_maps_no_smoothing():
    # Test no smoothing (default behavior)
    maps = {
        'cbf': np.random.random((10, 10, 10)),
        'att': np.random.random((10, 10, 10)),
    }
    result = _apply_smoothing_to_maps(maps)

    # Should return identical maps
    assert set(result.keys()) == set(maps.keys())
    for key in maps.keys():
        assert np.array_equal(result[key], maps[key])


def test_apply_smoothing_to_maps_gaussian():
    # Test gaussian smoothing
    maps = {
        'cbf': np.random.random((10, 10, 10)),
        'att': np.random.random((10, 10, 10)),
    }
    result = _apply_smoothing_to_maps(maps, smoothing='gaussian')

    # Should return different smoothed maps
    assert set(result.keys()) == set(maps.keys())
    for key in maps.keys():
        assert result[key].shape == maps[key].shape
        assert not np.array_equal(result[key], maps[key])
        # Smoothing should reduce noise (typically lower std)
        assert np.std(result[key]) <= np.std(maps[key])


def test_apply_smoothing_to_maps_median():
    # Test median smoothing
    maps = {
        'cbf': np.random.random((10, 10, 10)),
        'att': np.random.random((10, 10, 10)),
    }
    result = _apply_smoothing_to_maps(
        maps, smoothing='median', smoothing_params={'size': 3}
    )

    # Should return different smoothed maps
    assert set(result.keys()) == set(maps.keys())
    for key in maps.keys():
        assert result[key].shape == maps[key].shape
        assert not np.array_equal(result[key], maps[key])


def test_apply_smoothing_to_maps_invalid_type():
    # Test invalid smoothing type
    maps = {'cbf': np.random.random((10, 10, 10))}

    with pytest.raises(ValueError) as e:
        _apply_smoothing_to_maps(maps, smoothing='invalid')
    assert 'Unsupported smoothing type: invalid' in str(e.value)


def test_apply_smoothing_to_maps_non_array_values():
    # Test that non-array values are passed through unchanged
    maps = {
        'cbf': np.random.random((10, 10, 10)),
        'metadata': 'some_string',
        'number': 42,
    }
    result = _apply_smoothing_to_maps(maps, smoothing='gaussian')

    # Non-array values should be unchanged
    assert result['metadata'] == maps['metadata']
    assert result['number'] == maps['number']
    # Array should be smoothed
    assert not np.array_equal(result['cbf'], maps['cbf'])


def test_apply_smoothing_to_maps_custom_params():
    # Test custom smoothing parameters
    maps = {'cbf': np.random.random((10, 10, 10))}

    result1 = _apply_smoothing_to_maps(
        maps, smoothing='gaussian', smoothing_params={'sigma': 1.0}
    )
    result2 = _apply_smoothing_to_maps(
        maps, smoothing='gaussian', smoothing_params={'sigma': 2.0}
    )

    # Different parameters should produce different results
    assert not np.array_equal(result1['cbf'], result2['cbf'])


def test_apply_smoothing_to_maps_median_default_params():
    # Test median smoothing with default parameters
    maps = {
        'cbf': np.random.random((10, 10, 10)),
        'att': np.random.random((10, 10, 10)),
    }
    result = _apply_smoothing_to_maps(maps, smoothing='median')
    for key in maps.keys():
        assert result[key].shape == maps[key].shape
        assert not np.array_equal(result[key], maps[key])


def test_apply_smoothing_to_maps_median_different_sizes():
    # Test median smoothing with different kernel sizes
    maps = {'cbf': np.random.random((10, 10, 10))}
    result1 = _apply_smoothing_to_maps(
        maps, smoothing='median', smoothing_params={'size': 3}
    )
    result2 = _apply_smoothing_to_maps(
        maps, smoothing='median', smoothing_params={'size': 5}
    )
    assert not np.array_equal(result1['cbf'], result2['cbf'])


def test_apply_smoothing_to_maps_median_invalid_param():
    # Test median smoothing with invalid parameter
    maps = {'cbf': np.random.random((10, 10, 10))}
    with pytest.raises(Exception) as error:
        _apply_smoothing_to_maps(
            maps, smoothing='median', smoothing_params={'size': 'invalid'}
        )

    assert 'Invalid smoothing parameter type' in str(error.value)


def test_apply_smoothing_to_maps_median_non_array():
    # Test median smoothing with non-array values in maps
    maps = {'cbf': np.random.random((10, 10, 10)), 'meta': 'info'}
    result = _apply_smoothing_to_maps(maps, smoothing='median')
    assert result['meta'] == maps['meta']
    assert not np.array_equal(result['cbf'], maps['cbf'])


def test_apply_smoothing_to_maps_median_1d_array():
    # Test median smoothing with 1D array
    maps = {'cbf': np.random.random((10, 10, 10))}
    result = _apply_smoothing_to_maps(
        maps, smoothing='median', smoothing_params={'size': 3}
    )
    assert result['cbf'].shape == maps['cbf'].shape
    assert not np.array_equal(result['cbf'], maps['cbf'])
