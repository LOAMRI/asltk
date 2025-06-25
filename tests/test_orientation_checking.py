# """
# Tests for orientation checking and correction functionality.
# """

# import numpy as np
# import pytest

# from asltk.registration import (
#     _compute_normalized_correlation,
#     _normalize_image_intensity,
#     check_and_fix_orientation,
#     orientation_check,
# )


# class TestOrientationChecking:
#     """Test cases for orientation checking functionality."""

#     def setup_method(self):
#         """Set up test data."""
#         # Create a simple 3D image with identifiable features
#         self.test_image = np.zeros((20, 30, 40))

#         # Add some features to make orientation detection meaningful
#         self.test_image[5:15, 10:20, 15:25] = 100  # Central bright region
#         self.test_image[2:4, 5:25, 10:30] = 50     # Top bright strip
#         self.test_image[16:18, 5:25, 10:30] = 50   # Bottom bright strip

#         # Add some noise
#         noise = np.random.rand(*self.test_image.shape) * 10
#         self.test_image += noise

#     def test_identical_images(self):
#         """Test that identical images have high correlation."""
#         # Test with identical images
#         corrected, transform = check_and_fix_orientation(
#             self.test_image, self.test_image, verbose=False
#         )

#         # Should not apply any transformations
#         assert not transform['flip_x']
#         assert not transform['flip_y']
#         assert not transform['flip_z']
#         assert transform['transpose_axes'] is None

#         # Images should be nearly identical
#         np.testing.assert_array_almost_equal(corrected, self.test_image)

#     def test_axial_flip_detection(self):
#         """Test detection and correction of axial flip."""
#         # Create axially flipped version
#         flipped_image = np.flip(self.test_image, axis=0)

#         # Check and fix orientation
#         corrected, transform = check_and_fix_orientation(
#             flipped_image, self.test_image, verbose=False
#         )

#         # Should detect Z-axis flip
#         assert transform['flip_z'] == True

#         # Corrected image should be closer to original
#         original_corr = _compute_normalized_correlation(
#             flipped_image, self.test_image
#         )
#         corrected_corr = _compute_normalized_correlation(
#             corrected, self.test_image
#         )
#         assert corrected_corr > original_corr

#     def test_sagittal_flip_detection(self):
#         """Test detection and correction of sagittal flip."""
#         # Create sagittally flipped version
#         flipped_image = np.flip(self.test_image, axis=2)

#         # Check and fix orientation
#         corrected, transform = check_and_fix_orientation(
#             flipped_image, self.test_image, verbose=False
#         )

#         # Should detect X-axis flip
#         assert transform['flip_x'] == True

#         # Corrected image should be closer to original
#         original_corr = _compute_normalized_correlation(
#             flipped_image, self.test_image
#         )
#         corrected_corr = _compute_normalized_correlation(
#             corrected, self.test_image
#         )
#         assert corrected_corr > original_corr

#     def test_coronal_flip_detection(self):
#         """Test detection and correction of coronal flip."""
#         # Create coronally flipped version
#         flipped_image = np.flip(self.test_image, axis=1)

#         # Check and fix orientation
#         corrected, transform = check_and_fix_orientation(
#             flipped_image, self.test_image, verbose=False
#         )

#         # Should detect Y-axis flip
#         assert transform['flip_y'] == True

#         # Corrected image should be closer to original
#         original_corr = _compute_normalized_correlation(
#             flipped_image, self.test_image
#         )
#         corrected_corr = _compute_normalized_correlation(
#             corrected, self.test_image
#         )
#         assert corrected_corr > original_corr

#     def test_orientation_check(self):
#         """Test quick orientation compatibility check."""
#         # Test with identical images
#         result = orientation_check(self.test_image, self.test_image)
#         assert result['compatible'] == True
#         assert result['correlation'] > 0.9

#         # Test with flipped image
#         flipped_image = np.flip(self.test_image, axis=0)
#         result = orientation_check(flipped_image, self.test_image)
#         # May or may not be compatible depending on the threshold and symmetry
#         assert 'compatible' in result
#         assert 'correlation' in result
#         assert 'recommendation' in result

#     def test_normalize_image_intensity(self):
#         """Test image intensity normalization."""
#         # Test with positive values
#         test_array = np.array(
#             [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
#         )
#         normalized = _normalize_image_intensity(test_array)

#         assert np.min(normalized) == 0.0
#         assert np.max(normalized) == 1.0
#         assert normalized.shape == test_array.shape

#         # Test with constant values
#         constant_array = np.ones((5, 5, 5)) * 42
#         normalized_constant = _normalize_image_intensity(constant_array)
#         # Should handle constant values gracefully
#         assert normalized_constant.shape == constant_array.shape

#     def test_compute_normalized_correlation(self):
#         """Test normalized correlation computation."""
#         # Test with identical arrays
#         corr = _compute_normalized_correlation(
#             self.test_image, self.test_image
#         )
#         assert corr == 1.0

#         # Test with completely different arrays
#         random_image = np.random.rand(*self.test_image.shape) * 1000
#         corr = _compute_normalized_correlation(self.test_image, random_image)
#         assert 0 <= corr <= 1

#         # Test with different shapes (should return -1)
#         different_shape = np.random.rand(10, 10, 10)
#         corr = _compute_normalized_correlation(
#             self.test_image, different_shape
#         )
#         assert corr == -1

#     def test_multiple_transformations(self):
#         """Test detection of multiple orientation issues."""
#         # Apply both flip and transpose
#         transformed_image = np.flip(self.test_image, axis=0)  # Z flip
#         transformed_image = np.flip(transformed_image, axis=1)  # Y flip

#         # Check and fix orientation
#         corrected, transform = check_and_fix_orientation(
#             transformed_image, self.test_image, verbose=False
#         )

#         # Should detect multiple flips
#         flip_count = sum(
#             [transform['flip_x'], transform['flip_y'], transform['flip_z']]
#         )
#         assert flip_count >= 1  # At least one flip should be detected

#         # Corrected image should be closer to original
#         original_corr = _compute_normalized_correlation(
#             transformed_image, self.test_image
#         )
#         corrected_corr = _compute_normalized_correlation(
#             corrected, self.test_image
#         )
#         assert corrected_corr >= original_corr

#     def test_edge_cases(self):
#         """Test edge cases and error handling."""
#         # Test with very small images
#         small_image = np.random.rand(2, 2, 2)
#         small_fixed = np.random.rand(2, 2, 2)

#         # Should not crash
#         corrected, transform = check_and_fix_orientation(
#             small_image, small_fixed, verbose=False
#         )
#         assert corrected.shape == small_image.shape

#         # Test with zero images
#         zero_image = np.zeros((10, 10, 10))
#         zero_fixed = np.zeros((10, 10, 10))

#         # Should handle gracefully
#         corrected, transform = check_and_fix_orientation(
#             zero_image, zero_fixed, verbose=False
#         )
#         assert corrected.shape == zero_image.shape


# if __name__ == '__main__':
#     # Run tests manually if pytest is not available
#     test_case = TestOrientationChecking()
#     test_case.setup_method()

#     print('Running orientation checking tests...')

#     try:
#         test_case.test_identical_images()
#         print('✓ Identical images test passed')

#         test_case.test_axial_flip_detection()
#         print('✓ Axial flip detection test passed')

#         test_case.test_sagittal_flip_detection()
#         print('✓ Sagittal flip detection test passed')

#         test_case.test_coronal_flip_detection()
#         print('✓ Coronal flip detection test passed')

#         test_case.test_orientation_check()
#         print('✓ Quick orientation check test passed')

#         test_case.test_normalize_image_intensity()
#         print('✓ Image normalization test passed')

#         test_case.test_compute_normalized_correlation()
#         print('✓ Correlation computation test passed')

#         test_case.test_multiple_transformations()
#         print('✓ Multiple transformations test passed')

#         test_case.test_edge_cases()
#         print('✓ Edge cases test passed')

#         print('\nAll tests passed! ✓')

#     except Exception as e:
#         print(f'\nTest failed: {e}')
#         import traceback

#         traceback.print_exc()
