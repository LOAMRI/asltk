# #!/usr/bin/env python3
# """
# Example script demonstrating orientation checking and correction for image registration.

# This script shows how to use the new orientation checking utilities in ASLtk
# to detect and fix orientation mismatches between medical images before registration.
# """

# import numpy as np

# from asltk.data.brain_atlas import BrainAtlas
# from asltk.registration import (
#     check_and_fix_orientation,
#     create_orientation_report,
#     orientation_check,
#     preview_orientation_correction,
#     space_normalization,
# )
# from asltk.utils.io import load_image


# def example_basic_orientation_check():
#     """Basic example of orientation checking between two images."""
#     print('=== Basic Orientation Check Example ===')

#     # Load your moving and fixed images
#     # Replace these paths with your actual image files
#     try:
#         moving_image = load_image('path/to/your/moving_image.nii.gz')
#         fixed_image = load_image('path/to/your/fixed_image.nii.gz')
#     except:
#         # Create synthetic example data for demonstration
#         print('Using synthetic data for demonstration...')
#         moving_image = np.random.rand(64, 64, 64) * 100
#         # Create a flipped version to simulate orientation mismatch
#         fixed_image = np.flip(moving_image, axis=0)  # Flip axial axis
#         print('Created synthetic moving and fixed images with axial flip')

#     # Quick compatibility check
#     compatibility = orientation_check(moving_image, fixed_image)
#     print(f"Orientation compatible: {compatibility['compatible']}")
#     print(f"Correlation score: {compatibility['correlation']:.4f}")
#     print(f"Recommendation: {compatibility['recommendation']}")

#     # Check and fix orientation
#     corrected_moving, transform = check_and_fix_orientation(
#         moving_image, fixed_image, verbose=True
#     )

#     print(f'Applied transformations: {transform}')
#     print(f'Original shape: {moving_image.shape}')
#     print(f'Corrected shape: {corrected_moving.shape}')

#     return moving_image, fixed_image, corrected_moving, transform


# def example_registration_with_orientation_check():
#     """Example showing registration with automatic orientation checking."""
#     print('\n=== Registration with Orientation Check Example ===')

#     try:
#         # Load your ASL M0 image
#         moving_image = load_image('path/to/your/m0_image.nii.gz')

#         # Load brain atlas as template
#         atlas = BrainAtlas('MNI2009')

#         print('Performing registration with automatic orientation checking...')

#         # Register with orientation checking enabled (default)
#         normalized_image, transforms = space_normalization(
#             moving_image=moving_image,
#             template_image=atlas,
#             transform_type='Affine',
#             check_orientation=True,  # Enable orientation checking
#             orientation_verbose=True,  # Show detailed orientation analysis
#         )

#         print('Registration completed successfully!')
#         print(f'Original image shape: {moving_image.shape}')
#         print(f'Normalized image shape: {normalized_image.shape}')

#         return normalized_image, transforms

#     except Exception as e:
#         print(f'Registration example failed (likely missing data): {e}')
#         print("This is normal if you don't have the required image files.")
#         return None, None


# def example_detailed_orientation_analysis():
#     """Example showing detailed orientation analysis and reporting."""
#     print('\n=== Detailed Orientation Analysis Example ===')

#     # Create synthetic data with known orientation mismatch
#     original_image = np.random.rand(32, 64, 48) * 100

#     # Apply various transformations to simulate orientation mismatches
#     flipped_axial = np.flip(original_image, axis=0)  # Axial flip
#     flipped_sagittal = np.flip(original_image, axis=2)  # Sagittal flip
#     transposed = np.transpose(original_image, (1, 0, 2))  # Transpose X-Y

#     test_cases = [
#         ('Original vs Axial Flip', original_image, flipped_axial),
#         ('Original vs Sagittal Flip', original_image, flipped_sagittal),
#         ('Original vs Transposed', original_image, transposed),
#     ]

#     for case_name, moving, fixed in test_cases:
#         print(f'\n--- {case_name} ---')

#         # Quick check
#         compatibility = orientation_check(moving, fixed)
#         print(f"Compatible: {compatibility['compatible']}")
#         print(f"Correlation: {compatibility['correlation']:.4f}")

#         # Detailed correction
#         corrected, transform = check_and_fix_orientation(
#             moving, fixed, verbose=True
#         )

#         print(f'Applied transform: {transform}')

#         # Generate report
#         report = create_orientation_report(moving, fixed)
#         print('Generated orientation report (first 200 chars):')
#         print(report[:200] + '...')


# def example_manual_orientation_workflow():
#     """Example showing manual workflow for orientation checking."""
#     print('\n=== Manual Orientation Workflow Example ===')

#     # Simulate loading images
#     moving_image = np.random.rand(64, 64, 32) * 100
#     fixed_image = np.flip(moving_image, axis=1)  # Y-axis flip

#     print('Step 1: Quick orientation compatibility check')
#     compatibility = orientation_check(moving_image, fixed_image)
#     print(f'Result: {compatibility}')

#     if not compatibility['compatible']:
#         print('\nStep 2: Preview orientation correction')
#         preview = preview_orientation_correction(moving_image, fixed_image)
#         print(f"Preview generated for slice {preview['slice_index']}")
#         print(f"Original slice shape: {preview['original_slice'].shape}")
#         print(f"Corrected slice shape: {preview['corrected_slice'].shape}")

#         print('\nStep 3: Apply orientation correction')
#         corrected_moving, transform = check_and_fix_orientation(
#             moving_image, fixed_image, verbose=True
#         )

#         print('\nStep 4: Verify improvement')
#         post_correction_check = orientation_check(
#             corrected_moving, fixed_image
#         )
#         print(f'Post-correction compatibility: {post_correction_check}')

#         improvement = (
#             post_correction_check['correlation'] - compatibility['correlation']
#         )
#         print(f'Correlation improvement: {improvement:.4f}')

#         return corrected_moving, transform
#     else:
#         print('Images are already compatible - no correction needed')
#         return moving_image, None


# def example_advanced_usage():
#     """Advanced usage examples and tips."""
#     print('\n=== Advanced Usage Tips ===')

#     # Tip 1: Handling spacing information
#     print('Tip 1: Including voxel spacing for better orientation analysis')
#     moving_image = np.random.rand(64, 64, 32) * 100
#     fixed_image = np.random.rand(64, 64, 32) * 100

#     # Voxel spacing in mm (x, y, z)
#     moving_spacing = (1.0, 1.0, 3.0)  # Typical ASL spacing
#     fixed_spacing = (1.0, 1.0, 1.0)   # Typical T1 spacing

#     corrected_moving, transform = check_and_fix_orientation(
#         moving_image,
#         fixed_image,
#         moving_spacing=moving_spacing,
#         fixed_spacing=fixed_spacing,
#         verbose=True,
#     )

#     # Tip 2: Disabling orientation check for speed
#     print('\nTip 2: Disabling orientation check when not needed')
#     print('Use check_orientation=False in space_normalization() for speed')

#     # Tip 3: Batch processing
#     print('\nTip 3: For batch processing, check compatibility first')
#     print('Use orientation_check() to identify problematic cases')

#     # Tip 4: Error handling
#     print('\nTip 4: Always handle potential errors in orientation checking')
#     try:
#         # This might fail with incompatible shapes
#         incompatible_moving = np.random.rand(100, 50, 25)
#         incompatible_fixed = np.random.rand(32, 32, 32)

#         result = orientation_check(incompatible_moving, incompatible_fixed)
#         print(f'Handled incompatible shapes: {result}')
#     except Exception as e:
#         print(f'Caught expected error: {e}')


# if __name__ == '__main__':
#     print('ASLtk Orientation Checking Examples')
#     print('=' * 50)

#     # Run all examples
#     try:
#         example_basic_orientation_check()
#         example_registration_with_orientation_check()
#         example_detailed_orientation_analysis()
#         example_manual_orientation_workflow()
#         example_advanced_usage()

#         print('\n' + '=' * 50)
#         print('All examples completed successfully!')
#         print('\nNext steps:')
#         print('1. Replace the synthetic data with your actual image files')
#         print(
#             '2. Integrate orientation checking into your registration workflow'
#         )
#         print('3. Use the orientation utilities for quality control')

#     except Exception as e:
#         print(f'\nExample execution failed: {e}')
#         print('This is likely due to missing dependencies or data files.')
#         print('Please ensure you have the required packages installed.')
