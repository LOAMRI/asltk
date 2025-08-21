# Utils Module

## Image Manipulation API


The `utils.image_manipulation` module focuses on the preprocessing, orientation handling, and quality assessment of medical imaging data, particularly for Arterial Spin Labeling (ASL) MRI and similar volumetric neuroimaging datasets. It provides a set of utility functions designed to ensure that image volumes are properly aligned, normalized, and suitable for subsequent analysis or registration.

The code integrates well-established Python libraries for medical imaging, such as `SimpleITK`, `ANTsPy`, and `NumPy`, alongside project-specific utilities from asltk. Its main objectives are:

- Volume Management – Extracting and organizing 3D volumes from multi-dimensional ASL datasets.
- Orientation Analysis and Correction – Checking whether two images are properly aligned in orientation, automatically detecting mismatches, and applying corrective transformations (flips, transpositions, resampling).
- Image Quality Assessment – Computing key statistical measures such as mean intensity and signal-to-noise ratio (SNR) to assist in selecting representative or reference volumes for analysis.
- Reporting and Logging – Generating structured orientation analysis reports, with detailed information on image properties and recommended corrections.

In practice, these tools are expected to be used as an early step in the ASL processing pipeline. By ensuring that image volumes are consistent in orientation and quality, the subsequent stages of image registration, quantification, and biomarker extraction can be performed more reliably.

::: utils.image_manipulation

## Image Statistics API

The `utils.image_statistics` module provides core utility functions for the quantitative analysis of medical images, focusing on extracting essential statistical and structural properties from volumetric data. The implemented functions are designed to support preprocessing, quality assessment, and orientation verification of medical imaging.

The main goals of this code is to provide quantitative analysis such as SNR, mean image intensity, correlations, and many others image properties. 

By combining these functions, the module supports early steps in medical image analysis pipelines, where image quality and structural consistency must be verified before applying more advanced techniques like registration, segmentation, or quantitative biomarker extraction.

::: utils.image_statistics

## IO

The `utils.io` module provides utility functions for loading, saving, and managing medical imaging data, with a focus on Arterial Spin Labeling (ASL) MRI and BIDS-compliant datasets. It builds on libraries such as SimpleITK, NumPy, and dill, allowing users to handle both raw image files and serialized data objects in a reproducible way.

The main features include:

- Loading Images: Supports a wide range of formats (e.g., .nii, .nii.gz, .nrrd, .mha, .tif) and can automatically detect and load files from a BIDS directory structure.
- Saving Images: Exports images in multiple formats, either to a direct path or within a valid BIDS folder hierarchy.
- Managing ASL Data: Provides serialization (save_asl_data) and deserialization (load_asl_data) of ASL datasets using dill for robust object storage.
- BIDS Integration: Ensures compatibility with the BIDS specification, helping organize imaging data systematically across subjects and sessions.

These tools are intended to serve as a foundation for preprocessing and organizing ASL datasets, ensuring that images and related data are stored, retrieved, and shared in a standardized and efficient way before further analysis.

::: utils.io