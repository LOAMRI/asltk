# Generate Subtracted ASL Image Script (Hadamard acquisition)

This documentation provides an overview of the `generate_subtracted_asl_image.py` script, which assists in reconstructing the ASL image already subtracted from control and tagged volumes. The script assumes that the ASL raw data was acquired using MRI imaging protocols based on [Hadamard matrix](https://en.wikipedia.org/wiki/Hadamard_matrix) acquisition.

## Overview

The `generate_subtracted_asl_image.py` script processes ASL data to generate subtracted ASL images. The script takes ASL raw data, optional parameters such as PLD (Post Labeling Delay) values, LD (Labeling Duration) values, TE (Time of Echo) values, and DW (Diffusion Weights) values. The output includes the subtracted ASL image.

## Usage

To run the script, use the following command:

```bash
asltk_hadamard datafolder [--matrix_order MATRIX_ORDER] [--dynamic_vols DYNAMIC_VOLS] [--pld PLD [PLD ...]] [--ld LD [LD ...]] [--output_folder [OUTPUT_FOLDER]] [--mask [MASK]] [--te TE [TE ...]] [--dw DW [DW ...]] [--file_fmt FILE_FMT] [--verbose] [-h]
```

## General description

The full description of the script and more details about the necessary/optional parameters can be found by calling `--help` option:

```bash
usage: Generate Subtracted ASL Image datafolder [--matrix_order MATRIX_ORDER] [--dynamic_vols DYNAMIC_VOLS] 
[--pld PLD [PLD ...]] [--ld LD [LD ...]] [--output_folder [OUTPUT_FOLDER]] [--mask [MASK]] 
[--te TE [TE ...]] [--dw DW [DW ...]] [--file_fmt FILE_FMT] [--verbose] [-h] 

Python script to assist in reconstructing the ASL image already subtract from control and tagged volumes. This
script assumes that the ASL raw data was acquired using a MRI imaging protocols based on Hadamard matrix
acquisition. There are some default values for the PLD and LD, but the user can inform the values used in the MRI
protocol. Please, be aware about the default values and inform the correct values used in the MRI protocol.

Required parameters:
  datafolder            Folder containing the ASL raw data obtained from the MRI scanner. This folder must have the
                        Nifti files converted from the DICOM files. By default the output file name adopted is
                        pcasl.(file_fmt), where file_fmt is the file format informed in the parameter --file_fmt.
                        TIP: One can use other tools such dcm2nii to convert DICOM data to Nifti.
  --matrix_order MATRIX_ORDER
                        Informs the Hadamar matrix size used in the MRI imaging protocol. This must be a positive
                        power-of-two integer (n^2).
  --dynamic_vols DYNAMIC_VOLS
                        Informs the number of dynamic volumes used in the MRI acquisition.
  --pld PLD [PLD ...]   Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers. If not passed,
                        the default values will be used.
  --ld LD [LD ...]      Labeling Duration trend (LD), arranged in a sequence of float numbers. If not passed, the
                        default values will be used.

Optional parameters:
  --output_folder [OUTPUT_FOLDER]
                        The output folder that is the reference to save the output image. By default, the output
                        image will be saved in the same folder as the input data. If informed, the output image
                        will be saved in the folder informed.
  --mask [MASK]         Image mask defining the ROI where the calculations must be done. Any pixel value different
                        from zero will be assumed as the ROI area. Outside the mask (value=0) will be ignored. If
                        not provided, the entire image space will be calculated.
  --te TE [TE ...]      Time of Echos (TE), arranged in a sequence of float numbers. This is only required for
                        multi-TE ASL data. This sequence of values must be in accordance with the number of volumes
                        acquired in the MRI protocol.
  --dw DW [DW ...]      Diffusion weights (DW), arranged in a sequence of float numbers. This is only required for
                        multi-DW ASL data. This sequence of values must be in accordance with the number of volumes
                        acquired in the MRI protocol.
  --file_fmt FILE_FMT   Define the file format to load the ASL data in the datafolder parameter and also be used
                        for saving the output image. The default is Nifti format (nii). File formats allowed: nii,
                        mha, nrrd. TIP: This file format depends on the output of the DICOM converter tool used,
                        then it is important to check the output format of the tool used to convert the DICOM files
                        to Nifti files.
  --verbose             Show more details thoughout the processing.
```