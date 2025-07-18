# T2 Scalar Mapping Script

This documentation provides an overview of the `t2_maps.py` script, which is used to calculate the T2 scalar maps from multi echo (Multi-TE) ASL (Arterial Spin Labeling) data.

## Overview

The `t2_maps.py` script processes ASL data to generate T2 scalar maps. The script takes ASL raw data, an M0 image, and optional parameters such as a mask image, PLD (Post Labeling Delay) values, and LD (Labeling Duration) values. The output includes the T2 maps for each PLD value.

## Usage

To run the script, use the following command:

```bash
asltk_cbf pcasl m0 [mask] [out_folder]  --pld PLD [PLD ...] --ld LD [LD ...] [--verbose] [--file_fmt [FILE_FMT]] [-h] [options]
```

## General description

The full description of the script and more details about the necessary/optional parameters can be found by calling `--help` option:

```bash
usage: T2 Scalar Mapping from ASL Multi-TE ASLData [-h] [--pld PLD [PLD ...]] [--ld LD [LD ...]] [--te TE [TE ...]] [--verbose] [--file_fmt [FILE_FMT]] pcasl m0 [mask] [out_folder]

Python script to calculate the T2 scalar map from the ASL Multi-TE ASLData.

Required parameters:
  pcasl                 ASL raw data obtained from the MRI scanner. This must be the multi-TE ASL MRI acquisition protocol.
  m0                    M0 image reference used to calculate the ASL signal.
  out_folder            The output folder that is the reference to save all the output images in the script.

Optional parameters:
  mask                  Image mask defining the ROI where the calculations must be done. Any pixel value different from zero will be assumed as the ROI area. Outside the mask (value=0) will be
                        ignored. If not provided, the entire image space will be calculated.
  --pld PLD [PLD ...]   Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers. If not passed, the default values will be used.
  --ld LD [LD ...]      Labeling Duration trend (LD), arranged in a sequence of float numbers. If not passed, the default values will be used.
  --te TE [TE ...]      Time of Echos (TE), arranged in a sequence of float numbers. If not passed, the default values will be used.
  --verbose             Show more details thoughout the processing.
  --file_fmt [FILE_FMT]
                        The file format that will be used to save the output images. It is not allowed image compression (ex: .gz, .zip, etc). Default is nii, but it can be choosen: mha, nrrd.
```