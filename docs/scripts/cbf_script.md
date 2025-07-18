# CBF/ATT Mapping Script

This documentation provides an overview of the `cbf.py` script, which is used to calculate the basic CBF (Cerebral Blood Flow) and ATT (Arterial Transit Time) maps from ASL (Arterial Spin Labeling) data.

## Overview

The `cbf.py` script processes ASL data to generate CBF and ATT maps. The script takes ASL raw data, an M0 image, and optional parameters such as a mask image, PLD (Post Labeling Delay) values, and LD (Labeling Duration) values. The output includes the CBF map, normalized CBF map, and ATT map.

## Usage

To run the script, use the following command:

```bash
asltk_cbf pcasl m0 [mask] [out_folder]  --pld PLD [PLD ...] --ld LD [LD ...] [--verbose] [--file_fmt [FILE_FMT]] [-h] [options]
```

## General description

The full description of the script and more details about the necessary/optional parameters can be found by calling `--help` option:

```bash
usage: CBF/ATT Mapping pcasl m0 [mask] [out_folder]  --pld PLD [PLD ...] --ld LD [LD ...] 
[--file_fmt [FILE_FMT]] [--verbose] [-h] 
                       

Python script to calculate the basic CBF and ATT maps from ASL data.

Required parameters:
  pcasl                 ASL raw data obtained from the MRI scanner. This must be the basic PLD ASL MRI acquisition
                        protocol.
  m0                    M0 image reference used to calculate the ASL signal.
  out_folder            The output folder that is the reference to save all the output images in the script. The
                        images selected to be saved are given as tags in the script caller, e.g. the options
                        --cbf_map and --att_map. By default, the TblGM map is placed in the output folder with the
                        name tblgm_map.nii.gz
  --pld PLD [PLD ...]   Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers
  --ld LD [LD ...]      Labeling Duration trend (LD), arranged in a sequence of float numbers.

Optional parameters:
  mask                  Image mask defining the ROI where the calculations must be done. Any pixel value different
                        from zero will be assumed as the ROI area. Outside the mask (value=0) will be ignored. If
                        not provided, the entire image space will be calculated.
  --verbose             Show more details thoughout the processing.
  --file_fmt [FILE_FMT]
                        The file format that will be used to save the output images. It is not allowed image
                        compression (ex: .gz, .zip, etc). Default is nii, but it can be choosen: mha, nrrd.
```