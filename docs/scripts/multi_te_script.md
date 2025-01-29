# Multi-TE ASL Mapping Script

This documentation provides an overview of the `te_asl.py` script, which is used to calculate the Multi-TE ASL map for the T1 relaxation exchange between blood and Gray Matter (GM).

## Overview

The `te_asl.py` script processes ASL data to generate the T1 relaxation exchange between blood and Gray Matter (T1blGM) map and, also as an additional information, it can export the CBF (Cerebral Blood Flow), normalized CBF and ATT (Arterial Transit Time) maps. 

The script takes ASL raw data, an M0 image, and optional parameters such as a mask image, PLD (Post Labeling Delay) values, LD (Labeling Duration) values, and TE (Time of Echo) values. The output includes the CBF map, normalized CBF map, ATT map, and T1blGM map.

## Usage

To run the script, use the following command:

```bash
python -m asltk.scripts.te_asl pcasl m0 [mask] [out_folder] [--cbf [CBF]] [--att [ATT]] --pld PLD PLD ...] --ld LD [LD ...] --te TE [TE ...] [--file_fmt [FILE_FMT]] [--verbose] [-h] 
```

## General description

The full description of the script and more details about the necessary/optional parameters can be found by calling `--help` option:

```bash
usage: Multi-TE ASL Mapping pcasl m0 [mask] [out_folder] 
[--cbf [CBF]] [--att [ATT]] --pld PLD [PLD ...] 
--ld LD [LD ...] --te TE [TE ...] [--verbose] [--file_fmt [FILE_FMT]] [-h] 

Python script to calculate the Multi-TE ASL map for the T1 relaxation exchange between blood and Gray Matter (GM).

Required parameters:
  pcasl                 ASL raw data obtained from the MRI scanner. This must be the multi-TE ASL MRI acquisition
                        protocol.
  m0                    M0 image reference used to calculate the ASL signal.
  out_folder            The output folder that is the reference to save all the output images in the script. The
                        images selected to be saved are given as tags in the script caller, e.g. the options
                        --cbf_map and --att_map. By default, the TblGM map is placed in the output folder with the
                        name tblgm_map.nii.gz
  --pld PLD [PLD ...]   Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers
  --ld LD [LD ...]      Labeling Duration trend (LD), arranged in a sequence of float numbers.
  --te TE [TE ...]      Time of Echos (TE), arranged in a sequence of float numbers.

Optional parameters:
  mask                  Image mask defining the ROI where the calculations must be done. Any pixel value different
                        from zero will be assumed as the ROI area. Outside the mask (value=0) will be ignored. If
                        not provided, the entire image space will be calculated.
  --cbf [CBF]           The CBF map that is provided to skip this step in the MultiTE-ASL calculation. If CBF is
                        not provided, than a CBF map is calculated at the runtime. Important: The CBF passed here
                        is with the original voxel scale, i.e. without voxel normalization.
  --att [ATT]           The ATT map that is provided to skip this step in the MultiTE-ASL calculation. If ATT is
                        not provided, than a ATT map is calculated at the runtime.
  --verbose             Show more details thoughout the processing.
  --file_fmt [FILE_FMT]
                        The file format that will be used to save the output images. It is not allowed image
                        compression (ex: .gz, .zip, etc). Default is nii, but it can be choosen: mha, nrrd.
```