import argparse
import os
from functools import *

import numpy as np
import SimpleITK as sitk
from rich import print
from rich.progress import track
from scipy.optimize import curve_fit

from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping
from asltk.utils import load_image, save_image

parser = argparse.ArgumentParser(
    description='Python script to calculate the basic CBF and ATT maps from ASL data.'
)

parser.add_argument(
    'pcasl',
    type=str,
    help='ASL raw data obtained from the MRI scanner. This must be the basic PLD ASL MRI acquisition protocol.',
)
parser.add_argument('m0', type=str, help='M0 image in Nifti format.')
parser.add_argument(
    'mask',
    type=str,
    nargs='?',
    default='',
    help='Image mask defining the ROI where the calculations must be done. Any pixel value different from zero will be assumed as the ROI area. Outside the mask (value=0) will be ignored.',
)
parser.add_argument(
    'out_folder',
    type=str,
    nargs='?',
    default=os.path.expanduser('~'),
    help='The output folder that is the reference to save all the output images in the script. The images selected to be saved are given as tags in the script caller, e.g. the options --cbf_map and --att_map. By default, the TblGM map is placed in the output folder with the name tblgm_map.nii.gz',
)
parser.add_argument(
    '--pld',
    type=str,
    nargs='+',
    required=True,
    help='Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers',
)
parser.add_argument(
    '--ld',
    type=str,
    nargs='+',
    required=True,
    help='Labeling Duration trend (LD), arranged in a sequence of float numbers.',
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help='Show more details thoughout the processing.',
)

args = parser.parse_args()

# Script check-up parameters
def checkUpParameters():
    is_ok = True
    # Check output folder exist
    if not (os.path.isdir(args.out_folder)):
        print(
            f'Output folder path does not exist (path: {args.out_folder}). Please create the folder before executing the script.'
        )
        is_ok = False

    # Check ASL image exist
    if not (os.path.isfile(args.pcasl)):
        print(
            f'ASL input file does not exist (file path: {args.asl}). Please check the input file before executing the script.'
        )
        is_ok = False

    # Check M0  image exist
    if not (os.path.isfile(args.m0)):
        print(
            f'M0 input file does not exist (file path: {args.m0}). Please check the input file before executing the script.'
        )
        is_ok = False

    return is_ok


asl_img = load_image(args.pcasl)
m0_img = load_image(args.m0)

mask_img = np.ones(asl_img[0, 0, :, :, :].shape)
if args.mask != '':
    mask_img = load_image(args.mask)


try:
    pld = [float(s) for s in args.pld]
    ld = [float(s) for s in args.ld]
except:
    pld = [float(s) for s in str(args.pld[0]).split()]
    ld = [float(s) for s in str(args.ld[0]).split()]

if not checkUpParameters():
    raise RuntimeError(
        'One or more arguments are not well defined. Please, revise the script call.'
    )


# Step 2: Show the input information to assist manual conference
if args.verbose:
    print(' --- Script Input Data ---')
    print('ASL file path: ' + args.pcasl)
    print('ASL image dimension: ' + str(asl_img.shape))
    print('Mask file path: ' + args.mask)
    print('Mask image dimension: ' + str(mask_img.shape))
    print('M0 file path: ' + args.m0)
    print('M0 image dimension: ' + str(m0_img.shape))
    print('PLD: ' + str(pld))
    print('LD: ' + str(ld))

data = ASLData(pcasl=args.pcasl, m0=args.m0, ld_values=ld, pld_values=pld)
recon = CBFMapping(data)
recon.set_brain_mask(mask_img)
maps = recon.create_map()


save_path = args.out_folder + os.path.sep + 'cbf_map.nii.gz'
if args.verbose:
    print('Saving CBF map - Path: ' + save_path)
save_image(maps['cbf'], save_path)

save_path = args.out_folder + os.path.sep + 'cbf_map_normalized.nii.gz'
if args.verbose:
    print('Saving normalized CBF map - Path: ' + save_path)
save_image(maps['cbf_norm'], save_path)

save_path = args.out_folder + os.path.sep + 'att_map.nii.gz'
if args.verbose:
    print('Saving ATT map - Path: ' + save_path)
save_image(maps['att'], save_path)

if args.verbose:
    print('Execution: ' + parser.prog + ' finished successfully!')
