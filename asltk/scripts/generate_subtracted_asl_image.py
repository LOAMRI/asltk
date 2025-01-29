import argparse
import os
from functools import *
from glob import glob

import numpy as np
import SimpleITK as sitk
from rich import print
from rich.progress import track
from scipy.linalg import hadamard

from asltk.utils import load_image, save_image

parser = argparse.ArgumentParser(
    prog='Generate Subtracted ASL Image',
    description='Python script to assist in reconstructing the ASL image already subtract from control and tagged volumes. This script assumes that the ASL raw data was acquired using a MRI imaging protocols based on Hadamard matrix acquisition. There are some default values for the PLD and LD, but the user can inform the values used in the MRI protocol. Please, be aware about the default values and inform the correct values used in the MRI protocol.',
)
parser._action_groups.pop()
required = parser.add_argument_group(title='Required parameters')
optional = parser.add_argument_group(title='Optional parameters')


required.add_argument(
    'datafolder',
    type=str,
    help='Folder containing the ASL raw data obtained from the MRI scanner. This folder must have the Nifti files converted from the DICOM files.  By default the output file name adopted is pcasl.(file_fmt), where file_fmt is the file format informed in the parameter --file_fmt. TIP: One can use other tools such dcm2nii to convert DICOM data to Nifti.',
)
required.add_argument(
    '--matrix_order',
    type=int,
    required=False,
    default=8,
    help='Informs the Hadamar matrix size used in the MRI imaging protocol. This must be a positive power-of-two integer (n^2).',
)
required.add_argument(
    '--dynamic_vols',
    type=int,
    required=False,
    default=2,
    help='Informs the number of dynamic volumes used in the MRI acquisition.',
)
required.add_argument(
    '--pld',
    type=str,
    nargs='+',
    required=False,
    default=[170.0, 270.0, 370.0, 520.0, 670.0, 1070.0, 1870.0],
    help='Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers. If not passed, the default values will be used.',
)
required.add_argument(
    '--ld',
    type=str,
    nargs='+',
    required=False,
    default=[100.0, 100.0, 150.0, 150.0, 400.0, 800.0, 1800.0],
    help='Labeling Duration trend (LD), arranged in a sequence of float numbers. If not passed, the default values will be used.',
)
optional.add_argument(
    '--output_folder',
    type=str,
    nargs='?',
    default='',
    help='The output folder that is the reference to save the output image. By default, the output image will be saved in the same folder as the input data. If informed, the output image will be saved in the folder informed.',
)
optional.add_argument(
    '--mask',
    type=str,
    nargs='?',
    default='',
    help='Image mask defining the ROI where the calculations must be done. Any pixel value different from zero will be assumed as the ROI area. Outside the mask (value=0) will be ignored. If not provided, the entire image space will be calculated.',
)
optional.add_argument(
    '--te',
    type=float,
    nargs='+',
    default=[13.56, 67.82, 122.08, 176.33, 230.59, 284.84, 339.1, 393.36],
    help='Time of Echos (TE), arranged in a sequence of float numbers. This is only required for multi-TE ASL data. This sequence of values must be in accordance with the number of volumes acquired in the MRI protocol.',
)
optional.add_argument(
    '--dw',
    type=float,
    nargs='+',
    default=[0, 50, 100, 250],
    help='Diffusion weights (DW), arranged in a sequence of float numbers. This is only required for multi-DW ASL data. This sequence of values must be in accordance with the number of volumes acquired in the MRI protocol.',
)
optional.add_argument(
    '--file_fmt',
    type=str,
    required=False,
    default='nii',
    help='Define the file format to load the ASL data in the datafolder parameter and also be used for saving the output image. The default is Nifti format (nii). File formats allowed: nii, mha, nrrd. TIP: This file format depends on the output of the DICOM converter tool used, then it is important to check the output format of the tool used to convert the DICOM files to Nifti files.',
)
optional.add_argument(
    '--verbose',
    action='store_true',
    help='Show more details thoughout the processing.',
)

args = parser.parse_args()

# Check input parameters
def checkUpParameters():
    is_ok = True

    # Check if the data folder exists
    if not os.path.exists(args.datafolder):
        print(
            f"[red]Error: The data folder '{args.datafolder}' does not exist.[/red]"
        )
        is_ok = False

    # Check if the matrix order is a power-of-two integer
    if not (args.matrix_order & (args.matrix_order - 1) == 0):
        print(
            f'[red]Error: The matrix order must be a power-of-two integer.[/red]'
        )
        is_ok = False

    # Check if the dynamic volumes is a positive integer
    if args.dynamic_vols <= 0:
        print(
            f'[red]Error: The dynamic volumes must be a positive integer.[/red]'
        )
        is_ok = False

    # Check if file format is allowed
    if args.file_fmt not in ['nii', 'mha', 'nrrd']:
        print(
            f"[red]Error: The file format '{args.file_fmt}' is not allowed.[/red]"
        )
        is_ok = False

    # Check if mask file exists, if informed
    if args.mask and not os.path.exists(args.mask):
        print(f"[red]Error: The mask file '{args.mask}' does not exist.[/red]")
        is_ok = False

    # If passed, check if the output folder exists
    if args.output_folder and not os.path.exists(args.output_folder):
        print(
            f"[red]Error: The output folder '{args.output_folder}' does not exist.[/red]"
        )
        is_ok = False

    return is_ok


def load_asl_data(datafolder, file_fmt):
    asl_files = [
        os.path.join(datafolder, f)
        for f in os.listdir(datafolder)
        if f.endswith(f'.{file_fmt}')
    ]
    asl_images = [load_image(f) for f in asl_files]
    return asl_images


# Start the script execution
if not checkUpParameters():
    raise RuntimeError(
        'One or more arguments are not well defined. Please, revise the script call.'
    )

if args.verbose:
    print(' --- Script Input Data ---')
    print('Required parameters:')
    print('Input data folder path: ' + args.datafolder)
    print('Matrix order: ' + str(args.matrix_order))
    print('Dynamic volumes: ' + str(args.dynamic_vols))
    print('PLD: ' + str(args.pld))
    print('LD: ' + str(args.ld))
    print('----------------')
    print('Optional parameters:')
    if args.output_folder:
        print('Output folder path: ' + str(args.output_folder))
    else:
        print('Output folder path: assuming the input folder path')
    if args.mask:
        print('Mask image file path: ' + str(args.mask))
    else:
        print('Mask image file path: not applying brain mask')
    print('TE values: ' + str(args.te))
    print('DW values: ' + str(args.dw))
    print('Input/Output file format: ' + args.file_fmt)


# Load all the Nifti files in the folder
asl_images = load_asl_data(args.datafolder, args.file_fmt)
# If mask is passed, check if shape is the same as the ASL images
if args.mask:
    mask_image = load_image(args.mask)
    if mask_image.shape != asl_images[0].shape[1:]:
        raise ValueError(
            'The mask image shape does not match the ASL images shape.'
        )
    asl_images = [asl_image * (mask_image > 0) for asl_image in asl_images]

# Load the input ASL data (PLD, LD, and TE, DWI if available)
try:
    pld = [float(s) for s in args.pld]
    ld = [float(s) for s in args.ld]
    te = [float(s) for s in args.te]
    dw = [float(s) for s in args.dw]
except:
    pld = [float(s) for s in str(args.pld[0]).split()]
    ld = [float(s) for s in str(args.ld[0]).split()]
    te = [float(s) for s in str(args.te[0]).split()]
    dw = [float(s) for s in str(args.dw[0]).split()]

# Process the data using Hadamar's product
boli_cnt = args.matrix_order - 1  # number of sub-bolis
nTe = len(te)  # number of echo times
# TODO Adapt the script for DW images as well

h_mtrx = hadamard(args.matrix_order)
subtr_mtrx = -h_mtrx[:, 1 : args.matrix_order]
fn = [
    f
    for f in glob(os.path.join(args.datafolder, f'*.{args.file_fmt}'))
    if 'pcasl' not in os.path.basename(f)
]
head = sitk.ReadImage(fn[0])

for cnt in track(range(nTe), description='Processing volumes...'):
    dat = sitk.ReadImage(fn[cnt])
    dat_array = sitk.GetArrayFromImage(dat)
    dims = dat_array.shape

    if cnt == 0:
        dat_final = np.zeros(
            (nTe, boli_cnt, dims[1], dims[2], dims[3]), dtype=np.float64
        )

    dat_aux = np.zeros(
        (
            args.dynamic_vols,
            dims[0] // args.dynamic_vols,
            dims[1],
            dims[2],
            dims[3],
        ),
        dtype=np.float64,
    )

    dat_aux[0, ...] = dat_array[::2, ...]
    dat_aux[1, ...] = dat_array[1::2, ...]

    unsubtr_data_mean = np.mean(dat_aux, axis=0)

    Subtr_phases = np.zeros(unsubtr_data_mean[1:, ...].shape, dtype=np.float64)

    for bolus in range(boli_cnt):
        vector = subtr_mtrx[:, bolus]
        for line in range(args.matrix_order):
            Subtr_phases[boli_cnt - bolus - 1, ...] += (
                vector[line] * unsubtr_data_mean[line, ...]
            )

    dat_final[cnt, ...] = Subtr_phases * float(
        dat.GetMetaData('scl_slope')
    ) + float(dat.GetMetaData('scl_inter'))

# Create the final NIfTI image
dat_final_sitk = sitk.GetImageFromArray(dat_final)
# TODO Set the image information from the input ASL header
# dat_final_sitk.SetSpacing(head.GetSpacing())
# dat_final_sitk.SetDirection(head.GetDirection())
# dat_final_sitk.SetOrigin(head.GetOrigin())
# sitk.WriteImage(dat_final_sitk, os.path.join(args.datafolder, "pcasl.nii.gz"))

# Save the output images in the output folder (or other path passed by the user)
output_path = os.path.join(
    args.output_folder if args.output_folder else args.datafolder,
    f'pcasl.{args.file_fmt}',
)
if args.verbose:
    print('Saving file at: ' + output_path)
save_image(dat_final, output_path)

if args.verbose:
    print('Execution: ' + parser.prog + ' finished successfully!')
