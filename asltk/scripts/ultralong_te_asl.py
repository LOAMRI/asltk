import argparse
import os
from functools import *

import numpy as np
from rich import print

from asltk.asldata import ASLData
from asltk.logging_config import (
    configure_for_scripts,
    get_logger,
    log_processing_step,
)
from asltk.reconstruction import UltraLongTE_ASLMapping
from asltk.utils.io import ImageIO

parser = argparse.ArgumentParser(
    prog='UltraLong-TE ASL Mapping',
    description='Python script to calculate the UltraLong-TE ASL map for the T1 relaxation exchange between CSF and Gray Matter (GM).',
)
parser._action_groups.pop()
required = parser.add_argument_group(title='Required parameters')
optional = parser.add_argument_group(title='Optional parameters')

required.add_argument(
    'pcasl',
    type=str,
    help='ASL raw data obtained from the MRI scanner. This must be the ultralong-TE ASL MRI acquisition protocol.',
)
required.add_argument(
    'm0', type=str, help='M0 image reference used to calculate the ASL signal.'
)
optional.add_argument(
    'mask',
    type=str,
    nargs='?',
    default='',
    help='Image mask defining the ROI where the calculations must be done. Any pixel value different from zero will be assumed as the ROI area. Outside the mask (value=0) will be ignored. If not provided, the entire image space will be calculated.',
)
required.add_argument(
    'out_folder',
    type=str,
    nargs='?',
    default=os.path.expanduser('~'),
    help='The output folder that is the reference to save all the output images in the script. The images selected to be saved are given as tags in the script caller, e.g. the options --cbf_map and --att_map. By default, the TblGM map is placed in the output folder with the name tblgm_map.nii.gz',
)
optional.add_argument(
    '--cbf',
    type=str,
    nargs='?',
    required=False,
    help='The CBF map that is provided to skip this step in the MultiTE-ASL calculation. If CBF is not provided, than a CBF map is calculated at the runtime. Important: The CBF passed here is with the original voxel scale, i.e. without voxel normalization.',
)
optional.add_argument(
    '--att',
    type=str,
    nargs='?',
    required=False,
    help='The ATT map that is provided to skip this step in the MultiTE-ASL calculation. If ATT is not provided, than a ATT map is calculated at the runtime.',
)
optional.add_argument(
    '--pld',
    type=str,
    nargs='+',
    required=False,
    default=[500, 1000, 1500, 2000, 2500, 4000],
    help='Posts Labeling Delay (PLD) trend, arranged in a sequence of float numbers. If not passed, the default values will be used.',
)
optional.add_argument(
    '--ld',
    type=str,
    nargs='+',
    required=False,
    default=[1000, 1000, 1500, 2000, 3000, 3000],
    help='Labeling Duration trend (LD), arranged in a sequence of float numbers. If not passed, the default values will be used.',
)
optional.add_argument(
    '--te',
    type=str,
    nargs='+',
    required=False,
    default=[35, 315, 595, 875, 1155, 1435, 1715, 1995, 2275, 2555],
    help='Time of Echos (TE), arranged in a sequence of float numbers. If not passed, the default values will be used.',
)
optional.add_argument(
    '--verbose',
    action='store_true',
    help='Show more details thoughout the processing.',
)
optional.add_argument(
    '--file_fmt',
    type=str,
    nargs='?',
    default='nii',
    help='The file format that will be used to save the output images. It is not allowed image compression (ex: .gz, .zip, etc). Default is nii, but it can be choosen: mha, nrrd.',
)
optional.add_argument(
    '--average_m0',
    action='store_true',
    default=False,
    help='Whether to average the M0 images across the time series. Default is False.',
)

args = parser.parse_args()

# Configure logging based on verbose flag
configure_for_scripts(verbose=args.verbose)
logger = get_logger('te_asl_script')

# Script check-up parameters
def checkUpParameters():
    is_ok = True
    # Check output folder exist
    if not (os.path.isdir(args.out_folder)):
        error_msg = f'Output folder path does not exist (path: {args.out_folder}). Please create the folder before executing the script.'
        logger.error(error_msg)
        print(error_msg)
        is_ok = False

    # Check ASL image exist
    if not (os.path.isfile(args.pcasl)):
        error_msg = f'ASL input file does not exist (file path: {args.pcasl}). Please check the input file before executing the script.'
        logger.error(error_msg)
        print(error_msg)
        is_ok = False

    # Check M0  image exist
    if not (os.path.isfile(args.m0)):
        error_msg = f'M0 input file does not exist (file path: {args.m0}). Please check the input file before executing the script.'
        logger.error(error_msg)
        print(error_msg)
        is_ok = False

    if args.file_fmt not in ('nii', 'mha', 'nrrd'):
        error_msg = f'File format is not allowed or not available. The select type is {args.file_fmt}, but options are: nii, mha or nrrd'
        logger.error(error_msg)
        print(error_msg)
        is_ok = False

    return is_ok


asl_img = ImageIO(args.pcasl)
m0_img = ImageIO(args.m0)

average_m0 = args.average_m0

mask_img = ImageIO(image_array=np.ones(asl_img.get_as_numpy().shape[-3:]))
if args.mask != '':
    mask_img = ImageIO(args.mask)


cbf_map = None
if args.cbf is not None:
    cbf_map = ImageIO(args.cbf)

att_map = None
if args.att is not None:
    att_map = ImageIO(args.att)


try:
    te = [float(s) for s in args.te]
    pld = [float(s) for s in args.pld]
    ld = [float(s) for s in args.ld]
except:
    te = [float(s) for s in str(args.te[0]).split()]
    pld = [float(s) for s in str(args.pld[0]).split()]
    ld = [float(s) for s in str(args.ld[0]).split()]

if not checkUpParameters():
    raise RuntimeError(
        'One or more arguments are not well defined. Please, revise the script call.'
    )


# Step 2: Show the input information to assist manual conference
logger.info('UltraLong-TE ASL processing started')
if args.verbose:
    print(' --- Script Input Data ---')
    print('ASL file path: ' + args.pcasl)
    print('ASL image dimension: ' + str(asl_img.get_as_numpy().shape))
    print('Mask file path: ' + args.mask)
    print('Mask image dimension: ' + str(mask_img.get_as_numpy().shape))
    print('M0 file path: ' + args.m0)
    print('M0 image dimension: ' + str(m0_img.get_as_numpy().shape))
    print('PLD: ' + str(pld))
    print('LD: ' + str(ld))
    print('TE: ' + str(te))
    if args.cbf != '':
        print('(optional) CBF map: ' + str(args.cbf))
    if args.att != '':
        print('(optional) ATT map: ' + str(args.att))
    print('Output file format: ' + str(args.file_fmt))

# Log input parameters
logger.info(f'Input parameters - PLD: {pld}, LD: {ld}, TE: {te}')
logger.info(f'Output format: {args.file_fmt}')

log_processing_step(
    'Creating ASLData object', f'Multi-TE with {len(te)} echo times'
)
data = ASLData(
    pcasl=args.pcasl,
    m0=args.m0,
    ld_values=ld,
    pld_values=pld,
    te_values=te,
    average_m0=average_m0,
)

log_processing_step('Initializing UltraLong-TE ASL mapper')
recon = UltraLongTE_ASLMapping(data)
recon.set_brain_mask(mask_img)

if isinstance(cbf_map, ImageIO) and isinstance(att_map, ImageIO):
    logger.info('Setting optional CBF and ATT maps')
    recon.set_cbf_map(cbf_map)
    recon.set_att_map(att_map)

log_processing_step(
    'Generating UltraLong-TE ASL maps', 'this may take several minutes'
)
maps = recon.create_map()
logger.info('UltraLong-TE ASL map generation completed successfully')

log_processing_step('Saving output maps')
save_path = args.out_folder + os.path.sep + 'cbf_map.' + args.file_fmt
if args.verbose and cbf_map is not None:
    print('Saving CBF map - Path: ' + save_path)
logger.info(f'Saving CBF map to: {save_path}')
maps['cbf'].save_image(save_path)

save_path = (
    args.out_folder + os.path.sep + 'cbf_map_normalized.' + args.file_fmt
)
if args.verbose and cbf_map is not None:
    print('Saving normalized CBF map - Path: ' + save_path)
logger.info(f'Saving normalized CBF map to: {save_path}')
maps['cbf_norm'].save_image(save_path)

save_path = args.out_folder + os.path.sep + 'att_map.' + args.file_fmt
if args.verbose and att_map is not None:
    print('Saving ATT map - Path: ' + save_path)
logger.info(f'Saving ATT map to: {save_path}')
maps['att'].save_image(save_path)

save_path = args.out_folder + os.path.sep + 'mte_t1csfgm_map.' + args.file_fmt
if args.verbose:
    print('Saving ultralongTE-ASL T1csfGM map - Path: ' + save_path)
logger.info(f'Saving T1csfGM map to: {save_path}')
maps['t1csfgm'].save_image(save_path)

if args.verbose:
    print('Execution: ' + parser.prog + ' finished successfully!')
logger.info('UltraLong-TE ASL processing completed successfully')


def main():
    """
    Entry point function for the ultralong-TE Scalar ASL mapping command-line tool.

    This function is called when the `asltk_ultralong_te_asl` command is run.
    All script logic is already defined at the module level.
    """
    # Script logic is already defined at the module level
    pass


if __name__ == '__main__':
    main()
