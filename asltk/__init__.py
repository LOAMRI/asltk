import os

BIDS_IMAGE_FORMATS = ('.nii', '.nii.gz')
AVAILABLE_IMAGE_FORMATS = ('.nii', '.nii.gz', '.mha', '.nrrd')

PARCELLATION_REPORT_PATH = os.path.join(os.path.expanduser('~'), 'asltk',os.path.sep,'parcellation_reports')
