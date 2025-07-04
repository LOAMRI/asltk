BIDS_IMAGE_FORMATS = ('.nii', '.nii.gz')
AVAILABLE_IMAGE_FORMATS = ('.nii', '.nii.gz', '.mha', '.nrrd')

# Import logging functionality for easy access
from .logging_config import configure_for_scripts, get_logger, setup_logging

# Set up default logging (INFO level, console only)
setup_logging()

__all__ = ['setup_logging', 'get_logger', 'configure_for_scripts']
