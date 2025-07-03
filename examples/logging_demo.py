#!/usr/bin/env python3
"""
Example script demonstrating the ASLTK logging functionality.

This script shows how to use the logging system for ASL processing tasks.
"""

import sys
import tempfile
from pathlib import Path

# Add the asltk package to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from asltk import setup_logging, configure_for_scripts, get_logger
from asltk.asldata import ASLData
from asltk.logging_config import log_processing_step, log_data_info


def main():
    """Demonstrate logging features."""

    print('=== ASLTK Logging Demonstration ===\n')

    # Example 1: Basic logging setup
    print('1. Setting up basic logging (INFO level to console):')
    setup_logging(level='INFO', console_output=True)

    logger = get_logger()
    logger.info('ASLTK logging system initialized')
    logger.warning('This is a warning message')
    logger.debug("This debug message won't appear (level is INFO)")

    print('\n2. Setting up verbose logging (DEBUG level):')
    setup_logging(level='DEBUG', console_output=True)
    logger.debug('Now debug messages will appear')

    # Example 2: File logging
    print('\n3. Setting up file logging:')
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / 'asltk_demo.log'
        setup_logging(
            level='INFO', console_output=True, file_output=str(log_file)
        )

        logger.info('This message goes to both console and file')
        log_processing_step('Demonstration', 'showing file logging')

        print(f'   Log file created at: {log_file}')
        print(f'   Log file contents:')
        with open(log_file, 'r') as f:
            for line in f:
                print(f'   {line.strip()}')

    # Example 3: Script configuration
    print('\n4. Script-style configuration (verbose mode):')
    configure_for_scripts(verbose=True)
    logger = get_logger('demo_script')
    logger.info('Script started')
    logger.debug('Verbose mode enabled - debug messages visible')

    # Example 4: ASLData with logging
    print('\n5. ASLData object creation with logging:')
    asl_data = ASLData(ld_values=[1.8], pld_values=[1.8])
    logger.info('ASL data object created successfully')

    # Example 5: Helper functions
    print('\n6. Using logging helper functions:')
    log_data_info('Demo data', (64, 64, 30), '/path/to/demo.nii')
    log_processing_step('Model fitting', 'using Buxton model')

    print('\n=== Demonstration Complete ===')
    print('\nKey features demonstrated:')
    print('- Multiple log levels (DEBUG, INFO, WARNING)')
    print('- Console and file output')
    print('- Script configuration helpers')
    print('- Integration with ASL processing modules')
    print('- Structured logging with context')


if __name__ == '__main__':
    main()
