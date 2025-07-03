# ASLTK Logging System

The ASLTK library includes a comprehensive logging system that provides detailed runtime information about ASL data processing operations. This system helps with debugging, monitoring, and understanding the behavior of ASL processing workflows.

## Features

- **Multiple log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Flexible output**: Console output, file output, or both
- **Script integration**: Easy configuration for command-line scripts
- **Modular logging**: Different modules can have their own loggers
- **Performance monitoring**: Timing and statistics for processing steps

## Quick Start

### Basic Setup

```python
from asltk import setup_logging, get_logger

# Basic setup with INFO level to console
setup_logging()

# Get a logger for your module
logger = get_logger()
logger.info("Starting ASL processing")
```

### Script Configuration

For command-line scripts, use the convenient script configuration:

```python
from asltk import configure_for_scripts

# Configure logging based on verbose flag
configure_for_scripts(verbose=args.verbose)
```

### File Logging

To save logs to a file:

```python
from asltk import setup_logging

# Log to both console and file
setup_logging(
    level='INFO',
    console_output=True,
    file_output='asl_processing.log'
)
```

## Configuration Options

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about processing steps
- **WARNING**: Warnings about potential issues
- **ERROR**: Error conditions that may affect results
- **CRITICAL**: Critical errors that prevent processing

### Setup Function

```python
setup_logging(
    level='INFO',              # Logging level
    console_output=True,       # Output to console
    file_output=None,          # Optional log file path
    log_format=None,           # Custom log format
    date_format=None           # Custom date format
)
```

### Script Configuration

```python
configure_for_scripts(
    verbose=False,             # Enable DEBUG level if True
    log_file=None              # Optional log file
)
```

## Usage Examples

### Data Loading and Validation

```python
from asltk import ASLData

# Logging is automatically enabled for data loading
asl_data = ASLData(
    pcasl='data.nii.gz',
    m0='m0.nii.gz',
    ld_values=[1.8],
    pld_values=[1.8]
)
# Logs: Loading paths, image dimensions, parameter validation
```

### Processing Steps

```python
from asltk.reconstruction import CBFMapping

# Create CBF mapper
cbf_mapper = CBFMapping(asl_data)
cbf_mapper.set_brain_mask(brain_mask)

# Generate maps (with detailed logging)
results = cbf_mapper.create_map(cores=4)
# Logs: Processing parameters, progress, completion statistics
```

### Registration and Motion Correction

```python
from asltk.registration import head_movement_correction

# Apply motion correction with logging
corrected_data, transforms = head_movement_correction(
    asl_data, 
    ref_vol=0,
    verbose=True
)
# Logs: Volume processing, registration results, warnings
```

## Helper Functions

The logging system includes convenient helper functions:

```python
from asltk.logging_config import (
    log_processing_step,
    log_data_info,
    log_warning_with_context,
    log_error_with_traceback
)

# Log processing steps
log_processing_step("Model fitting", "using Buxton model")

# Log data information
log_data_info("ASL image", image.shape, "/path/to/image.nii")

# Log warnings with context
log_warning_with_context("Low SNR detected", "slice 15")

# Log errors with traceback
log_error_with_traceback("Processing failed")
```

## Integration with Existing Scripts

The logging system is designed to work alongside existing verbose flags in ASLTK scripts:

```bash
# Enable verbose logging in scripts
python -m asltk.scripts.te_asl --verbose input.nii m0.nii output/ --te 13 20 50
```

This will:
- Enable DEBUG level logging
- Show detailed processing information
- Maintain compatibility with existing print statements

## Customization

### Custom Log Formats

```python
setup_logging(
    log_format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    date_format='%Y-%m-%d %H:%M:%S'
)
```

### Module-Specific Loggers

```python
from asltk import get_logger

# Get loggers for specific modules
asl_logger = get_logger('asldata')
cbf_logger = get_logger('cbf_mapping')
reg_logger = get_logger('registration')
```

### Integration with External Systems

For integration with external logging systems:

```python
import logging
from asltk import get_logger

# Get the ASLTK root logger
asltk_logger = get_logger()

# Add your own handlers
handler = logging.StreamHandler(your_stream)
asltk_logger.addHandler(handler)

# Or integrate with existing loggers
asltk_logger.parent = your_logger
asltk_logger.propagate = True
```

## Best Practices

1. **Use appropriate log levels**: Reserve DEBUG for detailed tracing, INFO for general progress, WARNING for potential issues
2. **Include context**: Use helper functions to provide context about what's being processed
3. **Configure early**: Set up logging at the beginning of your scripts or applications
4. **Use file logging for long processes**: Save logs to files for batch processing or long-running operations
5. **Monitor warnings**: Pay attention to WARNING messages as they may indicate data quality issues

## Example: Complete Processing Script

```python
import argparse
from asltk import configure_for_scripts, get_logger
from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log-file', help='Save logs to file')
    parser.add_argument('input_file')
    parser.add_argument('m0_file')
    args = parser.parse_args()
    
    # Configure logging
    configure_for_scripts(
        verbose=args.verbose,
        log_file=args.log_file
    )
    
    logger = get_logger('my_script')
    logger.info("Starting ASL processing")
    
    try:
        # Load data (automatically logged)
        asl_data = ASLData(pcasl=args.input_file, m0=args.m0_file)
        
        # Process data (automatically logged)
        cbf_mapper = CBFMapping(asl_data)
        results = cbf_mapper.create_map()
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
```

This will provide comprehensive logging throughout the ASL processing workflow, making it easier to monitor progress, debug issues, and understand the processing behavior.