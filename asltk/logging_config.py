"""
Centralized logging configuration for the ASLTK library.

This module provides a unified logging system with configurable levels,
output formats, and destinations (console and file).
"""

import logging
import logging.handlers
import os
from typing import Optional, Union

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Package logger name
PACKAGE_LOGGER_NAME = 'asltk'


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the ASLTK package.

    Args:
        name: Logger name suffix. If None, returns the root package logger.

    Returns:
        Logger instance configured for ASLTK.
    """
    if name is None:
        logger_name = PACKAGE_LOGGER_NAME
    else:
        logger_name = f'{PACKAGE_LOGGER_NAME}.{name}'

    return logging.getLogger(logger_name)


def setup_logging(
    level: Union[str, int] = logging.INFO,
    console_output: bool = True,
    file_output: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Configure logging for the ASLTK package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL or numeric value)
        console_output: Whether to output logs to console
        file_output: Path to log file. If None, no file logging is configured
        log_format: Custom log format string
        date_format: Custom date format string

    Examples:
        Basic setup with INFO level to console:
        >>> setup_logging()

        Setup with DEBUG level to both console and file:
        >>> setup_logging(level='DEBUG', file_output='asltk.log')

        Setup with custom format:
        >>> setup_logging(log_format='%(levelname)s: %(message)s')
    """
    # Convert string level to numeric if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Use default formats if not provided
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Get the root logger for the package
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if requested
    if file_output:
        # Ensure directory exists
        log_dir = os.path.dirname(file_output)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(file_output, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate messages (but allow for testing)
    if os.environ.get('PYTEST_CURRENT_TEST'):
        logger.propagate = True
    else:
        logger.propagate = False


def configure_for_scripts(
    verbose: bool = False, log_file: Optional[str] = None
) -> None:
    """
    Convenience function to configure logging for command-line scripts.

    Args:
        verbose: If True, sets DEBUG level; otherwise INFO level
        log_file: Optional log file path

    Examples:
        Configure for verbose script execution:
        >>> configure_for_scripts(verbose=True)

        Configure with log file:
        >>> configure_for_scripts(log_file='processing.log')
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level, console_output=True, file_output=log_file)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log a function call with its parameters.

    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger()
    params = ', '.join(f'{k}={v}' for k, v in kwargs.items())
    logger.debug(f'Calling {func_name}({params})')


def log_processing_step(step_name: str, details: Optional[str] = None) -> None:
    """
    Log a processing step at INFO level.

    Args:
        step_name: Name of the processing step
        details: Optional additional details
    """
    logger = get_logger()
    message = f'Processing step: {step_name}'
    if details:
        message += f' - {details}'
    logger.info(message)


def log_data_info(
    data_type: str, shape: tuple, path: Optional[str] = None
) -> None:
    """
    Log information about loaded data.

    Args:
        data_type: Type of data (e.g., 'ASL image', 'M0 image', 'mask')
        shape: Shape of the data array
        path: Optional file path
    """
    logger = get_logger()
    message = f'Loaded {data_type}: shape={shape}'
    if path:
        message += f', path={path}'
    logger.info(message)


def log_warning_with_context(
    message: str, context: Optional[str] = None
) -> None:
    """
    Log a warning with optional context information.

    Args:
        message: Warning message
        context: Optional context information
    """
    logger = get_logger()
    full_message = message
    if context:
        full_message += f' (Context: {context})'
    logger.warning(full_message)


def log_error_with_traceback(message: str, exc_info: bool = True) -> None:
    """
    Log an error with traceback information.

    Args:
        message: Error message
        exc_info: Whether to include exception traceback
    """
    logger = get_logger()
    logger.error(message, exc_info=exc_info)
