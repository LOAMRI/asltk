"""
Tests for the ASLTK logging functionality.
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest

from asltk.logging_config import (
    get_logger,
    setup_logging,
    configure_for_scripts,
    log_function_call,
    log_processing_step,
    log_data_info,
    log_warning_with_context,
    log_error_with_traceback,
    PACKAGE_LOGGER_NAME
)


class TestLoggingSetup:
    """Test logging configuration functionality."""
    
    def test_get_logger_returns_correct_logger(self):
        """Test that get_logger returns a logger with the correct name."""
        logger = get_logger()
        assert logger.name == PACKAGE_LOGGER_NAME
        
        logger_with_name = get_logger('test')
        assert logger_with_name.name == f"{PACKAGE_LOGGER_NAME}.test"
    
    def test_setup_logging_console_only(self, caplog):
        """Test basic logging setup with console output."""
        # Enable propagation so caplog can capture messages
        caplog.set_level(logging.INFO, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.INFO, console_output=True, file_output=None)
        
        # Enable propagation for test
        logger = get_logger()
        logger.propagate = True
        logger.info("Test message")
        
        assert "Test message" in caplog.text
    
    def test_setup_logging_with_file_output(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            
            setup_logging(
                level=logging.INFO,
                console_output=False,
                file_output=log_file
            )
            
            logger = get_logger()
            logger.info("Test file message")
            
            # Check that log file was created and contains the message
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                log_content = f.read()
            assert "Test file message" in log_content
    
    def test_configure_for_scripts_verbose(self, caplog):
        """Test script configuration with verbose mode."""
        caplog.set_level(logging.DEBUG, logger=PACKAGE_LOGGER_NAME)
        configure_for_scripts(verbose=True)
        
        logger = get_logger()
        logger.propagate = True
        logger.debug("Debug message")
        
        assert "Debug message" in caplog.text
    
    def test_configure_for_scripts_normal(self, caplog):
        """Test script configuration without verbose mode."""
        caplog.set_level(logging.DEBUG, logger=PACKAGE_LOGGER_NAME)
        configure_for_scripts(verbose=False)
        
        logger = get_logger()
        logger.propagate = True
        logger.debug("Debug message")
        logger.info("Info message")
        
        # Debug should not appear (since it's filtered by level), info should
        assert "Info message" in caplog.text


class TestLoggingHelpers:
    """Test logging helper functions."""
    
    def test_log_function_call(self, caplog):
        """Test function call logging."""
        caplog.set_level(logging.DEBUG, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.DEBUG, console_output=True)
        
        logger = get_logger()
        logger.propagate = True
        log_function_call("test_function", param1="value1", param2=42)
        
        assert "Calling test_function(param1=value1, param2=42)" in caplog.text
    
    def test_log_processing_step(self, caplog):
        """Test processing step logging."""
        caplog.set_level(logging.INFO, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.INFO, console_output=True)
        
        logger = get_logger()
        logger.propagate = True
        log_processing_step("Data loading", "from file.nii")
        
        assert "Processing step: Data loading - from file.nii" in caplog.text
    
    def test_log_data_info(self, caplog):
        """Test data info logging."""
        caplog.set_level(logging.INFO, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.INFO, console_output=True)
        
        logger = get_logger()
        logger.propagate = True
        log_data_info("ASL image", (64, 64, 30, 100), "/path/to/file.nii")
        
        assert "Loaded ASL image: shape=(64, 64, 30, 100), path=/path/to/file.nii" in caplog.text
    
    def test_log_warning_with_context(self, caplog):
        """Test warning logging with context."""
        caplog.set_level(logging.WARNING, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.WARNING, console_output=True)
        
        logger = get_logger()
        logger.propagate = True
        log_warning_with_context("Something went wrong", "during registration")
        
        assert "Something went wrong (Context: during registration)" in caplog.text
    
    def test_log_error_with_traceback(self, caplog):
        """Test error logging."""
        caplog.set_level(logging.ERROR, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.ERROR, console_output=True)
        
        logger = get_logger()
        logger.propagate = True
        log_error_with_traceback("Critical error occurred", exc_info=False)
        
        assert "Critical error occurred" in caplog.text


class TestLoggingIntegration:
    """Test logging integration with ASL modules."""
    
    def test_asldata_logging_basic(self, caplog):
        """Test that ASLData creates logging messages."""
        caplog.set_level(logging.INFO, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.INFO, console_output=True)
        
        # Import after setting up logging to ensure logger is configured
        from asltk.asldata import ASLData
        
        # Enable propagation for the specific logger
        asldata_logger = get_logger('asldata')
        asldata_logger.propagate = True
        
        # Create empty ASLData object
        asl_data = ASLData()
        
        assert "Creating ASLData object" in caplog.text
    
    def test_asldata_logging_with_parameters(self, caplog):
        """Test ASLData logging with parameters."""
        caplog.set_level(logging.INFO, logger=PACKAGE_LOGGER_NAME)
        setup_logging(level=logging.INFO, console_output=True)
        
        from asltk.asldata import ASLData
        
        # Enable propagation for the specific logger
        asldata_logger = get_logger('asldata')
        asldata_logger.propagate = True
        
        # Create ASLData with timing parameters
        asl_data = ASLData(ld_values=[1.8], pld_values=[1.8])
        
        assert "ASL timing parameters - LD: [1.8], PLD: [1.8]" in caplog.text


def test_logging_level_configuration():
    """Test that different log levels work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "levels.log")
        
        # Test DEBUG level
        setup_logging(level=logging.DEBUG, console_output=False, file_output=log_file)
        logger = get_logger()
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content
        
        # Clear log file
        os.remove(log_file)
        
        # Test WARNING level
        setup_logging(level=logging.WARNING, console_output=False, file_output=log_file)
        logger = get_logger()
        
        logger.debug("Debug message 2")
        logger.info("Info message 2")
        logger.warning("Warning message 2")
        logger.error("Error message 2")
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert "Debug message 2" not in content
        assert "Info message 2" not in content
        assert "Warning message 2" in content
        assert "Error message 2" in content


def test_logging_does_not_break_existing_functionality():
    """Test that adding logging doesn't break existing functionality."""
    # Test that ASLData still works normally
    from asltk.asldata import ASLData
    
    # This should not raise any exceptions
    asl_data = ASLData(ld_values=[1.8], pld_values=[1.8])
    assert asl_data.get_ld() == [1.8]
    assert asl_data.get_pld() == [1.8]