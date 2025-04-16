# core/logging.py
"""
Configures and provides basic logging functionality for the application.
"""

import logging
import os
from pathlib import Path # Import pathlib
from typing import Dict, Any # Import typing helpers

# Removed Config import as setup_logging receives the config object directly
# from config.config import Config

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up basic file logging based on the provided configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary, expected to
                                 contain 'log_level' and 'output_dir'.
    """
    log_level_str: str = config.get("log_level", "INFO").upper() # Get level, default INFO, ensure uppercase
    log_dir_str: str = config.get("output_dir", "output") # Get dir, default "output"
    log_filename: str = "app.log" # Keep filename simple, could be made configurable

    log_dir_path = Path(log_dir_str)
    log_file_path = log_dir_path / log_filename

    # Ensure the log directory exists
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Cannot log this error to file yet, print to console
        print(f"ERROR: Failed to create log directory {log_dir_path}: {e}")
        # Optionally raise error or exit? For now, continue, logging might fail.

    # Define logging format
    log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'

    # Get numeric log level
    log_level: int = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger
    try:
        logging.basicConfig(
            filename=str(log_file_path), # basicConfig expects string path
            level=log_level,
            format=log_format,
            datefmt=date_format,
            # Force=True might be needed if basicConfig was called elsewhere (e.g. by a library)
            # force=True
        )
        # Optional: Prevent libraries (like transformers) from flooding logs below a certain level
        # logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.info("Logging setup complete.") # Log initial message
    except Exception as e:
         # If basicConfig fails, print error to console
        print(f"ERROR: Failed to configure logging to file {log_file_path}: {e}")


def log_info(message: str) -> None:
    """Logs an informational message."""
    # Get logger for the module calling this function (helps identify source)
    logger = logging.getLogger(__name__)
    logger.info(message)

def log_error(message: str) -> None:
    """Logs an error message."""
    logger = logging.getLogger(__name__)
    logger.error(message)

def log_warning(message: str) -> None:
    """Logs a warning message."""
    logger = logging.getLogger(__name__)
    logger.warning(message)