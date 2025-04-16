# core/file_management.py
"""
Provides a simple class for managing files within configured output and temp directories.
"""

import shutil
from pathlib import Path # Import Path
from typing import Dict, Any, Union # Import typing helpers

# Assuming utils provides these functions
from .utils import create_directory, get_temp_file

class FileManager:
    """
    Manages files within the base output and temporary directories specified in the config.

    Note: Pipeline currently handles job-specific directories separately. This class might
    be intended for managing other project-level files or future refactoring.
    """
    # Type hints for instance variables
    config: Dict[str, Any]
    output_dir: Path
    temp_dir: Path

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the FileManager with configuration and ensures base directories exist.

        Args:
            config (Dict[str, Any]): The application configuration dictionary.
                                     Expected keys: "output_dir", "temp_dir".
        """
        self.config = config
        # Store paths as Path objects
        output_dir_str: str = config.get("output_dir", "output") # Default if not in config
        temp_dir_str: str = config.get("temp_dir", "temp")       # Default if not in config

        self.output_dir = Path(output_dir_str)
        self.temp_dir = Path(temp_dir_str)

        # Ensure directories exist using the utility function (which now uses pathlib)
        create_directory(self.output_dir)
        create_directory(self.temp_dir)

    def create_temp_file(self, suffix: str = "") -> str:
        """
        Creates a temporary file (using utils.get_temp_file) within the system's
        default temporary location, returning its path.

        Note: Relies on utils.get_temp_file which requires manual deletion by caller.

        Args:
            suffix (str, optional): Suffix for the temporary file name. Defaults to "".

        Returns:
            str: The path to the created temporary file.
        """
        # This uses the system's temp dir, not self.temp_dir defined from config
        # The return type matches the current utils.get_temp_file return type
        return get_temp_file(suffix=suffix)

    def save_file(self, content: str, filename: Union[str, Path]) -> Path:
        """
        Saves string content to a file within the configured output directory.

        Args:
            content (str): The string content to save.
            filename (Union[str, Path]): The name of the file (relative to output_dir).

        Returns:
            Path: The full path to the saved file.

        Raises:
            IOError: If writing the file fails.
        """
        file_path: Path = self.output_dir / filename # Use pathlib's / operator
        try:
            # Ensure parent directory exists if filename includes subdirs
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            # Consider adding log_info here if logging is integrated
            print(f"INFO: File saved successfully to {file_path}") # Simple print for now
        except IOError as e:
            # Consider adding log_error here
            print(f"ERROR: Failed to save file {file_path}: {e}")
            raise # Re-raise the error
        return file_path

    def cleanup_temp_files(self) -> None:
        """
        Removes and recreates the temporary directory defined in the configuration.

        Warning: This will delete everything inside the configured temp_dir.
        """
        # Log potentially?
        print(f"INFO: Cleaning up temporary directory: {self.temp_dir}")
        try:
            if self.temp_dir.exists(): # Check if it exists before trying to remove
                 shutil.rmtree(self.temp_dir)
            # Recreate the directory using the utility function
            create_directory(self.temp_dir)
        except OSError as e:
            # Log potentially?
             print(f"ERROR: Failed to cleanup temporary directory {self.temp_dir}: {e}")
             # Decide if this should raise an error