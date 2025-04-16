# core/utils.py
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, TextIO, Union # Added Union

# --- ADD LOGGING IMPORTS ---
from .logging import log_warning, log_error

def create_directory(path: Union[str, Path]) -> None:
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_warning(f"Could not create directory {dir_path}: {e}") # USE LOG_WARNING

def get_temp_file(suffix: str = "") -> str:
    # ... (no changes needed here) ...
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file_path: str = temp_file.name
    temp_file.close()
    return temp_file_path

def safe_run(command: List[str], log_file_handle: Optional[TextIO], session_id: Optional[str] = None) -> None:
    safe_command: List[str] = [str(item) for item in command]
    log_prefix: str = f"[{session_id if session_id else 'PROC'}] "
    process: Optional[subprocess.Popen] = None

    try:
        process = subprocess.Popen( # ... (popen args) ...
            safe_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )

        if log_file_handle and process.stdout:
            for line in iter(process.stdout.readline, ''):
                log_line: str = f"{log_prefix}{line}"
                try:
                    log_file_handle.write(log_line)
                    log_file_handle.flush()
                except Exception as log_e:
                     # Keep fallback print
                     print(f"WARN: Failed to write log line: {log_e}") # KEEP PRINT (fallback)
            process.stdout.close()

        return_code: int = process.wait()

        if return_code != 0:
            error_msg: str = f"Command failed with exit code {return_code}: {' '.join(safe_command)}"
            if log_file_handle:
                 try: log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
                 except Exception as log_e:
                      # Keep fallback print
                      print(f"WARN: Failed to write command error to log: {log_e}") # KEEP PRINT (fallback)
            raise RuntimeError(error_msg)

    except FileNotFoundError:
        error_msg = f"Command not found: {safe_command[0]}. Ensure it is installed and in PATH."
        if log_file_handle:
             try: log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
             except Exception as log_e:
                  # Keep fallback print
                  print(f"WARN: Failed to write FileNotFoundError to log: {log_e}") # KEEP PRINT (fallback)
        # Log the error using the logger as well, if possible
        log_error(error_msg) # USE LOG_ERROR here in addition to writing to handle
        raise FileNotFoundError(error_msg) from None

    except Exception as e:
        error_msg = f"An error occurred while running command {' '.join(safe_command)}: {e}"
        if log_file_handle:
             try: log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n{traceback.format_exc()}\n")
             except Exception as log_e:
                  # Keep fallback print
                  print(f"WARN: Failed to write other exception to log: {log_e}") # KEEP PRINT (fallback)
        # Log the error using the logger as well
        log_error(error_msg) # USE LOG_ERROR here
        log_error(traceback.format_exc()) # Log traceback too

        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Keep termination warnings as prints
                print(f"{log_prefix}WARN: Process did not terminate gracefully, killing.") # KEEP PRINT
                process.kill()
            except Exception as term_err:
                print(f"{log_prefix}WARN: Error terminating process after failure: {term_err}") # KEEP PRINT
        raise RuntimeError(error_msg) from e