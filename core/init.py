# core/init.py
# Removed FileManager import
# Removed EmotionAnalysis import previously
from .diarization import Diarization # Keep if diarization.py exists or is planned
from .logging import log_error, log_info, log_warning # Add logging imports if used directly from core
from .pipeline import Pipeline # Add Pipeline if needed at core level
from .plotting import generate_all_plots # Add plotting if needed at core level
from .transcription import Transcription
from .utils import create_directory, get_temp_file, safe_run # Add utils functions if needed

# Define what gets imported with 'from core import *' if desired
# __all__ = ['Pipeline', 'Transcription', 'utils', 'logging', 'plotting']