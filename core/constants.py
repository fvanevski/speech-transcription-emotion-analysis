# core/constants.py
"""
Stores constant values used across the core modules.
"""

from typing import Dict

# Emotion-to-Value mapping for quantitative analysis (e.g., volatility, score timelines)
EMO_VAL: Dict[str, float] = {
    "joy": 1.0,
    "love": 0.8, # Added love based on plotting colors, assign a value
    "surprise": 0.5,
    "neutral": 0.0,
    "fear": -1.5,
    "sadness": -1.0,
    "disgust": -1.8, # Added disgust based on plotting colors, assign a value
    "anger": -2.0,
    # Handle special/meta categories
    "unknown": 0.0,
    "analysis_skipped": 0.0,
    "analysis_failed": 0.0,
    "no_text": 0.0,
    # Add other emotions from your model if necessary, e.g.:
    # "optimism": 0.7,
    # "pessimism": -0.7,
}

# Define other constants here as needed, for example:
# LOG_FILE_NAME: str = "process_log.txt"
# STRUCTURED_TRANSCRIPT_NAME: str = "structured_transcript.json"
# ...etc...
# Consider moving constants from pipeline.py here if they are widely used.