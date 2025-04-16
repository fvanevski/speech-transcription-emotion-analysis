# main.py
import torch
import sys
import traceback  # <--- ADD THIS IMPORT

from config.config import Config
# --- UPDATE THIS IMPORT ---
from core.logging import setup_logging, log_error, log_info # Add log_info
from ui.main_gui import UI


def main():
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Keep print here as logging might not be set up yet
            print("INFO: Applied TF32 optimizations for CUDA.")
    except Exception as e:
        # Keep print here
        print(f"WARN: Could not set Torch backend settings: {e}")

    try:
        config = Config()
        # Logging is set up HERE
        setup_logging(config.config)

        hf_token = config.get("hf_token")
        # Check token *after* logging is set up
        if not hf_token or hf_token == "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":
            log_error( # Use log_error now
                "Hugging Face token is missing or insecurely configured in config.py or environment variables. Diarization may fail."
            )
        else:
            # Use log_info now
            log_info("Hugging Face token configured.") # <--- This line caused the first error

        app = UI(config)
        log_info("Launching Gradio UI...") # Use log_info
        if not hasattr(sys, 'ps1'):
            try:
                print(f"DEBUG: Attributes of app object: {dir(app)}") # List available methods/attributes
            except Exception as debug_e:
                print(f"DEBUG: Failed to get dir(app): {debug_e}")
            app.launch(server_name="0.0.0.0")
        else:
             log_warning("Running in interactive mode. Gradio UI might not launch correctly. Run as script.")

    except ValueError as ve:
        log_error(f"Configuration error: {ve}")
        print(f"FATAL: Configuration error prevented startup. Please check config/environment: {ve}")
    except Exception as e:
        log_error(f"An unexpected error occurred during startup: {e}")
        # --- USE IMPORTED TRACEBACK ---
        log_error(traceback.format_exc()) # <--- This line caused the second error
        print(f"FATAL: Unexpected error during startup: {e}. Check logs.")


if __name__ == "__main__":
    main()