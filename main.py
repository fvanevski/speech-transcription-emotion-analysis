# main.py
import torch # Import torch
from config.config import Config
from ui.main_gui import UI
# Assuming PostProcessUI might be launched separately or via the main UI
# from ui.postprocess_gui import PostProcessUI
from core.logging import setup_logging

def main():
    # Apply Torch backend settings (consider adding checks for GPU availability/compatibility if needed)
    try:
        if torch.cuda.is_available(): # Only set if CUDA is available
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("INFO: Applied TF32 optimizations for CUDA.") # Optional: info message
    except Exception as e:
        print(f"WARN: Could not set Torch backend settings: {e}") # Optional: warning

    # --- Original main logic ---
    config = Config()
    setup_logging(config)
    # Ensure HF token is handled securely (check done in Config, but verify config.py is fixed)
    hf_token = config.get("hf_token")
    if not hf_token or hf_token == "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": # Check if token is missing or still the leaked one
         print("CRITICAL WARNING: Hugging Face token is missing or insecurely configured in config.py or environment variables. Diarization may fail.")
         # Optionally, you could raise an error here if the token is absolutely required:
         # raise ValueError("Hugging Face token (HF_TOKEN) not configured securely.")

    app = UI(config)
    print("Launching Gradio UI...") # Optional: info message
    app.launch()

if __name__ == "__main__":
    main()