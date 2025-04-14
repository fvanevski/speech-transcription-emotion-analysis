# core/transcription.py
import os
import subprocess
import json
import torch # Added import
from torch.nn.functional import softmax # Added import
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Added imports
import yt_dlp # Keep for download function
# Removed whisperx import as it's only used via CLI in run_whisperx

# Import logging functions if needed within this module, or rely on pipeline logging
# from .logging import log_info, log_error

class Transcription:
    def __init__(self, config):
        """
        Initializes the Transcription class, loading necessary configurations
        and the emotion analysis model.
        """
        self.config = config
        self.device = config.get("device", "cpu") # Default to CPU if not specified
        self.hf_token = config.get("hf_token") # Already validated in Config class

        # --- Load Emotion Analysis Model Once ---
        self.emotion_tokenizer = None
        self.emotion_model = None
        emotion_model_name = "nateraw/bert-base-uncased-emotion"
        try:
            print(f"INFO: Loading emotion analysis tokenizer: {emotion_model_name}")
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
            print(f"INFO: Loading emotion analysis model: {emotion_model_name} onto device: {self.device}")
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(self.device)
            # Set model to evaluation mode (important for dropout, batchnorm layers)
            self.emotion_model.eval()
            print("INFO: Emotion analysis model loaded successfully.")
        except Exception as e:
            # Log error appropriately here if logging is set up
            print(f"ERROR: Failed to load emotion analysis model '{emotion_model_name}': {e}")
            # Depending on requirements, either raise the error or allow continuation without emotion analysis
            # For now, we let it continue, but convert_json_to_structured will fail if model is None
            # raise RuntimeError(f"Failed to load emotion model: {e}")

    def download_audio_from_youtube(self, youtube_url, temp_dir, log_file_handle, session_id):
        """Downloads audio from YouTube and converts it to WAV."""
        # This method relies on safe_run being available
        basename = "audio_input"
        # Ensure unique names if multiple downloads happen in parallel to the same temp_dir
        # Using session_id could help here if temp_dir is shared, but pipeline.py creates unique job dirs
        webm_path = os.path.join(temp_dir, f"{basename}_{session_id}.webm")
        wav_path = os.path.join(temp_dir, f"{basename}_{session_id}.wav")

        try:
            print(f"INFO: Downloading YouTube URL: {youtube_url} to {webm_path}")
            # Prefered audio format 251 (opus), check yt-dlp docs for alternatives if needed
            safe_run(["yt-dlp", "-f", "251", "-o", webm_path, youtube_url], log_file_handle, session_id)

            print(f"INFO: Converting {webm_path} to WAV format: {wav_path}")
            # Convert to WAV, 16kHz sample rate, mono channel using ffmpeg
            safe_run(["ffmpeg", "-y", "-i", webm_path, "-ac", "1", "-ar", "16000", "-vn", wav_path], log_file_handle, session_id)

            return wav_path
        except Exception as e:
            # Log error
            print(f"ERROR: YouTube download/conversion failed for {youtube_url}: {e}")
            raise # Re-raise the exception to be caught by the pipeline
        finally:
            # Clean up intermediate webm file
            if os.path.exists(webm_path):
                try:
                    os.remove(webm_path)
                    print(f"INFO: Removed intermediate file: {webm_path}")
                except OSError as e:
                    print(f"WARN: Failed to remove intermediate file {webm_path}: {e}")


    def run_whisperx(self, audio_path, output_dir, log_file_handle, session_id):
        """Runs the WhisperX CLI command for transcription and diarization."""
        # This method relies on safe_run being available
        # Ensure hf_token and device are correctly passed from config
        if not self.hf_token:
             # This should have been caught by Config validation, but double-check
             raise ValueError("Cannot run WhisperX diarization without Hugging Face token.")
             
        command = [
            "whisperx", audio_path,
            "--model", self.config.get("whisper_model_size", "large-v2"), # Allow model size config
            "--diarize", # Diarization is enabled
            "--hf_token", self.hf_token,
            "--output_dir", output_dir,
            "--output_format", "json",
            "--device", self.device,
            # Add other relevant whisperx parameters from config if needed
            # e.g., "--language", self.config.get("language", "en"),
            # e.g., "--batch_size", str(self.config.get("whisper_batch_size", 16)),
            # e.g., "--compute_type", self.config.get("compute_type", "float16") # if using GPU
        ]
        print(f"INFO: Running WhisperX command: {' '.join(command)}") # Log command without token for security
        safe_run(command, log_file_handle, session_id)


    def convert_json_to_structured(self, json_path):
        """
        Reads WhisperX JSON output, adds emotion analysis results using the pre-loaded model,
        and returns a list of structured segment dictionaries.
        """
        # Check if emotion model loaded successfully during initialization
        if not self.emotion_model or not self.emotion_tokenizer:
            raise RuntimeError("Emotion analysis model was not loaded successfully during initialization. Cannot perform emotion analysis.")

        print(f"INFO: Reading WhisperX JSON output from: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: WhisperX output JSON file not found at {json_path}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Failed to decode JSON from {json_path}")
            raise

        structured = []
        segments = data.get("segments", [])
        print(f"INFO: Processing {len(segments)} segments for emotion analysis...")

        for i, segment in enumerate(segments):
            # Use .get for safer access to potentially missing keys
            text = segment.get("text", "").strip()
            start_time = segment.get("start")
            end_time = segment.get("end")
            speaker = segment.get("speaker", "unknown")
            words = segment.get("words", [])

            emotion = "analysis_skipped" # Default emotion if text is empty or model fails

            if text: # Only analyze if text is present
                try:
                    # Tokenize and predict using pre-loaded model/tokenizer
                    inputs = self.emotion_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=self.emotion_tokenizer.model_max_length # Use model's max length
                    ).to(self.device)

                    # Ensure inference runs without calculating gradients
                    with torch.no_grad():
                        logits = self.emotion_model(**inputs).logits

                    # Get probabilities and predicted label ID
                    probs = softmax(logits, dim=1)
                    predicted_id = torch.argmax(probs).item()
                    # Get the label string from the model's config
                    emotion = self.emotion_model.config.id2label[predicted_id]

                    # Optional: Log progress periodically
                    # if (i + 1) % 50 == 0:
                    #     print(f"INFO: Processed {i+1}/{len(segments)} segments...")

                except Exception as e:
                    print(f"WARN: Failed to analyze emotion for segment {i} ('{text[:50]}...'): {e}")
                    emotion = "analysis_failed" # Indicate failure for this segment
            else:
                emotion = "no_text" # Indicate segment had no text

            structured.append({
                "start": start_time,
                "end": end_time,
                "text": text, # Store the stripped text
                "speaker": speaker,
                "emotion": emotion, # Emotion label or status
                "words": words # Include word timings if present
            })
            
        print(f"INFO: Finished processing segments. Returning {len(structured)} structured segments.")
        return structured

# --- Helper Function (kept outside class, could be moved to utils.py) ---
def safe_run(command, log_file_handle, session_id=None):
    """
    Runs an external command safely, logging output in real-time.

    Args:
        command (list): The command and its arguments as a list of strings.
        log_file_handle: An open file handle for writing logs.
        session_id (str, optional): Identifier for logging context. Defaults to None.
    """
    log_prefix = f"[{session_id if session_id else 'PROC'}] "
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            encoding='utf-8', # Be explicit about encoding
            errors='replace' # Handle potential decoding errors in output
        )

        # Log output line by line
        if log_file_handle:
            for line in iter(process.stdout.readline, ''):
                log_line = f"{log_prefix}{line}"
                log_file_handle.write(log_line)
                log_file_handle.flush() # Ensure logs are written immediately
        else:
            # If no log handle, just consume output to prevent pipe filling
             process.communicate()


        process.wait() # Wait for the process to complete

        if process.returncode != 0:
            error_msg = f"Command failed with exit code {process.returncode}: {' '.join(command)}"
            # Log the error before raising
            if log_file_handle:
                 log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
                 log_file_handle.flush()
            raise RuntimeError(error_msg)

    except FileNotFoundError:
        error_msg = f"Command not found: {command[0]}. Ensure it is installed and in PATH."
        if log_file_handle:
             log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
             log_file_handle.flush()
        raise FileNotFoundError(error_msg)
    except Exception as e:
        # Catch other potential errors during process execution
        error_msg = f"An error occurred while running command {' '.join(command)}: {e}"
        if log_file_handle:
             log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
             log_file_handle.flush()
        raise RuntimeError(error_msg)