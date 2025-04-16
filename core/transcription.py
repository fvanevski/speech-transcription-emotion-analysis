# core/transcription.py
import json
import os
import traceback
from pathlib import Path # Import Path
from typing import Dict, List, Optional, TextIO, Any

import torch
import yt_dlp  # noqa: F401
from torch import Tensor
from torch.nn.functional import softmax
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer)

# --- ADD LOGGING IMPORTS ---
from .logging import log_info, log_warning, log_error
from .utils import safe_run

Segment = Dict[str, Any]
SegmentsList = List[Segment]

class Transcription:
    config: Dict[str, Any]
    device: str
    hf_token: Optional[str]
    emotion_tokenizer: Optional[AutoTokenizer]
    emotion_model: Optional[AutoModelForSequenceClassification]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")
        self.hf_token = config.get("hf_token")
        self.emotion_tokenizer = None
        self.emotion_model = None
        emotion_model_name: Optional[str] = self.config.get("emotion_model_name")
        if not emotion_model_name:
            emotion_model_name = "nateraw/bert-base-uncased-emotion"
            log_warning(f"'emotion_model_name' not found in config, using default: {emotion_model_name}") # USE LOG_WARNING

        try:
            log_info(f"Loading emotion analysis tokenizer: {emotion_model_name}") # USE LOG_INFO
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
            self.emotion_tokenizer = tokenizer

            log_info(f"Loading emotion analysis model: {emotion_model_name} onto device: {self.device}") # USE LOG_INFO
            model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(self.device)
            self.emotion_model = model
            self.emotion_model.eval()
            log_info("Emotion analysis model loaded successfully.") # USE LOG_INFO
        except Exception as e:
            log_error(f"Failed to load emotion analysis model '{emotion_model_name}': {e}") # USE LOG_ERROR
            self.emotion_tokenizer = None
            self.emotion_model = None
            # raise RuntimeError(f"Failed to load emotion model '{emotion_model_name}': {e}") # Keep commented out

    def download_audio_from_youtube(
        self, youtube_url: str, temp_dir: str, log_file_handle: TextIO, session_id: str
    ) -> str:
        temp_dir_path = Path(temp_dir) # Convert temp_dir string to Path object
        basename: str = "audio_input"
        webm_path: Path = temp_dir_path / f"{basename}_{session_id}.webm"
        wav_path: Path = temp_dir_path / f"{basename}_{session_id}.wav"
        webm_path_str: str = str(webm_path)
        wav_path_str: str = str(wav_path)

        yt_dlp_format: str = self.config.get("youtube_dl_format", "251")
        ffmpeg_ac: str = str(self.config.get("ffmpeg_audio_channels", 1))
        ffmpeg_ar: str = str(self.config.get("ffmpeg_audio_samplerate", 16000))

        try:
            log_info(f"Downloading YouTube URL: {youtube_url} to {webm_path_str}") # USE LOG_INFO
            safe_run(
                ["yt-dlp", "-f", yt_dlp_format, "-o", webm_path_str, youtube_url],
                log_file_handle,
                session_id,
            )

            log_info(f"Converting {webm_path_str} to WAV format: {wav_path_str}") # USE LOG_INFO
            safe_run(
                [
                    "ffmpeg", "-y", "-i", webm_path_str,
                    "-ac", ffmpeg_ac, "-ar", ffmpeg_ar,
                    "-vn", wav_path_str,
                ],
                log_file_handle,
                session_id,
            )
            return wav_path_str
        except Exception as e:
            log_error(f"YouTube download/conversion failed for {youtube_url}: {e}") # USE LOG_ERROR
            raise
        finally:
            if webm_path.exists():
                try:
                    webm_path.unlink()
                    log_info(f"Removed intermediate file: {webm_path}") # USE LOG_INFO
                except OSError as e:
                    warn_msg = f"WARN: Failed to remove intermediate file {webm_path}: {e}"
                    try: # Keep fallback print if logging to handle fails
                        if log_file_handle and not log_file_handle.closed:
                            log_file_handle.write(f"[{session_id}] {warn_msg}\n")
                        else:
                             print(warn_msg) # KEEP PRINT (fallback)
                    except:
                        print(warn_msg) # KEEP PRINT (fallback)

    def run_whisperx(
        self, audio_path: str, output_dir: str, log_file_handle: TextIO, session_id: str
    ) -> None:
        if not self.hf_token:
            # Error is raised immediately, so log_error might not be hit, but add anyway
            log_error("Cannot run WhisperX diarization without Hugging Face token.") # USE LOG_ERROR
            raise ValueError("Cannot run WhisperX diarization without Hugging Face token.")

        command: List[str] = [ # ... (command build logic) ...
            "whisperx", audio_path,
            "--model", self.config.get("whisper_model_size", "large-v2"),
            "--diarize",
            "--hf_token", self.hf_token,
            "--output_dir", output_dir,
            "--output_format", self.config.get("whisper_output_format", "json"),
            "--device", self.device,
        ]
        lang: Optional[str] = self.config.get("whisper_language")
        if lang: command.extend(["--language", lang])
        batch_size_val: Optional[Any] = self.config.get("whisper_batch_size")
        if batch_size_val is not None: command.extend(["--batch_size", str(batch_size_val)])
        compute_type: Optional[str] = self.config.get("whisper_compute_type")
        if compute_type: command.extend(["--compute_type", compute_type])

        command_log: List[str] = [arg if i != command.index("--hf_token") + 1 else "*****"
                                  for i, arg in enumerate(command) if "--hf_token" in command]
        log_info(f"Running WhisperX command: {' '.join(command_log)}") # USE LOG_INFO
        safe_run(command, log_file_handle, session_id)

    def convert_json_to_structured(self, json_path: str) -> SegmentsList:
        if not self.emotion_model or not self.emotion_tokenizer:
             # Error is raised immediately
             log_error("Emotion analysis model/tokenizer was not loaded successfully during initialization.") # USE LOG_ERROR
             raise RuntimeError("Emotion analysis model/tokenizer was not loaded successfully during initialization.")

        log_info(f"Reading WhisperX JSON output from: {json_path}") # USE LOG_INFO
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: Dict = json.load(f)
        except FileNotFoundError as e:
            log_error(f"WhisperX output JSON file not found at {json_path}") # USE LOG_ERROR
            raise e
        except json.JSONDecodeError as e:
            log_error(f"Failed to decode JSON from {json_path}") # USE LOG_ERROR
            raise e

        structured: SegmentsList = []
        segments: List[Dict] = data.get("segments", [])
        if not isinstance(segments, list):
             log_warning(f"'segments' key in {json_path} is not a list. Processing as empty.") # USE LOG_WARNING
             segments = []

        log_info(f"Processing {len(segments)} segments for emotion analysis...") # USE LOG_INFO

        for i, segment in enumerate(segments):
            text: str = segment.get("text", "").strip()
            start_time: Optional[float] = segment.get("start")
            end_time: Optional[float] = segment.get("end")
            speaker: str = segment.get("speaker", "unknown")
            words: List[Dict[str, Any]] = segment.get("words", [])

            emotion: str = "analysis_skipped"

            if text:
                try:
                    inputs = self.emotion_tokenizer( # ... (tokenizer call) ...
                        text, return_tensors="pt", truncation=True, padding=True,
                        max_length=self.emotion_tokenizer.model_max_length
                    ).to(self.device)
                    with torch.no_grad():
                        logits: Tensor = self.emotion_model(**inputs).logits
                    probs: Tensor = softmax(logits, dim=1)
                    predicted_id: int = int(torch.argmax(probs).item())
                    if self.emotion_model.config.id2label:
                         emotion = self.emotion_model.config.id2label[predicted_id]
                    else:
                         log_warning(f"Model config missing id2label mapping for segment {i}. Using predicted ID.") # USE LOG_WARNING
                         emotion = f"ID_{predicted_id}"
                except Exception as e:
                    log_warning(f"Failed to analyze emotion for segment {i} ('{text[:50]}...'): {e}") # USE LOG_WARNING
                    emotion = "analysis_failed"
            else:
                emotion = "no_text"

            segment_output: Segment = { # ... (segment dict creation) ...
                "start": start_time, "end": end_time, "text": text,
                "speaker": speaker, "emotion": emotion, "words": words,
            }
            structured.append(segment_output)

        log_info(f"Finished processing segments. Returning {len(structured)} structured segments.") # USE LOG_INFO
        return structured