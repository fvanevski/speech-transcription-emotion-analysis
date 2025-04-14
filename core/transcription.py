# core/transcription.py
import whisperx
import yt_dlp
import os
import subprocess
import json
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Transcription:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device")
        self.hf_token = config.get("hf_token")

    def download_audio_from_youtube(self, youtube_url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(self.config.get("temp_dir"), 'yt_audio.%(ext)s'),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return os.path.join(self.config.get("temp_dir"), 'yt_audio.mp3')

    def run_whisperx(self, audio_path, output_dir, log_file, session_id):
        safe_run([
            "whisperx", audio_path,
            "--model", "large-v2",
            "--diarize",
            "--hf_token", self.hf_token,
            "--output_dir", output_dir,
            "--output_format", "json",
            "--device", self.device  # Add device parameter
        ], log_file, session_id)

    def convert_json_to_structured(self, json_path):
        tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
        model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion").to(self.device)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        structured = []
        for segment in data.get("segments", []):
            text = segment.get("text", "")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = softmax(logits, dim=1)
            predicted = torch.argmax(probs).item()
            emotion = model.config.id2label[predicted]
            structured.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
                "speaker": segment.get("speaker", "unknown"),
                "emotion": emotion,
                "words": segment.get("words", [])
            })
        return structured

def safe_run(command, log_file, session_id):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        log_file.write(line)
        log_file.flush()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")