import whisperx
import yt_dlp
import os
import subprocess

class Transcription:
    def __init__(self, config):
        self.config = config

    def transcribe(self, input_file):
        model = whisperx.load_model("base")
        audio = whisperx.load_audio(input_file)
        result = model.transcribe(audio)
        return result

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