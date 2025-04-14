# core/transcription.py
import whisperx

class Transcription:
    def __init__(self, config):
        self.config = config

    def transcribe(self, input_file):
        model = whisperx.load_model("base")
        audio = whisperx.load_audio(input_file)
        result = model.transcribe(audio)
        return result