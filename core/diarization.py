# core/diarization.py
import whisperx

class Diarization:
    def __init__(self, config):
        self.config = config

    def diarize(self, input_file):
        model = whisperx.load_model("base")
        audio = whisperx.load_audio(input_file)
        diarization_result = model.diarize(audio)
        return diarization_result