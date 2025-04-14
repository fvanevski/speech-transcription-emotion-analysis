# core/pipeline.py
from core.transcription import Transcription
from core.diarization import Diarization
from core.emotion_analysis import EmotionAnalysis
from core.file_management import FileManager
from core.logging import log_info, log_error
from config.config import Config
import os
import json
import csv
import zipfile
from collections import defaultdict, Counter
from statistics import mean, stdev

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.transcription = Transcription(config)
        self.diarization = Diarization(config)
        self.emotion_analysis = EmotionAnalysis(config)
        self.file_manager = FileManager(config)

    def transcribe(self, input_file):
        try:
            log_info("Starting transcription...")
            result = self.transcription.transcribe(input_file)
            log_info("Transcription completed.")
            return result
        except Exception as e:
            log_error(f"Transcription error: {e}")
            return f"Error: {e}"

    def diarize(self, input_file):
        try:
            log_info("Starting diarization...")
            result = self.diarization.diarize(input_file)
            log_info("Diarization completed.")
            return result
        except Exception as e:
            log_error(f"Diarization error: {e}")
            return f"Error: {e}"

    def analyze_emotion(self, text):
        try:
            log_info("Starting emotion analysis...")
            result = self.emotion_analysis.analyze_emotion(text)
            log_info("Emotion analysis completed.")
            return result
        except Exception as e:
            log_error(f"Emotion analysis error: {e}")
            return f"Error: {e}"

    def download_audio_from_youtube(self, youtube_url):
        try:
            log_info("Starting YouTube audio download...")
            audio_file = self.transcription.download_audio_from_youtube(youtube_url)
            log_info("YouTube audio download completed.")
            return audio_file
        except Exception as e:
            log_error(f"YouTube audio download error: {e}")
            return f"Error: {e}"

    def run_whisperx(self, audio_path, output_dir, log_file, session_id):
        try:
            log_info("Starting WhisperX...")
            self.transcription.run_whisperx(audio_path, output_dir, log_file, session_id)
            log_info("WhisperX completed.")
        except Exception as e:
            log_error(f"WhisperX error: {e}")
            return f"Error: {e}"

    def convert_json_to_structured(self, json_path):
        try:
            log_info("Starting JSON conversion...")
            structured = self.transcription.convert_json_to_structured(json_path)
            log_info("JSON conversion completed.")
            return structured
        except Exception as e:
            log_error(f"JSON conversion error: {e}")
            return f"Error: {e}"

    def save_emotion_summary(self, speaker_stats, output_dir, job_id):
        csv_path = os.path.join(output_dir, f"emotion_summary_{job_id}.csv")
        json_path = os.path.join(output_dir, f"emotion_summary_{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(speaker_stats, jf, indent=2)
        with open(csv_path, "w", newline='', encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["speaker", "total_segments", "emotion_transitions", "dominant_emotion"])
            for speaker, stats in speaker_stats.items():
                writer.writerow([
                    speaker,
                    stats.get("total_segments", 0),
                    stats.get("emotion_transitions", 0),
                    stats.get("dominant_emotion", "unknown")
                ])
        return csv_path, json_path

    def generate_csv_template(self, json_file):
        with open(json_file.name, "r", encoding="utf-8") as f:
            data = json.load(f)
            speakers = {}
            for seg in data:
                spk = seg["speaker"]
                if spk not in speakers:
                    preview = seg["text"][:60].replace(",", "")
                    speakers[spk] = preview
            path = json_file.name.replace(".json", "_speaker_template.csv")
            with open(path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(["speaker", "label", "preview"])
                for spk, example in speakers.items():
                    writer.writerow([spk, "", example])
        return path

    def apply_labels_from_csv(self, json_file, csv_file):
        with open(csv_file.name, "r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            mapping = {row["speaker"]: row["label"] or row["speaker"] for row in reader}
            with open(json_file.name, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                for seg in data:
                    seg["speaker_original"] = seg["speaker"]
                    seg["speaker"] = mapping.get(seg["speaker"], seg["speaker"])

            outpath = json_file.name.replace(".json", "_relabeled.json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            speaker_segments = defaultdict(list)
            for s in data:
                speaker_segments[s['speaker']].append(s)

            summarized = {}
            emotion_vals = {'joy': 1, 'neutral': 0, 'sadness': -1, 'anger': -2, 'surprise': 0.5, 'fear': -1.5}
            for spk, segments in speaker_segments.items():
                emos = [s['emotion'] for s in segments]
                timestamps = [round(s['start'], 2) for s in segments]
                transitions = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i-1])
                dominant = Counter(emos).most_common(1)[0][0]
                emo_numeric = [emotion_vals.get(e, 0) for e in emos]
                volatility = stdev(emo_numeric) if len(emo_numeric) > 1 else 0
                avg_emotion = mean(emo_numeric) if emo_numeric else 0
                timeline = [{'time': timestamps[i], 'emotion': emos[i]} for i in range(len(emos))]
                summarized[spk] = {
                    'total_segments': len(emos),
                    'emotion_transitions': transitions,
                    'dominant_emotion': dominant,
                    'emotion_volatility': volatility,
                    'emotion_score_mean': avg_emotion,
                    'emotion_timeline': timeline
                }

            output_dir = os.path.dirname(outpath)
            job_id = os.path.basename(json_file.name).split('_')[-1].replace('.json', '')
            csv_path, json_path = self.save_emotion_summary(summarized, output_dir, job_id + '_relabeled')

            zip_path = outpath.replace(".json", "_bundle.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(outpath, arcname=os.path.basename(outpath))
                zipf.write(csv_path, arcname=os.path.basename(csv_path))
                zipf.write(json_path, arcname=os.path.basename(json_path))

            return zip_path