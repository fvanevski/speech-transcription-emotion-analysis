import csv
import gradio as gr
import subprocess
import os
import json
import shutil
import zipfile
import uuid
import torch
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import time

def save_emotion_summary(speaker_stats, output_dir, job_id):
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

def cleanup_old_logs(log_dir="logs", max_age_days=7):
    now = time.time()
    max_age = max_age_days * 86400  # seconds in a day
    for fname in os.listdir(log_dir):
        path = os.path.join(log_dir, fname)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age:
            os.remove(path)

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

HF_TOKEN = "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"
LOG_FILE_NAME = "transcription_log.txt"
ACTIVE_PROCESSES = {}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def safe_run(command, log_file, session_id):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        log_file.write(line)
        log_file.flush()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")

def download_audio(youtube_url, temp_dir, log_file, session_id):
    basename = "audio_input"
    webm_path = os.path.join(temp_dir, f"{basename}.webm")
    wav_path = os.path.join(temp_dir, f"{basename}.wav")
    safe_run(["yt-dlp", "-f", "251", "-o", webm_path, youtube_url], log_file, session_id)
    safe_run(["ffmpeg", "-y", "-i", webm_path, "-ac", "1", "-ar", "16000", "-vn", wav_path], log_file, session_id)
    os.remove(webm_path)
    return wav_path

def run_whisperx(audio_path, output_dir, log_file, session_id):
    safe_run([
        "whisperx", audio_path,
        "--model", "large-v2",
        "--diarize",
        "--hf_token", HF_TOKEN,
        "--output_dir", output_dir,
        "--output_format", "json"
    ], log_file, session_id)


def convert_json_to_structured(json_path):
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    structured = []
    for segment in data.get("segments", []):
        text = segment.get("text", "")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
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
def kill_process(session_id):
    proc = ACTIVE_PROCESSES.pop(session_id, None)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def transcribe_audio(youtube_url, audio_file, cancel_flag):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    webm_path = None
    job_id = str(uuid.uuid4())[:8]
    temp_dir = f"temp_{session_id}_{job_id}"
    output_dir = f"output_{session_id}_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'log_{session_id}_{job_id}.txt')
    zip_path = f"transcription_{session_id}_{job_id}.zip"
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"📄 Transcription Job ID: {job_id}\nStarted: {datetime.now()}\n\n")
            log_file.flush()
            if youtube_url:
                audio_path = download_audio(youtube_url, temp_dir, log_file, session_id)
            elif audio_file:
                input_ext = os.path.splitext(audio_file)[-1]
                audio_path = os.path.join(temp_dir, f"audio_input{input_ext}")
                shutil.copy(audio_file, audio_path)
            else:
                raise ValueError("No input provided.")

            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ], capture_output=True, text=True)
            duration = float(result.stdout.strip())
            if duration < 5.0:
                raise RuntimeError("Audio is too short for reliable diarization.")

            run_whisperx(audio_path, output_dir, log_file, session_id)

        json_output = next((Path(output_dir) / f for f in os.listdir(output_dir) if f.endswith(".json")), None)
        if not json_output or not json_output.exists():
            raise FileNotFoundError("Missing structured JSON output.")

        segments = convert_json_to_structured(json_output)
        structured_path = os.path.join(output_dir, f"structured_transcript_{job_id}.json")
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
        from collections import defaultdict, Counter
        speaker_stats = defaultdict(list)
        for s in segments:
            speaker_stats[s['speaker']].append(s['emotion'])
        summarized = {}
        for spk, emos in speaker_stats.items():
            transitions = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i-1])
            dominant = Counter(emos).most_common(1)[0][0]
            summarized[spk] = {
                'total_segments': len(emos),
                'emotion_transitions': transitions,
                'dominant_emotion': dominant
            }
        emotion_csv, emotion_json = save_emotion_summary(summarized, output_dir, job_id)
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(structured_path, arcname=os.path.basename(structured_path))
            zipf.write(emotion_csv, arcname=os.path.basename(emotion_csv))
            zipf.write(emotion_json, arcname=os.path.basename(emotion_json))
            zipf.write(log_path, arcname=LOG_FILE_NAME)

        shutil.rmtree(temp_dir)
        shutil.rmtree(output_dir)
        return zip_path, f"✅ Transcription complete. Job ID: {job_id}"

    except Exception as e:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n❌ Error: {e}\n")
        except:
            pass
        return None, f"❌ {e}"

    finally:
        kill_process(session_id)
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(output_dir): shutil.rmtree(output_dir, ignore_errors=True)
        if cancel_flag.get("cancel"):
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            if os.path.exists(zip_path): os.remove(zip_path)

def interface_ui():
    cleanup_old_logs()
    with gr.Blocks() as demo:
        youtube = gr.Textbox(label="YouTube Link")
        audio = gr.Audio(label="Upload Audio", type="filepath", sources=["upload"])
        status = gr.Textbox(label="Status", interactive=False)
        download = gr.File(label="Download Transcript")
        button = gr.Button("Submit")
        cancel_flag = {"cancel": False, "running": False}

        def handle_submit(youtube_url, audio_file):
            if cancel_flag.get("running"):
                cancel_flag["cancel"] = True
                cancel_flag["running"] = False
                return None, "❌ Transcription cancelled.", gr.update(value="Submit")
            else:
                cancel_flag["cancel"] = False
                cancel_flag["running"] = True
                result, msg = transcribe_audio(youtube_url, audio_file, cancel_flag)
                cancel_flag["running"] = False
                return gr.update(value=result), gr.update(value=msg), gr.update(value="Submit")

        button.click(fn=handle_submit, inputs=[youtube, audio], outputs=[download, status, button])

        gr.Markdown("### Speaker Relabeling Post-Processor")

        relabel_input = gr.File(label="Upload structured_transcript_*.json")
        relabel_output = gr.File(label="Download Relabeled JSON")
        relabel_btn = gr.Button("Apply Speaker Labels")

        def load_relabel_fields(json_file):
            with open(json_file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            speakers = {}
            for s in data:
                spk = s["speaker"]
                if spk not in speakers:
                    speakers[spk] = s["text"][:60] + "..." if len(s["text"]) > 60 else s["text"]
            return [gr.Textbox(label=f"{k} (e.g., {v})") for k, v in speakers.items()], list(speakers.keys())

        def apply_speaker_labels(json_file, *labels):
            with open(json_file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            speakers = list({seg["speaker"] for seg in data})
            mapping = {speakers[i]: labels[i] if labels[i] else speakers[i] for i in range(len(labels))}
            for seg in data:
                seg["speaker_original"] = seg["speaker"]
                seg["speaker"] = mapping[seg["speaker"]]
            outpath = json_file.name.replace(".json", "_relabeled.json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return outpath

        relabel_input.change(fn=load_relabel_fields, inputs=[relabel_input], outputs=[gr.Column()])
        relabel_btn.click(fn=apply_speaker_labels, inputs=[relabel_input], outputs=[relabel_output])
    return demo

interface_ui().launch()


import csv
import gradio as gr
import subprocess
import os
import json
import shutil
import zipfile
import uuid
import torch
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import time

def cleanup_old_logs(log_dir="logs", max_age_days=7):
    now = time.time()
    max_age = max_age_days * 86400  # seconds in a day
    for fname in os.listdir(log_dir):
        path = os.path.join(log_dir, fname)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age:
            os.remove(path)

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

HF_TOKEN = "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"
LOG_FILE_NAME = "transcription_log.txt"
ACTIVE_PROCESSES = {}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def safe_run(command, log_file, session_id):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        log_file.write(line)
        log_file.flush()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")

def download_audio(youtube_url, temp_dir, log_file, session_id):
    basename = "audio_input"
    webm_path = os.path.join(temp_dir, f"{basename}.webm")
    wav_path = os.path.join(temp_dir, f"{basename}.wav")
    safe_run(["yt-dlp", "-f", "251", "-o", webm_path, youtube_url], log_file, session_id)
    safe_run(["ffmpeg", "-y", "-i", webm_path, "-ac", "1", "-ar", "16000", "-vn", wav_path], log_file, session_id)
    os.remove(webm_path)
    return wav_path

def run_whisperx(audio_path, output_dir, log_file, session_id):
    safe_run([
        "whisperx", audio_path,
        "--model", "large-v2",
        "--diarize",
        "--hf_token", HF_TOKEN,
        "--output_dir", output_dir,
        "--output_format", "json"
    ], log_file, session_id)


def convert_json_to_structured(json_path):
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    structured = []
    for segment in data.get("segments", []):
        text = segment.get("text", "")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
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
def kill_process(session_id):
    proc = ACTIVE_PROCESSES.pop(session_id, None)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def transcribe_audio(youtube_url, audio_file, cancel_flag):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    webm_path = None
    job_id = str(uuid.uuid4())[:8]
    temp_dir = f"temp_{session_id}_{job_id}"
    output_dir = f"output_{session_id}_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'log_{session_id}_{job_id}.txt')
    zip_path = f"transcription_{session_id}_{job_id}.zip"
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"📄 Transcription Job ID: {job_id}\nStarted: {datetime.now()}\n\n")
            log_file.flush()
            if youtube_url:
                audio_path = download_audio(youtube_url, temp_dir, log_file, session_id)
            elif audio_file:
                input_ext = os.path.splitext(audio_file)[-1]
                audio_path = os.path.join(temp_dir, f"audio_input{input_ext}")
                shutil.copy(audio_file, audio_path)
            else:
                raise ValueError("No input provided.")

            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ], capture_output=True, text=True)
            duration = float(result.stdout.strip())
            if duration < 5.0:
                raise RuntimeError("Audio is too short for reliable diarization.")

            run_whisperx(audio_path, output_dir, log_file, session_id)

        json_output = next((Path(output_dir) / f for f in os.listdir(output_dir) if f.endswith(".json")), None)
        if not json_output or not json_output.exists():
            raise FileNotFoundError("Missing structured JSON output.")

        segments = convert_json_to_structured(json_output)
        structured_path = os.path.join(output_dir, f"structured_transcript_{job_id}.json")
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
        from collections import defaultdict, Counter
        speaker_stats = defaultdict(list)
        for s in segments:
            speaker_stats[s['speaker']].append(s['emotion'])
        summarized = {}
        for spk, emos in speaker_stats.items():
            transitions = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i-1])
            dominant = Counter(emos).most_common(1)[0][0]
            summarized[spk] = {
                'total_segments': len(emos),
                'emotion_transitions': transitions,
                'dominant_emotion': dominant
            }
        emotion_csv, emotion_json = save_emotion_summary(summarized, output_dir, job_id)
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(structured_path, arcname=os.path.basename(structured_path))
            zipf.write(emotion_csv, arcname=os.path.basename(emotion_csv))
            zipf.write(emotion_json, arcname=os.path.basename(emotion_json))
            zipf.write(log_path, arcname=LOG_FILE_NAME)

        shutil.rmtree(temp_dir)
        shutil.rmtree(output_dir)
        return zip_path, f"✅ Transcription complete. Job ID: {job_id}"

    except Exception as e:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n❌ Error: {e}\n")
        except:
            pass
        return None, f"❌ {e}"

    finally:
        kill_process(session_id)
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(output_dir): shutil.rmtree(output_dir, ignore_errors=True)
        if cancel_flag.get("cancel"):
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            if os.path.exists(zip_path): os.remove(zip_path)

def interface_ui():
    cleanup_old_logs()
    with gr.Blocks() as demo:
        youtube = gr.Textbox(label="YouTube Link")
        audio = gr.Audio(label="Upload Audio", type="filepath", sources=["upload"])
        status = gr.Textbox(label="Status", interactive=False)
        download = gr.File(label="Download Transcript")
        button = gr.Button("Submit")
        cancel_flag = {"cancel": False, "running": False}

        def handle_submit(youtube_url, audio_file):
            if cancel_flag.get("running"):
                cancel_flag["cancel"] = True
                cancel_flag["running"] = False
                return None, "❌ Transcription cancelled.", gr.update(value="Submit")
            else:
                cancel_flag["cancel"] = False
                cancel_flag["running"] = True
                result, msg = transcribe_audio(youtube_url, audio_file, cancel_flag)
                cancel_flag["running"] = False
                return gr.update(value=result), gr.update(value=msg), gr.update(value="Submit")

        button.click(fn=handle_submit, inputs=[youtube, audio], outputs=[download, status, button])

        gr.Markdown("### Speaker Relabeling Post-Processor")

        relabel_input = gr.File(label="Upload structured_transcript_*.json")
        relabel_output = gr.File(label="Download Relabeled JSON")
        relabel_btn = gr.Button("Apply Speaker Labels")

        def load_relabel_fields(json_file):
            with open(json_file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            speakers = {}
            for s in data:
                spk = s["speaker"]
                if spk not in speakers:
                    speakers[spk] = s["text"][:60] + "..." if len(s["text"]) > 60 else s["text"]
            return [gr.Textbox(label=f"{k} (e.g., {v})") for k, v in speakers.items()], list(speakers.keys())

        def apply_speaker_labels(json_file, *labels):
            with open(json_file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            speakers = list({seg["speaker"] for seg in data})
            mapping = {speakers[i]: labels[i] if labels[i] else speakers[i] for i in range(len(labels))}
            for seg in data:
                seg["speaker_original"] = seg["speaker"]
                seg["speaker"] = mapping[seg["speaker"]]
            outpath = json_file.name.replace(".json", "_relabeled.json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return outpath

        relabel_input.change(fn=load_relabel_fields, inputs=[relabel_input], outputs=[gr.Column()])
        relabel_btn.click(fn=apply_speaker_labels, inputs=[relabel_input], outputs=[relabel_output])
    return demo

interface_ui().launch()


def save_emotion_summary(speaker_stats, output_dir, job_id):
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
