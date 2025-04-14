
import gradio as gr
import os
import json
import csv
import zipfile
from collections import defaultdict, Counter
from statistics import mean, stdev

def save_emotion_summary(speaker_stats, output_dir, job_id):
    csv_path = os.path.join(output_dir, f"emotion_summary_{job_id}.csv")
    json_path = os.path.join(output_dir, f"emotion_summary_{job_id}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(speaker_stats, jf, indent=2)
    with open(csv_path, "w", newline='', encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["speaker", "total_segments", "emotion_transitions", "dominant_emotion", "emotion_volatility", "emotion_score_mean"])
        for speaker, stats in speaker_stats.items():
            writer.writerow([
                speaker,
                stats.get("total_segments", 0),
                stats.get("emotion_transitions", 0),
                stats.get("dominant_emotion", "unknown"),
                stats.get("emotion_volatility", 0),
                stats.get("emotion_score_mean", 0)
            ])
    return csv_path, json_path

def generate_csv_template(json_file):
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

def apply_labels_from_csv(json_file, csv_file):
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
    csv_path, json_path = save_emotion_summary(summarized, output_dir, job_id + '_relabeled')

    zip_path = outpath.replace(".json", "_bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(outpath, arcname=os.path.basename(outpath))
        zipf.write(csv_path, arcname=os.path.basename(csv_path))
        zipf.write(json_path, arcname=os.path.basename(json_path))

    return zip_path

def interface_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Speaker Labeling & Emotion Summary")

        transcript_json = gr.File(label="Upload structured_transcript_*.json")
        label_csv = gr.File(label="Upload speaker_labels.csv")
        labeled_output = gr.File(label="Download Relabeled Files (ZIP)")
        apply_btn = gr.Button("Apply Speaker Labels")

        template_csv = gr.File(label="Download Speaker CSV Template")
        generate_btn = gr.Button("Generate Label Template")

        generate_btn.click(fn=generate_csv_template, inputs=[transcript_json], outputs=[template_csv])
        apply_btn.click(fn=apply_labels_from_csv, inputs=[transcript_json, label_csv], outputs=[labeled_output])

    return demo

interface_ui().launch()
