# core/pipeline.py
import os
import json
import csv
import zipfile
import uuid
import shutil
import subprocess
import traceback # Import for error logging
import pandas as pd # Import for DataFrame handling
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from statistics import mean, stdev

# Assuming these imports provide the necessary classes and functions
from core.transcription import Transcription
from core.file_management import FileManager
from core.logging import log_info, log_error
# Assumes plotting.py exists and has generate_all_plots
from core.plotting import generate_all_plots

# --- Define emotion values ---
EMO_VAL = {'joy': 1, 'neutral': 0, 'sadness': -1, 'anger': -2, 'surprise': 0.5, 'fear': -1.5,
           'unknown': 0, 'analysis_skipped': 0, 'analysis_failed': 0, 'no_text': 0 }

# --- Define constants ---
LOG_FILE_NAME = "process_log.txt"
STRUCTURED_TRANSCRIPT_NAME = "structured_transcript.json" # Base name for initial output
RELABELED_SUFFIX = "_relabeled" # Suffix for relabeled files (NO extension)
EMOTION_SUMMARY_CSV_NAME = "emotion_summary.csv" # Base name for summary csv
EMOTION_SUMMARY_JSON_NAME = "emotion_summary.json" # Base name for summary json
FINAL_ZIP_SUFFIX = "_final_bundle.zip" # Suffix for final zip

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.transcription = Transcription(config)
        self.file_manager = FileManager(config)

    def _run_ffprobe_duration_check(self, audio_path):
        """Checks if audio duration is sufficient."""
        try:
            log_info(f"Running ffprobe duration check for: {audio_path}")
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ], capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            log_info(f"Audio duration: {duration} seconds")
            min_duration = self.config.get("min_diarization_duration", 5.0)
            if duration < min_duration:
                raise ValueError(f"Audio duration ({duration:.2f}s) is less than minimum required ({min_duration}s) for reliable diarization.")
            return True
        except FileNotFoundError: log_error("ffprobe command not found."); return True
        except subprocess.CalledProcessError as e: log_error(f"ffprobe failed: {e}"); raise RuntimeError(f"ffprobe check failed: {e.stderr}")
        except ValueError as e: log_error(f"ffprobe parsing/check error: {e}"); raise e
        except Exception as e: log_error(f"ffprobe unexpected error: {e}"); raise RuntimeError(f"Duration check failed unexpectedly: {e}")

    def _get_speaker_previews(self, segments):
        """Extracts unique speaker IDs and their first dialogue snippet."""
        speakers_data = {}
        for seg in segments:
            spk_id = str(seg.get('speaker', 'unknown'))
            if spk_id not in speakers_data:
                preview_text = seg.get("text", "").strip()[:80]
                if preview_text: speakers_data[spk_id] = preview_text
        preview_list = [
            {'Speaker ID': spk_id, 'Dialogue Preview': preview, 'Enter Label Here': ''}
            for spk_id, preview in sorted(speakers_data.items())
        ]
        return preview_list

    def process_audio(self, input_source):
        """
        Performs initial processing. Returns status, path to structured JSON,
        and a list of dicts containing speaker IDs and previews for the UI Dataframe.
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = str(uuid.uuid4())[:8]
        base_dir = self.config.get("output_dir", "output")
        job_output_dir = os.path.join(base_dir, f"job_{session_id}_{job_id}")
        temp_dir = os.path.join(job_output_dir, "temp")
        log_path = os.path.join(job_output_dir, LOG_FILE_NAME)

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(job_output_dir, exist_ok=True)

        log_file_handle = None
        structured_path = None

        try:
            log_file_handle = open(log_path, "w", encoding="utf-8")
            log_file_handle.write(f"📄 Processing Job ID: {job_id}\nSession ID: {session_id}\nStarted: {datetime.now()}\n\n")
            log_file_handle.flush()
            log_info(f"Job {job_id} started. Output dir: {job_output_dir}")

            # --- Handle Input ---
            audio_path = None
            if isinstance(input_source, str) and input_source.startswith(('http://', 'https://')):
                 audio_path = self.transcription.download_audio_from_youtube(input_source, temp_dir, log_file_handle, session_id)
            elif isinstance(input_source, str) and os.path.exists(input_source):
                 input_ext = os.path.splitext(input_source)[-1]
                 temp_audio_path = os.path.join(temp_dir, f"input_audio{input_ext}")
                 shutil.copy(input_source, temp_audio_path)
                 audio_path = temp_audio_path
            else: raise ValueError("Invalid input source provided.")

            # --- Duration Check ---
            self._run_ffprobe_duration_check(audio_path)

            # --- Run WhisperX ---
            self.transcription.run_whisperx(audio_path, job_output_dir, log_file_handle, session_id)

            # --- Find WhisperX Output JSON ---
            audio_filename_stem = Path(audio_path).stem
            expected_json_filename = f"{audio_filename_stem}.json"
            json_output_path = os.path.join(job_output_dir, expected_json_filename)
            if not os.path.exists(json_output_path):
                 found_json = None
                 for f_name in os.listdir(job_output_dir): # Renamed loop var to f_name
                     if f_name.endswith(".json") and EMOTION_SUMMARY_JSON_NAME not in f_name:
                         found_json = os.path.join(job_output_dir, f_name); break
                 if not found_json: raise FileNotFoundError(f"WhisperX JSON not found in {job_output_dir}.")
                 json_output_path = found_json

            # --- Add Emotion Analysis ---
            structured_data = self.transcription.convert_json_to_structured(json_output_path)

            # --- Save Structured Transcript ---
            structured_path = os.path.join(job_output_dir, STRUCTURED_TRANSCRIPT_NAME)
            with open(structured_path, "w", encoding="utf-8") as f_json_out: # Renamed file handle
                json.dump(structured_data, f_json_out, indent=2, ensure_ascii=False)

            # --- Extract Speaker Previews for UI ---
            speaker_previews = self._get_speaker_previews(structured_data)
            num_speakers = len(speaker_previews)
            log_info(f"Job {job_id}: Found {num_speakers} unique speakers.")

            # --- Return intermediate results ---
            status_message = f"✅ Job {job_id} processed. Found {num_speakers} speakers. Ready for relabeling below."
            return status_message, structured_path, speaker_previews

        except Exception as e:
            error_message = f"❌ Job {job_id} failed during initial processing: {e}"
            log_error(error_message); log_error(traceback.format_exc())

            # Write error to log file safely
            if log_file_handle:
                try:
                    log_file_handle.write(f"\n\n--- ERROR ---\n{error_message}\n{traceback.format_exc()}\n")
                except Exception as log_e:
                    print(f"Additionally failed to write error to log file: {log_e}")

            return error_message, None, [] # Return error status and empty results

        finally:
            log_info(f"Job {job_id}: Cleaning up temporary files...")

            # Close log file handle safely
            if log_file_handle:
                try:
                    log_file_handle.close()
                except Exception:
                    pass # Ignore closing errors

            # Remove temp directory safely
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except OSError as e:
                    log_error(f"Error removing temp directory {temp_dir}: {e}")


    def _calculate_emotion_summary(self, segments, include_timeline=False):
        """Calculates summary statistics based on structured segments."""
        # (Keep this method as previously defined - it uses EMO_VAL correctly)
        speaker_stats = defaultdict(list); segment_details = defaultdict(list)
        for s in segments:
             speaker_key = str(s.get('speaker', 'unknown')); emotion = s.get('emotion', 'unknown')
             speaker_stats[speaker_key].append(emotion)
             if include_timeline: segment_details[speaker_key].append({"time": s.get("start", 0), "emotion": emotion})
        summarized = {}
        for spk, emos in speaker_stats.items():
            if not emos: continue
            transitions = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i-1])
            dominant = Counter(emos).most_common(1)[0][0] if emos else "unknown"
            emo_numeric = [EMO_VAL.get(e, 0) for e in emos] # Uses EMO_VAL here
            volatility = stdev(emo_numeric) if len(emo_numeric) > 1 else 0
            avg_emotion = mean(emo_numeric) if emo_numeric else 0
            summary_item = {'total_segments': len(emos), 'emotion_transitions': transitions,'dominant_emotion': dominant, 'emotion_volatility': round(volatility, 3),'emotion_score_mean': round(avg_emotion, 3)}
            if include_timeline and spk in segment_details:
                 timeline = sorted(segment_details[spk], key=lambda x: x.get("time", 0))
                 summary_item['emotion_timeline'] = timeline
            summarized[spk] = summary_item
        return summarized

    def save_emotion_summary(self, speaker_stats, output_dir, file_suffix):
        """Saves the calculated emotion summary to CSV and JSON."""
        # (Keep this method as previously defined)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{EMOTION_SUMMARY_CSV_NAME.replace('.csv', '')}_{file_suffix}.csv")
        json_path = os.path.join(output_dir, f"{EMOTION_SUMMARY_JSON_NAME.replace('.json', '')}_{file_suffix}.json")
        log_info(f"Saving emotion summary: {json_path}, {csv_path}")
        with open(json_path, "w", encoding="utf-8") as jf: json.dump(speaker_stats, jf, indent=2, ensure_ascii=False)
        with open(csv_path, "w", newline='', encoding="utf-8") as cf:
            headers = ["speaker"];
            if speaker_stats:
                 stat_keys = list(list(speaker_stats.values())[0].keys());
                 if 'emotion_timeline' in stat_keys: stat_keys.remove('emotion_timeline')
                 headers.extend(sorted(stat_keys))
            else: headers.extend(['total_segments', 'emotion_transitions', 'dominant_emotion', 'emotion_volatility', 'emotion_score_mean'])
            writer = csv.DictWriter(cf, fieldnames=headers, extrasaction='ignore'); writer.writeheader()
            for speaker, stats in speaker_stats.items(): row_data = {'speaker': speaker}; row_data.update(stats); writer.writerow(row_data)
        return csv_path, json_path

    def relabel_and_finalize(self, structured_json_path, edited_dataframe_data):
        """
        Applies speaker labels from edited Dataframe data, recalculates summary,
        generates plots, and zips final results.
        """
        log_info(f"Starting relabeling and finalization for: {structured_json_path}")
        output_dir = os.path.dirname(structured_json_path)
        base_name = os.path.basename(output_dir)

        try:
            # --- Create Speaker Mapping ---
            speaker_mapping = {}
            # (Keep parsing logic for dataframe data as before)
            if isinstance(edited_dataframe_data, pd.DataFrame):
                 for index, row in edited_dataframe_data.iterrows():
                     original_speaker = str(row.get('Speaker ID'))
                     new_label = str(row.get('Enter Label Here', '')).strip()
                     if original_speaker: speaker_mapping[original_speaker] = new_label if new_label else original_speaker
            elif isinstance(edited_dataframe_data, list):
                 print("WARN: Received list data from DataFrame, attempting basic parsing.")
                 if len(edited_dataframe_data) > 0:
                    if isinstance(edited_dataframe_data[0], dict):
                         for row in edited_dataframe_data:
                             original_speaker = str(row.get('Speaker ID'))
                             new_label = str(row.get('Enter Label Here', '')).strip()
                             if original_speaker: speaker_mapping[original_speaker] = new_label if new_label else original_speaker
                    elif isinstance(edited_dataframe_data[0], list):
                        for row in edited_dataframe_data:
                             if len(row) >= 3:
                                 original_speaker = str(row[0]); new_label = str(row[2]).strip()
                                 if original_speaker: speaker_mapping[str(original_speaker)] = new_label if new_label else str(original_speaker)
            else: log_warning("Received unexpected data format from DataFrame. Skipping relabeling.")

            log_info(f"Applying speaker mapping: {speaker_mapping}")

            # --- Apply Mapping and Save Relabeled JSON ---
            with open(structured_json_path, "r", encoding="utf-8") as jf: data = json.load(jf) # Use jf here
            modified_count = 0
            if speaker_mapping:
                 for seg in data:
                    original_speaker = str(seg.get("speaker", "unknown"))
                    if original_speaker in speaker_mapping:
                        mapped_label = speaker_mapping[original_speaker]
                        if seg["speaker"] != mapped_label:
                            seg["speaker_original"] = original_speaker; seg["speaker"] = mapped_label
                            modified_count += 1
            log_info(f"Applied labels to {modified_count} segments.")

            # --- FIX START: Correct filename construction ---
            # Construct filename correctly using the suffix
            base_structured_name = STRUCTURED_TRANSCRIPT_NAME.replace('.json', '')
            relabeled_json_filename = f"{base_structured_name}{RELABELED_SUFFIX}.json" # Use constant RELABELED_SUFFIX
            relabeled_json_path = os.path.join(output_dir, relabeled_json_filename)
            # --- FIX END ---

            log_info(f"Saving final transcript data to: {relabeled_json_path}")
            # Use a different file handle name here to avoid confusion with 'f' potentially used elsewhere
            with open(relabeled_json_path, "w", encoding="utf-8") as f_relabeled:
                 json.dump(data, f_relabeled, indent=2, ensure_ascii=False)

            # --- Recalculate Summary ---
            log_info("Recalculating final emotion summary...")
            final_summary = self._calculate_emotion_summary(data, include_timeline=True)

            # --- Save Final Summary ---
            final_summary_suffix = base_name + RELABELED_SUFFIX # Use constant
            csv_path, json_path = self.save_emotion_summary(final_summary, output_dir, final_summary_suffix)
            log_info(f"Final emotion summary saved.")

            # --- Generate Plots ---
            log_info("Generating emotion plots...")
            plot_filenames = []
            try:
                plot_suffix = final_summary_suffix
                plot_filenames = generate_all_plots(final_summary, output_dir, plot_suffix)
                log_info(f"Generated {len(plot_filenames)} plot files.")
            except Exception as plot_err:
                 log_error(f"Failed during plot generation: {plot_err}\n{traceback.format_exc()}")

            # --- Package Final Results ---
            final_zip_path = os.path.join(os.path.dirname(output_dir), f"{base_name}{FINAL_ZIP_SUFFIX}") # Use constant
            log_info(f"Creating final results ZIP: {final_zip_path}")
            with zipfile.ZipFile(final_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf: # Use zipf
                zipf.write(relabeled_json_path, arcname=os.path.basename(relabeled_json_path))
                zipf.write(csv_path, arcname=os.path.basename(csv_path))
                zipf.write(json_path, arcname=os.path.basename(json_path))
                plots_arc_dir = "plots"
                for plot_file in plot_filenames:
                     if os.path.exists(plot_file): zipf.write(plot_file, arcname=os.path.join(plots_arc_dir, os.path.basename(plot_file)))
                     else: log_error(f"Plot file {plot_file} not found for zipping.")
                log_path_orig = os.path.join(output_dir, LOG_FILE_NAME) # Corrected path for log file
                if os.path.exists(log_path_orig): zipf.write(log_path_orig, arcname=LOG_FILE_NAME)

            status_message = f"✅ Relabeling and final processing complete for {base_name}."
            return final_zip_path, status_message

        except Exception as e:
            error_message = f"❌ Failed during relabeling/finalization for {structured_json_path}: {e}"
            log_error(error_message); log_error(traceback.format_exc())
            return None, error_message