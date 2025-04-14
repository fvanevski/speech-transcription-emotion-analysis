# core/pipeline.py
import os
import json
import csv
import zipfile
import uuid
import shutil
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from statistics import mean, stdev

from core.transcription import Transcription
from core.file_management import FileManager
from core.logging import log_info, log_error
# Assuming plotting.py exists and has generate_all_plots
from core.plotting import generate_all_plots

# Define emotion values for quantitative summary
EMO_VAL = {'joy': 1, 'neutral': 0, 'sadness': -1, 'anger': -2, 'surprise': 0.5, 'fear': -1.5,
           'unknown': 0, 'analysis_skipped': 0, 'analysis_failed': 0, 'no_text': 0 }

# Define constants
LOG_FILE_NAME = "process_log.txt"
STRUCTURED_TRANSCRIPT_NAME = "structured_transcript.json"
RELABELED_JSON_SUFFIX = "_relabeled.json"
EMOTION_SUMMARY_CSV_NAME = "emotion_summary.csv"
EMOTION_SUMMARY_JSON_NAME = "emotion_summary.json"
FINAL_ZIP_SUFFIX = "_final_bundle.zip"

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.transcription = Transcription(config)
        self.file_manager = FileManager(config)

    def _run_ffprobe_duration_check(self, audio_path):
        # (Keep this method as previously defined)
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
        except FileNotFoundError:
            log_error("ffprobe command not found. Skipping duration check. Ensure ffmpeg (which includes ffprobe) is installed and in PATH.")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"ffprobe failed for {audio_path}: {e}")
            raise RuntimeError(f"ffprobe check failed: {e.stderr}")
        except ValueError as e:
            log_error(f"ffprobe output parsing error or duration check failed: {e}")
            raise e
        except Exception as e:
            log_error(f"Unexpected error during ffprobe check: {e}")
            raise RuntimeError(f"Duration check failed unexpectedly: {e}")


    def process_audio(self, input_source):
        """
        Performs initial processing: download (if URL), transcription, diarization,
        initial emotion analysis.
        Returns status message, path to the structured transcript JSON, and list of speaker IDs found.
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = str(uuid.uuid4())[:8]
        base_dir = self.config.get("output_dir", "output")
        # Create a job-specific directory to hold all related files
        job_output_dir = os.path.join(base_dir, f"job_{session_id}_{job_id}")
        temp_dir = os.path.join(job_output_dir, "temp")
        # Note: whisperx output now goes directly into the job_output_dir
        log_path = os.path.join(job_output_dir, LOG_FILE_NAME)

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(job_output_dir, exist_ok=True) # Ensure main job dir exists

        log_file_handle = None
        structured_path = None # Define structured_path outside try

        try:
            log_file_handle = open(log_path, "w", encoding="utf-8")
            log_file_handle.write(f"📄 Processing Job ID: {job_id}\nSession ID: {session_id}\nStarted: {datetime.now()}\n\n")
            log_file_handle.flush()
            log_info(f"Job {job_id} started. Output dir: {job_output_dir}")

            # --- Handle Input ---
            audio_path = None
            # (Keep input handling logic as before)
            if isinstance(input_source, str) and input_source.startswith(('http://', 'https://')):
                 log_info(f"Job {job_id}: Input is a URL. Downloading...")
                 audio_path = self.transcription.download_audio_from_youtube(input_source, temp_dir, log_file_handle, session_id)
                 log_info(f"Job {job_id}: YouTube audio downloaded to {audio_path}")
            elif isinstance(input_source, str) and os.path.exists(input_source):
                 log_info(f"Job {job_id}: Input is a file path: {input_source}")
                 input_ext = os.path.splitext(input_source)[-1]
                 # Use a more predictable name in temp if needed, or pass original path if whisperx handles it
                 temp_audio_path = os.path.join(temp_dir, f"input_audio{input_ext}")
                 shutil.copy(input_source, temp_audio_path)
                 audio_path = temp_audio_path # Use the path in temp dir
                 log_info(f"Job {job_id}: Copied input file to {audio_path}")
            else:
                 raise ValueError("Invalid input source provided. Expecting YouTube URL or valid file path.")

            # --- Duration Check ---
            self._run_ffprobe_duration_check(audio_path)

            # --- Run WhisperX (Output goes to job_output_dir) ---
            log_info(f"Job {job_id}: Running WhisperX (with --diarize) on {audio_path}...")
            # Pass job_output_dir as the output directory for whisperx results
            self.transcription.run_whisperx(audio_path, job_output_dir, log_file_handle, session_id)
            log_info(f"Job {job_id}: WhisperX finished.")

            # --- Find WhisperX Output JSON ---
            # (Keep logic to find json_output_path, now looking in job_output_dir)
            audio_filename_stem = Path(audio_path).stem
            expected_json_filename = f"{audio_filename_stem}.json"
            json_output_path = os.path.join(job_output_dir, expected_json_filename)
            log_info(f"Job {job_id}: Looking for WhisperX output: {json_output_path}")
            if not os.path.exists(json_output_path):
                 # Fallback search
                 found_json = None
                 for f in os.listdir(job_output_dir):
                     if f.endswith(".json") and f != EMOTION_SUMMARY_JSON_NAME: # Check against summary name constant
                         found_json = os.path.join(job_output_dir, f)
                         log_info(f"Job {job_id}: Found JSON file via listing: {found_json}")
                         break
                 if not found_json:
                     raise FileNotFoundError(f"WhisperX output JSON not found in {job_output_dir}. Expected name like {expected_json_filename}.")
                 json_output_path = found_json # Use the found file

            # --- Add Emotion Analysis ---
            log_info(f"Job {job_id}: Adding emotion analysis via convert_json_to_structured...")
            structured_data = self.transcription.convert_json_to_structured(json_output_path)
            log_info(f"Job {job_id}: Emotion analysis added.")

            # --- Save Structured Transcript (within job_output_dir) ---
            structured_path = os.path.join(job_output_dir, STRUCTURED_TRANSCRIPT_NAME)
            log_info(f"Job {job_id}: Saving structured transcript to {structured_path}")
            with open(structured_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)

            # --- Extract Speaker IDs ---
            speaker_ids = sorted(list(set(str(seg.get('speaker', 'unknown')) for seg in structured_data)))
            log_info(f"Job {job_id}: Found Speaker IDs: {speaker_ids}")

            # --- Return intermediate results needed for UI ---
            status_message = f"✅ Job {job_id} processed. Found speakers: {', '.join(speaker_ids)}. Ready for relabeling."
            # Return path to the structured JSON and the speaker list
            return status_message, structured_path, speaker_ids

        except Exception as e:
            error_message = f"❌ Job {job_id} failed during initial processing: {e}"
            log_error(error_message)
            log_error(traceback.format_exc())
            if log_file_handle:
                try: log_file_handle.write(f"\n\n--- ERROR ---\n{error_message}\n{traceback.format_exc()}\n")
                except Exception: pass
            # Return error status and None for path/speakers
            return error_message, None, None

        finally:
            log_info(f"Job {job_id}: Cleaning up temporary files...")
            if log_file_handle:
                try: log_file_handle.close()
                except Exception: pass
            # Cleanup only temp processing dir, keep main job output dir
            if os.path.exists(temp_dir):
                 try: shutil.rmtree(temp_dir)
                 except OSError as e: log_error(f"Error removing temp directory {temp_dir}: {e}")

    def _calculate_emotion_summary(self, segments, include_timeline=False):
        # (Keep this method as previously defined)
        speaker_stats = defaultdict(list)
        segment_details = defaultdict(list)
        for s in segments:
             speaker_key = str(s.get('speaker', 'unknown'))
             emotion = s.get('emotion', 'unknown')
             speaker_stats[speaker_key].append(emotion)
             if include_timeline:
                 segment_details[speaker_key].append({"time": s.get("start", 0), "emotion": emotion})
        summarized = {}
        for spk, emos in speaker_stats.items():
            if not emos: continue
            transitions = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i-1])
            dominant = Counter(emos).most_common(1)[0][0] if emos else "unknown"
            emo_numeric = [EMO_VAL.get(e, 0) for e in emos]
            volatility = stdev(emo_numeric) if len(emo_numeric) > 1 else 0
            avg_emotion = mean(emo_numeric) if emo_numeric else 0
            summary_item = {
                'total_segments': len(emos), 'emotion_transitions': transitions,
                'dominant_emotion': dominant, 'emotion_volatility': round(volatility, 3),
                'emotion_score_mean': round(avg_emotion, 3),
            }
            if include_timeline and spk in segment_details:
                 timeline = sorted(segment_details[spk], key=lambda x: x.get("time", 0))
                 summary_item['emotion_timeline'] = timeline
            summarized[spk] = summary_item
        return summarized

    def save_emotion_summary(self, speaker_stats, output_dir, file_suffix):
        # (Keep this method as previously defined)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{EMOTION_SUMMARY_CSV_NAME.replace('.csv', '')}_{file_suffix}.csv")
        json_path = os.path.join(output_dir, f"{EMOTION_SUMMARY_JSON_NAME.replace('.json', '')}_{file_suffix}.json")
        log_info(f"Saving emotion summary: {json_path}, {csv_path}")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(speaker_stats, jf, indent=2, ensure_ascii=False)
        with open(csv_path, "w", newline='', encoding="utf-8") as cf:
            headers = ["speaker"]
            if speaker_stats:
                 stat_keys = list(list(speaker_stats.values())[0].keys())
                 if 'emotion_timeline' in stat_keys: stat_keys.remove('emotion_timeline')
                 headers.extend(sorted(stat_keys))
            else: headers.extend(['total_segments', 'emotion_transitions', 'dominant_emotion', 'emotion_volatility', 'emotion_score_mean'])
            writer = csv.DictWriter(cf, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for speaker, stats in speaker_stats.items():
                 row_data = {'speaker': speaker}; row_data.update(stats)
                 writer.writerow(row_data)
        return csv_path, json_path

    def relabel_and_finalize(self, structured_json_path, speaker_mapping):
        """
        Applies speaker labels from a mapping dict, recalculates summary,
        generates plots, and zips final results.
        """
        log_info(f"Starting relabeling and finalization for: {structured_json_path}")
        output_dir = os.path.dirname(structured_json_path) # Use the same job directory
        # Extract base name for suffixes (e.g., job_id)
        base_name = os.path.basename(output_dir) # e.g., job_20250414_113407_47f47dfa

        try:
            # --- Apply Mapping and Save Relabeled JSON ---
            log_info(f"Applying speaker mapping: {speaker_mapping}")
            with open(structured_json_path, "r", encoding="utf-8") as jf:
                data = json.load(f) # Assumes data is list of segments

            if not isinstance(speaker_mapping, dict):
                 raise TypeError("speaker_mapping must be a dictionary.")

            modified_count = 0
            for seg in data:
                original_speaker = str(seg.get("speaker", "unknown"))
                # Apply mapping only if speaker is in the map and mapped value is not empty
                if original_speaker in speaker_mapping and speaker_mapping[original_speaker]:
                    seg["speaker_original"] = original_speaker
                    seg["speaker"] = speaker_mapping[original_speaker]
                    modified_count += 1
            log_info(f"Applied labels to {modified_count} segments.")

            # Save the relabeled transcript data
            relabeled_json_path = os.path.join(output_dir, STRUCTURED_TRANSCRIPT_NAME.replace('.json', RELABELED_SUFFIX)) # Consistent naming
            log_info(f"Saving relabeled transcript to: {relabeled_json_path}")
            with open(relabeled_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # --- Recalculate Emotion Summary (include timeline) ---
            log_info("Recalculating emotion summary for relabeled data...")
            summarized_relabeled = self._calculate_emotion_summary(data, include_timeline=True)

            # --- Save Relabeled Summary ---
            final_summary_suffix = base_name + RELABELED_SUFFIX # e.g., job_..._relabeled
            csv_path, json_path = self.save_emotion_summary(summarized_relabeled, output_dir, final_summary_suffix)
            log_info(f"Relabeled emotion summary saved.")

            # --- Generate Plots ---
            log_info("Generating emotion plots...")
            plot_filenames = []
            try:
                # Suffix for plot filenames should be unique
                plot_suffix = final_summary_suffix
                plot_filenames = generate_all_plots(summarized_relabeled, output_dir, plot_suffix)
                log_info(f"Generated {len(plot_filenames)} plot files.")
            except Exception as plot_err:
                 log_error(f"Failed during plot generation: {plot_err}\n{traceback.format_exc()}")
                 # Continue without plots

            # --- Package Final Results ---
            final_zip_path = os.path.join(os.path.dirname(output_dir), f"{base_name}{FINAL_ZIP_SUFFIX}") # Place zip outside results folder
            log_info(f"Creating final results ZIP: {final_zip_path}")
            with zipfile.ZipFile(final_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(relabeled_json_path, arcname=os.path.basename(relabeled_json_path))
                zipf.write(csv_path, arcname=os.path.basename(csv_path))
                zipf.write(json_path, arcname=os.path.basename(json_path))
                # Add plots if generated
                plots_arc_dir = "plots" # Subdirectory in zip
                for plot_file in plot_filenames:
                     if os.path.exists(plot_file):
                         zipf.write(plot_file, arcname=os.path.join(plots_arc_dir, os.path.basename(plot_file)))
                     else: log_error(f"Plot file {plot_file} not found for zipping.")
                # Add log file
                log_path = os.path.join(output_dir, LOG_FILE_NAME) # Original log path
                if os.path.exists(log_path):
                     zipf.write(log_path, arcname=LOG_FILE_NAME)


            status_message = f"✅ Relabeling and final processing complete for {base_name}."
            return final_zip_path, status_message

        except Exception as e:
            error_message = f"❌ Failed during relabeling/finalization for {structured_json_path}: {e}"
            log_error(error_message)
            log_error(traceback.format_exc())
            return None, error_message

    # Removed generate_csv_template and apply_labels_from_csv methods