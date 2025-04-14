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
# Removed: from core.diarization import Diarization
# Removed: from core.emotion_analysis import EmotionAnalysis
from core.file_management import FileManager
from core.logging import log_info, log_error
from core.plotting import generate_all_plots # <--- IMPORT PLOTTING FUNCTION

# Define emotion values for quantitative summary (consistent with plotting.py)
EMO_VAL = {'joy': 1, 'neutral': 0, 'sadness': -1, 'anger': -2, 'surprise': 0.5, 'fear': -1.5,
           # Add defaults for statuses possibly returned by transcription.py's emotion analysis
           'unknown': 0, 'analysis_skipped': 0, 'analysis_failed': 0, 'no_text': 0
           }

# Define constants for clarity
LOG_FILE_NAME = "process_log.txt"
STRUCTURED_TRANSCRIPT_NAME = "structured_transcript.json"
EMOTION_SUMMARY_CSV_NAME = "emotion_summary.csv"
EMOTION_SUMMARY_JSON_NAME = "emotion_summary.json"
RELABELED_SUFFIX = "_relabeled"
ZIP_BUNDLE_SUFFIX = "_bundle.zip"

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
        # (Keep this method largely as previously defined, ensure it saves structured_path)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = str(uuid.uuid4())[:8]
        base_dir = self.config.get("output_dir", "output")
        job_base_path = os.path.join(base_dir, f"job_{session_id}_{job_id}")
        temp_dir = os.path.join(job_base_path, "temp")
        output_dir = os.path.join(job_base_path, "results") # Define output dir for results
        log_path = os.path.join(job_base_path, LOG_FILE_NAME)

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True) # Ensure results dir is created

        log_file_handle = None

        try:
            log_file_handle = open(log_path, "w", encoding="utf-8")
            log_file_handle.write(f"📄 Transcription Job ID: {job_id}\nSession ID: {session_id}\nStarted: {datetime.now()}\n\n")
            log_file_handle.flush()
            log_info(f"Job {job_id} started. Temp: {temp_dir}, Output: {output_dir}")

            # --- Handle Input ---
            # (Keep input handling logic as before)
            audio_path = None
            if isinstance(input_source, str) and input_source.startswith(('http://', 'https://')):
                 log_info(f"Job {job_id}: Input is a URL. Downloading...")
                 audio_path = self.transcription.download_audio_from_youtube(input_source, temp_dir, log_file_handle, session_id)
                 log_info(f"Job {job_id}: YouTube audio downloaded to {audio_path}")
            elif isinstance(input_source, str) and os.path.exists(input_source):
                 log_info(f"Job {job_id}: Input is a file path: {input_source}")
                 input_ext = os.path.splitext(input_source)[-1]
                 audio_path = os.path.join(temp_dir, f"audio_input{input_ext}")
                 shutil.copy(input_source, audio_path)
                 log_info(f"Job {job_id}: Copied input file to {audio_path}")
            else:
                raise ValueError("Invalid input source provided. Expecting YouTube URL or valid file path.")


            # --- Duration Check ---
            self._run_ffprobe_duration_check(audio_path)

            # --- Run WhisperX ---
            log_info(f"Job {job_id}: Running WhisperX (with --diarize) on {audio_path}...")
            self.transcription.run_whisperx(audio_path, output_dir, log_file_handle, session_id) # Writes JSON to output_dir
            log_info(f"Job {job_id}: WhisperX finished.")

            # --- Find WhisperX Output JSON ---
            # (Keep logic to find json_output_path as before)
            audio_filename_stem = Path(audio_path).stem
            expected_json_filename = f"{audio_filename_stem}.json"
            json_output_path = os.path.join(output_dir, expected_json_filename)
            log_info(f"Job {job_id}: Looking for WhisperX output: {json_output_path}")
            if not os.path.exists(json_output_path):
                 found_json = None
                 for f in os.listdir(output_dir):
                     if f.endswith(".json") and f != EMOTION_SUMMARY_JSON_NAME:
                         found_json = os.path.join(output_dir, f)
                         log_info(f"Job {job_id}: Found JSON file via listing: {found_json}")
                         break
                 if not found_json:
                     raise FileNotFoundError(f"WhisperX output JSON not found in {output_dir}. Expected name like {expected_json_filename}.")
                 json_output_path = found_json


            # --- Add Emotion Analysis ---
            log_info(f"Job {job_id}: Adding emotion analysis via convert_json_to_structured...")
            structured_data = self.transcription.convert_json_to_structured(json_output_path)
            log_info(f"Job {job_id}: Emotion analysis added.")

            # --- Save Structured Transcript ---
            # Ensure it's saved within the job's output directory
            structured_path = os.path.join(output_dir, STRUCTURED_TRANSCRIPT_NAME)
            log_info(f"Job {job_id}: Saving structured transcript to {structured_path}")
            with open(structured_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)

            # --- Generate Initial Emotion Summary ---
            # Note: Plots are usually generated *after* relabeling
            log_info(f"Job {job_id}: Generating initial emotion summary...")
            initial_summary_stats = self._calculate_emotion_summary(structured_data) # Use helper
            # Save initial summary (optional, might only need relabeled summary)
            self.save_emotion_summary(initial_summary_stats, output_dir, job_id + "_initial")
            log_info(f"Job {job_id}: Initial emotion summary saved.")


            # --- Package Initial Results (Transcript + Initial Summary) ---
            zip_path = os.path.join(base_dir, f"results_{session_id}_{job_id}{ZIP_BUNDLE_SUFFIX}")
            log_info(f"Job {job_id}: Creating results ZIP file (initial): {zip_path}")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(structured_path, arcname=STRUCTURED_TRANSCRIPT_NAME)
                # Add initial summary files if saved
                initial_csv, initial_json = self.save_emotion_summary(initial_summary_stats, output_dir, job_id + "_initial")
                zipf.write(initial_csv, arcname=os.path.basename(initial_csv))
                zipf.write(initial_json, arcname=os.path.basename(initial_json))
                # Close log file handle before adding to zip
                if log_file_handle:
                    log_file_handle.close()
                    log_file_handle = None
                zipf.write(log_path, arcname=LOG_FILE_NAME)

            log_info(f"Job {job_id}: Initial results ZIP created.")
            return zip_path, f"✅ Job {job_id} completed. Transcript ready for speaker labeling."


        except Exception as e:
            error_message = f"❌ Job {job_id} failed during processing: {e}"
            log_error(error_message)
            # Add traceback for detailed debugging
            log_error(traceback.format_exc())
            if log_file_handle:
                try:
                    log_file_handle.write(f"\n\n--- ERROR ---\n{error_message}\n{traceback.format_exc()}\n")
                except Exception as log_e:
                    print(f"Additionally failed to write error to log file: {log_e}")
            return None, error_message

        finally:
            log_info(f"Job {job_id}: Cleaning up...")
            if log_file_handle:
                try: log_file_handle.close()
                except Exception: pass # Ignore closing errors
            # Cleanup temp dir
            if os.path.exists(temp_dir):
                 try: shutil.rmtree(temp_dir)
                 except OSError as e: log_error(f"Error removing temp directory {temp_dir}: {e}")
            # Note: output_dir is kept as it contains results

    def _calculate_emotion_summary(self, segments, include_timeline=False): # Added option
        """Calculates summary statistics based on structured segments."""
        # (Keep logic as before, potentially add include_timeline logic if needed by plotting)
        speaker_stats = defaultdict(list)
        segment_details = defaultdict(list) # To store more details per speaker

        for s in segments:
             speaker_key = str(s.get('speaker', 'unknown'))
             emotion = s.get('emotion', 'unknown')
             speaker_stats[speaker_key].append(emotion)
             # Store details needed for timeline/score calculation if include_timeline is True
             if include_timeline:
                 segment_details[speaker_key].append({
                     "time": s.get("start", 0), # Need start time for timeline
                     "emotion": emotion
                 })

        summarized = {}
        for spk, emos in speaker_stats.items():
            if not emos: continue

            transitions = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i-1])
            dominant = Counter(emos).most_common(1)[0][0] if emos else "unknown"

            # Calculate volatility/score mean
            emo_numeric = [EMO_VAL.get(e, 0) for e in emos]
            volatility = stdev(emo_numeric) if len(emo_numeric) > 1 else 0
            avg_emotion = mean(emo_numeric) if emo_numeric else 0

            summary_item = {
                'total_segments': len(emos),
                'emotion_transitions': transitions,
                'dominant_emotion': dominant,
                'emotion_volatility': round(volatility, 3),
                'emotion_score_mean': round(avg_emotion, 3),
            }

            # Add timeline if requested and available
            if include_timeline and spk in segment_details:
                 # Sort timeline points by time
                 timeline = sorted(segment_details[spk], key=lambda x: x.get("time", 0))
                 summary_item['emotion_timeline'] = timeline

            summarized[spk] = summary_item
        return summarized


    # --- Post-processing Methods ---

    def save_emotion_summary(self, speaker_stats, output_dir, job_id_suffix):
        # (Keep logic as previously defined, ensure headers match calculated stats)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{EMOTION_SUMMARY_CSV_NAME.replace('.csv', '')}_{job_id_suffix}.csv")
        json_path = os.path.join(output_dir, f"{EMOTION_SUMMARY_JSON_NAME.replace('.json', '')}_{job_id_suffix}.json")
        log_info(f"Saving emotion summary: {json_path}, {csv_path}")

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(speaker_stats, jf, indent=2, ensure_ascii=False)

        with open(csv_path, "w", newline='', encoding="utf-8") as cf:
            # Dynamically get headers from the first item
            headers = ["speaker"]
            if speaker_stats:
                 # Get keys, remove timeline if present as it doesn't fit well in CSV row
                 stat_keys = list(list(speaker_stats.values())[0].keys())
                 if 'emotion_timeline' in stat_keys:
                     stat_keys.remove('emotion_timeline')
                 headers.extend(sorted(stat_keys)) # Sort for consistent column order
            else:
                 # Fallback default headers if no stats calculated
                 headers.extend(['total_segments', 'emotion_transitions', 'dominant_emotion', 'emotion_volatility', 'emotion_score_mean'])

            writer = csv.DictWriter(cf, fieldnames=headers, extrasaction='ignore') # Ignore extra fields like timeline
            writer.writeheader()
            for speaker, stats in speaker_stats.items():
                 row_data = {'speaker': speaker}
                 row_data.update(stats)
                 writer.writerow(row_data)
        return csv_path, json_path


    def generate_csv_template(self, json_file_obj):
        # (Keep logic as previously defined)
        if not json_file_obj or not hasattr(json_file_obj, 'name'):
            log_error("generate_csv_template: Invalid input file object provided.")
            raise ValueError("Please upload a valid structured transcript JSON file.")
        json_file_path = json_file_obj.name
        log_info(f"Generating CSV template from: {json_file_path}")
        try:
             with open(json_file_path, "r", encoding="utf-8") as f:
                # Expecting the output of process_audio -> structured_transcript.json
                data = json.load(f) # Assumes data is list of segments
             speakers = {}
             for seg in data:
                spk = str(seg.get("speaker", "unknown"))
                if spk not in speakers:
                     preview = seg.get("text", "")[:60].replace("\n", " ").replace(",", ";")
                     speakers[spk] = preview
             # Save template relative to input json? Or dedicated output? Assume relative.
             base_dir = os.path.dirname(json_file_path)
             # Construct name based on the standard output name if possible
             template_name = STRUCTURED_TRANSCRIPT_NAME.replace(".json", "_speaker_template.csv")
             template_path = os.path.join(base_dir, template_name)
             log_info(f"Saving CSV template to: {template_path}")
             with open(template_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(["speaker", "label", "preview"])
                for spk, example in sorted(speakers.items()):
                     writer.writerow([spk, "", example])
             return template_path
        except FileNotFoundError:
             log_error(f"generate_csv_template: Input JSON file not found at {json_file_path}")
             raise FileNotFoundError("The uploaded JSON transcript file could not be found.")
        except json.JSONDecodeError:
             log_error(f"generate_csv_template: Error decoding JSON from {json_file_path}")
             raise ValueError("The uploaded file is not a valid JSON transcript.")
        except Exception as e:
             log_error(f"generate_csv_template: Unexpected error: {e}\n{traceback.format_exc()}")
             raise RuntimeError(f"Failed to generate CSV template: {e}")


    def apply_labels_from_csv(self, json_file_obj, csv_file_obj):
        """Applies speaker labels, recalculates summary, generates plots, and zips results."""
        # (Keep validation and mapping logic as before)
        if not json_file_obj or not hasattr(json_file_obj, 'name'):
             raise ValueError("Please upload a valid structured transcript JSON file.")
        if not csv_file_obj or not hasattr(csv_file_obj, 'name'):
             raise ValueError("Please upload a valid speaker label CSV file.")
        json_file_path = json_file_obj.name
        csv_file_path = csv_file_obj.name
        log_info(f"Applying labels from {csv_file_path} to {json_file_path}")

        try:
            # --- Read Mapping ---
            mapping = {}
            with open(csv_file_path, "r", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                if not all(col in reader.fieldnames for col in ["speaker", "label"]):
                    raise ValueError("CSV file must contain 'speaker' and 'label' columns.")
                for row in reader:
                    original_speaker = row.get("speaker")
                    new_label = row.get("label", "").strip()
                    if original_speaker:
                        mapping[original_speaker] = new_label if new_label else original_speaker
            log_info(f"Loaded label mapping: {mapping}")

            # --- Apply Mapping and Save Relabeled JSON ---
            with open(json_file_path, "r", encoding="utf-8") as jf:
                data = json.load(f)
            for seg in data:
                original_speaker = str(seg.get("speaker", "unknown"))
                seg["speaker_original"] = original_speaker
                seg["speaker"] = mapping.get(original_speaker, original_speaker)
            output_dir = os.path.dirname(json_file_path) # Save results in same dir as input json
            base_name = os.path.basename(json_file_path).replace(STRUCTURED_TRANSCRIPT_NAME, "") # Get base job name
            relabeled_json_path = os.path.join(output_dir, f"{base_name}{STRUCTURED_TRANSCRIPT_NAME.replace('.json', '')}{RELABELED_SUFFIX}.json")
            log_info(f"Saving relabeled transcript to: {relabeled_json_path}")
            with open(relabeled_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # --- Recalculate Emotion Summary (include timeline data) ---
            log_info("Recalculating emotion summary for relabeled data...")
            # Pass include_timeline=True to get data needed for plots
            summarized_relabeled = self._calculate_emotion_summary(data, include_timeline=True)

            # --- Save Relabeled Summary ---
            job_id_suffix = f"{base_name}{RELABELED_SUFFIX}"
            csv_path, json_path = self.save_emotion_summary(summarized_relabeled, output_dir, job_id_suffix)
            log_info(f"Relabeled emotion summary saved: {csv_path}, {json_path}")

            # --- *** Generate Plots *** ---
            log_info("Generating emotion plots...")
            plot_filenames = []
            try:
                # Call the plotting function from the plotting module
                plot_filenames = generate_all_plots(summarized_relabeled, output_dir, job_id_suffix)
                log_info(f"Generated {len(plot_filenames)} plot files.")
            except Exception as plot_err:
                 # Log error but continue to package other results
                 log_error(f"Failed during plot generation: {plot_err}\n{traceback.format_exc()}")

            # --- Package Final Results (Relabeled JSON, Summary, Plots) ---
            zip_path = relabeled_json_path.replace(".json", ZIP_BUNDLE_SUFFIX)
            log_info(f"Creating final results ZIP: {zip_path}")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add relabeled transcript
                zipf.write(relabeled_json_path, arcname=os.path.basename(relabeled_json_path))
                # Add relabeled summaries
                zipf.write(csv_path, arcname=os.path.basename(csv_path))
                zipf.write(json_path, arcname=os.path.basename(json_path))
                # Add generated plots
                for plot_file in plot_filenames:
                    if os.path.exists(plot_file):
                        zipf.write(plot_file, arcname=os.path.join("plots", os.path.basename(plot_file))) # Put plots in a subfolder
                    else:
                        log_error(f"Plot file {plot_file} not found for zipping.")
                # Optionally include the mapping CSV used
                # zipf.write(csv_file_path, arcname="speaker_mapping_applied.csv")

            return zip_path # Return path to the final zip file containing plots

        except FileNotFoundError as e:
             log_error(f"apply_labels_from_csv: File not found: {e}")
             raise FileNotFoundError(f"Input file not found: {e}")
        except json.JSONDecodeError:
             log_error(f"apply_labels_from_csv: Error decoding JSON from {json_file_path}")
             raise ValueError("The uploaded transcript file is not valid JSON.")
        except ValueError as e: # Catch specific value errors (e.g., bad CSV)
             log_error(f"apply_labels_from_csv: Data error: {e}")
             raise e # Re-raise specific error for UI
        except Exception as e:
             log_error(f"apply_labels_from_csv: Unexpected error: {e}\n{traceback.format_exc()}")
             raise RuntimeError(f"Failed to apply labels: {e}")