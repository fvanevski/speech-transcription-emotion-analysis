# core/pipeline.py
import csv
import json
import difflib # For fuzzy snippet matching
import shutil
import statistics
import subprocess
import traceback
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

# Need DataFrame for type hint in relabel_and_finalize (from UI component)
from pandas import DataFrame

# --- Use Relative Imports for modules within the 'core' package ---
from .constants import EMO_VAL # Use constants
from .logging import log_error, log_info, log_warning
from .plotting import generate_all_plots
from .transcription import Transcription
# --- End Relative Imports ---

# --- Define constants ---
LOG_FILE_NAME: str = "process_log.txt"
STRUCTURED_TRANSCRIPT_NAME: str = "structured_transcript.json"
RELABELED_SUFFIX: str = "_relabeled"
EMOTION_SUMMARY_CSV_NAME: str = "emotion_summary.csv"
EMOTION_SUMMARY_JSON_NAME: str = "emotion_summary.json"
FINAL_ZIP_SUFFIX: str = "_final_bundle.zip"
# Default threshold for fuzzy snippet matching (can be overridden by config)
DEFAULT_SNIPPET_MATCH_THRESHOLD: float = 0.85

# --- Define type aliases ---
Segment = Dict[str, Any]
SegmentsList = List[Segment]
SpeakerPreview = Dict[str, str]
SpeakerPreviewsList = List[SpeakerPreview]
EmotionSummary = Dict[str, Dict[str, Any]]
# Final data from UI speaker editing table
SpeakerLabelData = Optional[List[Dict[str, Any]]]


class Pipeline:
    """
    Orchestrates the end-to-end speech processing workflow.

    Manages input handling, transcription/diarization (via Transcription class),
    optional AI-assisted speaker labeling via snippets, manual speaker label editing,
    final analysis, plotting, and result packaging.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initializes the pipeline with configuration."""
        self.config: Dict[str, Any] = config
        self.transcription: Transcription = Transcription(config)
        # FileManager was removed previously

    # --- Helper Methods ---

    def _prepare_audio_input(
        self, input_source: str, temp_dir: Path, log_file_handle: TextIO, session_id: str
    ) -> Path:
        """Handles URL download or local file copying, returns Path to audio file."""
        audio_path: Optional[Path] = None
        log_info(f"Preparing audio input from: {input_source}")
        if input_source.startswith(("http://", "https://")):
            dl_path_str = self.transcription.download_audio_from_youtube(
                input_source, str(temp_dir), log_file_handle, session_id
            )
            if not dl_path_str:
                 raise RuntimeError(f"Audio download failed for URL: {input_source}")
            audio_path = Path(dl_path_str)
        elif Path(input_source).exists():
            input_path = Path(input_source)
            temp_audio_path = temp_dir / f"input_audio{input_path.suffix}"
            try:
                shutil.copy(input_path, temp_audio_path)
                audio_path = temp_audio_path
            except shutil.Error as copy_err:
                 raise IOError(f"Failed to copy input file {input_path} to {temp_audio_path}: {copy_err}")
        else:
            raise ValueError(f"Invalid input source provided (not URL or existing file): {input_source}")

        if not audio_path or not audio_path.exists():
             raise FileNotFoundError(f"Audio file could not be obtained/found from input: {input_source}")

        log_info(f"Audio prepared at: {audio_path}")
        return audio_path

    def _run_ffprobe_duration_check(self, audio_path: Path) -> bool:
        """Checks if audio duration is sufficient using ffprobe."""
        audio_path_str = str(audio_path)
        try:
            log_info(f"Running ffprobe duration check for: {audio_path_str}")
            result: subprocess.CompletedProcess = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", audio_path_str,
                ],
                capture_output=True, text=True, check=True,
            )
            duration: float = float(result.stdout.strip())
            log_info(f"Audio duration: {duration} seconds")
            min_duration: float = float(self.config.get("min_diarization_duration", 5.0))
            if duration < min_duration:
                raise ValueError(
                    f"Audio duration ({duration:.2f}s) is less than minimum required ({min_duration}s) for reliable diarization."
                )
            return True
        except FileNotFoundError:
            log_error("ffprobe command not found. Cannot check duration.")
            return True # Assume OK if ffprobe isn't installed
        except subprocess.CalledProcessError as e:
            log_error(f"ffprobe failed: {e}")
            raise RuntimeError(f"ffprobe check failed: {e.stderr or e.stdout}")
        except ValueError as e:
            log_error(f"ffprobe parsing/check error: {e}")
            raise e
        except Exception as e:
            log_error(f"ffprobe unexpected error: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Duration check failed unexpectedly: {e}")

    def _find_whisperx_output(self, job_output_dir: Path, audio_path: Path) -> Path:
        """Finds the primary JSON output from WhisperX."""
        # Use audio filename stem to predict WhisperX output name
        expected_json_filename: str = f"{audio_path.stem}.json"
        json_output_path: Path = job_output_dir / expected_json_filename

        if not json_output_path.exists():
            found_json_path: Optional[Path] = None
            log_warning(f"Expected WhisperX JSON '{expected_json_filename}' not found. Searching directory...")
            # Search for any .json file in the output dir that isn't one of ours
            for item_path in job_output_dir.iterdir():
                if (
                    item_path.is_file() and item_path.suffix == ".json"
                    and EMOTION_SUMMARY_JSON_NAME not in item_path.name
                    and STRUCTURED_TRANSCRIPT_NAME not in item_path.name
                    and not item_path.name.endswith(f"{RELABELED_SUFFIX}.json")
                ):
                    found_json_path = item_path
                    log_info(f"Found likely WhisperX JSON output: {found_json_path}")
                    break # Assume first one found is correct
            if not found_json_path:
                raise FileNotFoundError(
                    f"WhisperX JSON output (like {expected_json_filename}) not found in {job_output_dir}."
                )
            json_output_path = found_json_path # Use the found path

        return json_output_path

    def _group_segments_by_speaker(self, segments: SegmentsList) -> List[Dict[str, Any]]:
        """
        Groups consecutive segments by the same speaker ID into blocks.

        Args:
            segments (SegmentsList): The list of transcript segments.

        Returns:
            List[Dict[str, Any]]: A list where each dict represents a contiguous
                                  block of speech from one speaker, containing
                                  'speaker', combined 'text', 'start', 'end',
                                  and 'original_segment_indices'.
        """
        if not segments:
            return []

        speaker_blocks: List[Dict[str, Any]] = []
        current_block: Optional[Dict[str, Any]] = None

        for i, segment in enumerate(segments):
            speaker_id = segment.get("speaker")
            # Use .get() for text and handle potential None before strip()
            segment_text_raw = segment.get("text")
            segment_text = segment_text_raw.strip() if segment_text_raw is not None else ""
            start_time = segment.get("start")
            end_time = segment.get("end")

            # Skip segments without speaker ID
            # Keep segments even if they have no text, as they might separate blocks
            if not speaker_id:
                # If we were building a block, finalize it before the gap
                if current_block:
                    speaker_blocks.append(current_block)
                current_block = None # Reset block due to missing speaker
                continue

            # If this segment continues the current block
            if current_block and current_block["speaker"] == speaker_id:
                # Append text (only if segment has text)
                if segment_text:
                     # Add space only if block already has text
                     separator = " " if current_block["text"] else ""
                     current_block["text"] += separator + segment_text
                # Update end time if available
                if end_time is not None:
                    current_block["end"] = end_time
                # Add original segment index
                current_block["original_segment_indices"].append(i)
            else: # Start a new block
                # Finalize the previous block if it exists
                if current_block:
                    speaker_blocks.append(current_block)
                # Start the new block
                current_block = {
                    "speaker": speaker_id,
                    "text": segment_text, # Start with current segment's text
                    "start": start_time,
                    "end": end_time,
                    "original_segment_indices": [i]
                }

        # Append the last block after the loop finishes
        if current_block:
            speaker_blocks.append(current_block)

        return speaker_blocks

    def _match_snippets_to_speakers(self, segments: SegmentsList, speaker_snippet_map: Dict[str, str]) -> Dict[str, str]:
        """
        Attempts to map generic speaker IDs to known names using fuzzy substring
        matching for provided dialogue snippets within contiguous speaker blocks.

        Args:
            segments (SegmentsList): The list of transcript segments.
            speaker_snippet_map (Dict[str, str]): Mapping of known names to snippets.

        Returns:
            Dict[str, str]: Mapping of generic speaker IDs to matched known names.
        """
        if not speaker_snippet_map:
            log_info("No speaker snippets provided, skipping snippet matching.")
            return {}

        # Group consecutive segments by speaker first
        speaker_blocks = self._group_segments_by_speaker(segments)
        if not speaker_blocks:
            log_warning("Could not form speaker blocks from segments for snippet matching.")
            return {}

        snippet_based_mapping: Dict[str, str] = {}
        match_scores: Dict[str, float] = {} # speaker_id -> best_match_ratio
        # Get threshold from config or use default
        match_threshold: float = float(self.config.get("snippet_match_threshold", DEFAULT_SNIPPET_MATCH_THRESHOLD))
        log_info(f"Attempting fuzzy snippet matching within speaker blocks (Threshold: {match_threshold})...")

        # Iterate through each known speaker and their snippet
        for speaker_name, snippet in speaker_snippet_map.items():
            if not snippet: continue # Skip empty snippets
            snippet_len = len(snippet)
            # Avoid matching on extremely short snippets as they are prone to false positives
            if snippet_len < 5:
                log_warning(f"Skipping very short snippet (len < 5) for '{speaker_name}': '{snippet}'")
                continue

            snippet_lower = snippet.lower()

            # Iterate through each contiguous block of text from a speaker
            for block in speaker_blocks:
                block_text: str = block.get("text", "")
                speaker_id: Optional[str] = block.get("speaker")
                block_text_lower = block_text.lower()

                # Skip blocks without text or speaker ID
                if not block_text or not speaker_id:
                    continue

                # Find the best matching contiguous subsequence using SequenceMatcher
                matcher = difflib.SequenceMatcher(
                    None, snippet_lower, block_text_lower,
                    autojunk=False # Process all characters for potentially better accuracy
                )
                # Find the single best matching block (longest common subsequence)
                match = matcher.find_longest_match(0, snippet_len, 0, len(block_text_lower))

                # Calculate ratio based on the size of the longest match relative to snippet length
                ratio = 0.0
                if snippet_len > 0: # Avoid division by zero for empty snippet edge case
                     # match.size is the length of the longest common subsequence found
                     ratio = float(match.size) / snippet_len

                # Log if potentially interesting (e.g., ratio > 0.5) for debugging
                log_ratio_threshold = 0.5 # Threshold for just logging, lower than matching threshold
                if ratio > log_ratio_threshold:
                    matched_block_text = block_text[match.b : match.b + match.size] # Extract matched part from original case block
                    log_info(
                        f"Snippet Check: Ratio={ratio:.3f} for '{speaker_name}' (Snippet: '{snippet[:50]}...') "
                        f"vs {speaker_id} Block (Matched Text: '{matched_block_text[:100]}...')"
                    )

                # Check if this match meets the actual matching threshold
                if ratio >= match_threshold:
                    current_best_ratio_for_id = match_scores.get(speaker_id, -1.0)

                    # Check if this ratio is better than any previous match for this speaker_id
                    if ratio > current_best_ratio_for_id:
                        # Check if this assignment conflicts with a previous snippet's assignment for the same ID
                        if speaker_id in snippet_based_mapping and snippet_based_mapping[speaker_id] != speaker_name:
                             log_warning(
                                 f"Conflict for {speaker_id}! Was '{snippet_based_mapping[speaker_id]}', "
                                 f"now matching '{speaker_name}' with higher ratio {ratio:.3f}. Overwriting."
                             )
                        elif speaker_id not in snippet_based_mapping:
                             log_info(
                                 f"--> Match Found! (Ratio: {ratio:.3f}): Assigning '{speaker_name}' to {speaker_id}."
                             )
                        # If same speaker/id assignment but higher ratio, just update score silently

                        # Update the mapping and the best score found for this speaker_id
                        snippet_based_mapping[speaker_id] = speaker_name
                        match_scores[speaker_id] = ratio
                        # Continue checking other blocks; a later block for the same speaker might
                        # contain the snippet with an even better ratio (less noise etc.)

        log_info(f"Fuzzy snippet matching complete. Found {len(snippet_based_mapping)} mappings: {snippet_based_mapping}")
        return snippet_based_mapping

    def _get_speaker_previews(self, segments: SegmentsList, initial_mapping: Optional[Dict[str, str]] = None) -> SpeakerPreviewsList:
        """
        Extracts unique speaker IDs and their first dialogue snippet,
        pre-filling labels based on initial mapping (e.g., from snippets).
        """
        speakers_data: Dict[str, str] = {}
        # Use defaultdict to ensure initial_mapping exists even if None
        mapping = initial_mapping or {}
        processed_ids: set[str] = set() # Keep track of IDs already added to preview

        # First pass: Find first dialogue preview for each speaker ID present in segments
        for seg in segments:
            spk_id: Optional[str] = seg.get("speaker")
            # Ensure spk_id is treated as string if not None
            spk_id_str = str(spk_id) if spk_id is not None else None

            if spk_id_str and spk_id_str not in processed_ids:
                # Get preview text, handle None before strip()
                preview_text_raw: Optional[str] = seg.get("text")
                preview_text: str = preview_text_raw.strip()[:80] if preview_text_raw else ""

                if preview_text:
                    speakers_data[spk_id_str] = preview_text
                # Still add speaker ID even if preview is empty, if it's in mapping
                elif spk_id_str in mapping:
                    speakers_data[spk_id_str] = "" # Add with empty preview

                # Mark ID as processed only if we added it
                if spk_id_str in speakers_data:
                    processed_ids.add(spk_id_str)

        # Second pass: Ensure speakers identified *only* by snippets are included
        for spk_id_str, mapped_name in mapping.items():
            if spk_id_str not in processed_ids:
                # Add this speaker ID identified only via snippet
                speakers_data[spk_id_str] = "[Identified by snippet]" # Placeholder preview
                processed_ids.add(spk_id_str) # Mark as processed

        # Build the final list for the UI DataFrame, sorted by Speaker ID
        preview_list: SpeakerPreviewsList = []
        # Sort by speaker ID numerically if possible (e.g., SPEAKER_00, SPEAKER_01, ...)
        def sort_key(item):
            spk_id = item[0]
            parts = spk_id.split('_')
            if len(parts) == 2 and parts[0] == 'SPEAKER' and parts[1].isdigit():
                return (0, int(parts[1])) # Sort numerically
            return (1, spk_id) # Sort alphabetically otherwise

        for spk_id_str, preview in sorted(speakers_data.items(), key=sort_key):
             # Get pre-filled label from the snippet mapping, default to empty string
             initial_label = mapping.get(spk_id_str, "")
             preview_list.append(
                {"Speaker ID": spk_id_str, "Dialogue Preview": preview, "Enter Label Here": initial_label}
             )
        return preview_list

    def _calculate_emotion_summary(self, segments: SegmentsList, include_timeline: bool = False) -> EmotionSummary:
        """Calculates summary statistics based on structured segments."""
        # Uses speaker labels potentially modified by snippet matching + manual edits
        speaker_stats: Dict[str, List[str]] = defaultdict(list)
        segment_details: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in segments:
            speaker_key: str = str(s.get("speaker", "unknown"))
            emotion: str = s.get("emotion", "unknown")
            speaker_stats[speaker_key].append(emotion)
            if include_timeline:
                # Ensure 'start' time exists, default to 0.0 if missing
                start_time = s.get("start", 0.0)
                segment_details[speaker_key].append({"time": start_time, "emotion": emotion})

        summarized: EmotionSummary = {}
        for spk, emos in speaker_stats.items():
            if not emos: continue # Skip speakers with no emotion entries

            transitions: int = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i - 1])
            dominant: str = Counter(emos).most_common(1)[0][0] if emos else "unknown"
            # Use EMO_VAL imported from constants
            emo_numeric: List[float] = [EMO_VAL.get(e, 0.0) for e in emos]
            volatility: float = 0.0
            if len(emo_numeric) > 1:
                 try:
                     volatility = stdev(emo_numeric)
                 except statistics.StatisticsError: # Handle case with single identical value or not enough data
                     volatility = 0.0
            avg_emotion: float = mean(emo_numeric) if emo_numeric else 0.0

            summary_item: Dict[str, Any] = {
                "total_segments": len(emos),
                "emotion_transitions": transitions,
                "dominant_emotion": dominant,
                "emotion_volatility": round(volatility, 3),
                "emotion_score_mean": round(avg_emotion, 3),
            }
            # Include sorted timeline if requested and available
            if include_timeline and spk in segment_details:
                # Ensure sorting key handles potential None times gracefully
                timeline: List[Dict[str, Any]] = sorted(
                    segment_details[spk], key=lambda x: x.get("time", float('inf')) # Sort None times last? Or 0?
                )
                summary_item["emotion_timeline"] = timeline
            summarized[spk] = summary_item
        return summarized

    def _save_emotion_summary(self, speaker_stats: EmotionSummary, output_dir: Path, file_suffix: str) -> Tuple[Path, Path]:
        """Saves the calculated emotion summary to CSV and JSON, returns Paths."""
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        # Use pathlib for cleaner path manipulation
        base_csv_name: str = Path(EMOTION_SUMMARY_CSV_NAME).stem
        base_json_name: str = Path(EMOTION_SUMMARY_JSON_NAME).stem

        csv_path: Path = output_dir / f"{base_csv_name}_{file_suffix}.csv"
        json_path: Path = output_dir / f"{base_json_name}_{file_suffix}.json"
        log_info(f"Saving emotion summary: {json_path}, {csv_path}")

        try:
            # Save JSON summary
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(speaker_stats, json_file, indent=2, ensure_ascii=False)

            # Save CSV summary
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                # Dynamically determine headers from the first valid entry, excluding timeline
                headers: List[str] = ["speaker"]
                # Define default keys in case no data is found
                default_stat_keys: List[str] = [
                    "total_segments", "emotion_transitions", "dominant_emotion",
                    "emotion_volatility", "emotion_score_mean"
                ]
                if speaker_stats:
                    # Find first speaker key that has non-empty stats
                    first_valid_speaker_key = next((spk for spk, stats in speaker_stats.items() if stats), None)
                    if first_valid_speaker_key:
                         stats_dict = speaker_stats[first_valid_speaker_key]
                         # Ensure all keys are strings before sorting, exclude timeline
                         stat_keys: List[str] = sorted([str(k) for k in stats_dict.keys() if k != "emotion_timeline"])
                         headers.extend(stat_keys)
                    else:
                        headers.extend(default_stat_keys) # Fallback if no speaker has stats
                else:
                    headers.extend(default_stat_keys) # Fallback if no speakers at all

                writer = csv.DictWriter(csv_file, fieldnames=headers, extrasaction="ignore")
                writer.writeheader()
                # Sort speakers before writing for consistent CSV output
                for speaker, stats in sorted(speaker_stats.items()):
                    row_data: Dict[str, Any] = {"speaker": speaker}
                    # Ensure stats is a dict before updating
                    if isinstance(stats, dict):
                        row_data.update(stats)
                    else:
                         log_warning(f"Unexpected stats type for speaker '{speaker}': {type(stats)}. Skipping CSV row details.")
                    writer.writerow(row_data) # Write row, ignoring extra fields (like timeline)

        except (IOError, TypeError, csv.Error) as e:
             log_error(f"Error saving emotion summary files ({csv_path}, {json_path}): {e}")
             raise # Re-raise to be handled by caller
        return csv_path, json_path

    def _create_speaker_mapping(self, speaker_label_list: SpeakerLabelData) -> Dict[str, str]:
        """
        Parses list of speaker label dicts from UI edits into a speaker ID mapping dict.
        It reflects the final state of the UI DataFrame after user interaction.
        """
        speaker_mapping: Dict[str, str] = {}

        if not speaker_label_list:
            log_info("No speaker label data provided from UI for final mapping.")
            return speaker_mapping

        log_info(f"Processing {len(speaker_label_list)} edited speaker label entries from UI.")
        expected_keys = ["Speaker ID", "Enter Label Here"] # Headers from UI DataFrame

        # Iterate through each dictionary (row) provided by the UI edits
        for i, row_dict in enumerate(speaker_label_list):
            if not isinstance(row_dict, dict):
                log_warning(f"Skipping final speaker label item {i}: Expected dictionary, got {type(row_dict)}.")
                continue

            # Check if required keys are present
            if not all(key in row_dict for key in expected_keys):
                log_warning(f"Skipping final speaker label item {i}: Missing expected keys ({expected_keys}). Found: {list(row_dict.keys())}.")
                continue

            try:
                # Extract original ID and the label potentially edited by the user
                original_speaker: str = str(row_dict.get("Speaker ID"))
                # Use the value from "Enter Label Here" column, which reflects user's final choice
                final_label: str = str(row_dict.get("Enter Label Here", "")).strip()

                # Ensure original speaker ID is valid
                if not original_speaker:
                    log_warning(f"Skipping final speaker label item {i}: 'Speaker ID' is missing or empty.")
                    continue

                # Create the final mapping: Use the label from the UI if provided, otherwise fall back to the original ID.
                # This handles cases where the user clears a pre-filled label or leaves it unchanged.
                speaker_mapping[original_speaker] = (final_label if final_label else original_speaker)

            except Exception as e:
                log_warning(f"Error processing final speaker label entry {i} ({row_dict}): {e}")
                continue # Skip problematic rows

        log_info(f"Created final speaker mapping from UI edits: {speaker_mapping}")
        return speaker_mapping

    def _apply_speaker_mapping(self, segments: SegmentsList, mapping: Dict[str, str]) -> Tuple[SegmentsList, int]:
        """Applies final speaker mapping to segments list, returns modified list and count."""
        modified_count: int = 0
        if not mapping: # If no mapping provided, return early
            log_info("No final speaker mapping provided to apply.")
            return segments, 0

        log_info(f"Applying final speaker mapping: {mapping}")
        for seg in segments:
            original_speaker: str = str(seg.get("speaker", "unknown"))
            # Apply mapping if the original speaker ID is in the map
            if original_speaker in mapping:
                mapped_label: str = mapping[original_speaker]
                # Only count as modified if the label actually changes from its current value
                if seg.get("speaker") != mapped_label:
                    # Optionally store the original generic speaker ID if it's not already stored
                    if "speaker_original_id" not in seg:
                         seg["speaker_original_id"] = original_speaker # Store generic ID
                    seg["speaker"] = mapped_label # Update to final name
                    modified_count += 1
            # Else: Keep the existing speaker label (which might be original ID or from a previous step)

        if modified_count > 0:
             log_info(f"Applied final labels to {modified_count} segments based on mapping.")
        else:
             log_info("No segment speaker labels were changed by the final mapping.")
        return segments, modified_count

    def _create_final_zip(
        self, zip_path: Path, files_to_add: Dict[Path, str], plot_files: List[Path], log_file: Path
    ) -> None:
        """Creates the final zip archive containing specified results."""
        log_info(f"Creating final results ZIP: {zip_path}")
        try:
            # Ensure parent directory for the zip file exists
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file_handle:
                # Add main output files defined in the dictionary
                for file_path, arc_name in files_to_add.items():
                     if file_path and file_path.exists():
                          zip_file_handle.write(file_path, arcname=arc_name)
                     else:
                         log_warning(f"Required output file {file_path} (zip name: {arc_name}) not found for zipping.")

                # Add plot files to a 'plots' subdirectory within the zip
                plots_arc_dir: str = "plots"
                for plot_file in plot_files:
                    if plot_file and plot_file.exists():
                        # Create the path inside the zip archive
                        archive_plot_path = Path(plots_arc_dir) / plot_file.name
                        zip_file_handle.write(plot_file, arcname=str(archive_plot_path))
                    else:
                        log_warning(f"Plot file {plot_file} not found for zipping.")

                # Add the job-specific process log file
                if log_file.exists():
                    zip_file_handle.write(log_file, arcname=log_file.name)
                else:
                    log_warning(f"Job log file {log_file} not found for zipping.")
        except (zipfile.BadZipFile, IOError, OSError) as zip_err:
             log_error(f"Failed to create zip file {zip_path}: {zip_err}")
             raise # Re-raise to be handled by caller

    # --- Main Public Methods ---

    def process_audio(
        self, input_source: str, speaker_snippet_map: Optional[Dict[str, str]] = None
        ) -> Tuple[str, Optional[str], SpeakerPreviewsList]:
        """
        Orchestrates the initial audio processing steps: download/copy, transcribe,
        structure, apply initial speaker labels via snippet matching, and generate
        speaker previews for the UI.

        Args:
            input_source (str): Path to local audio file or YouTube URL.
            speaker_snippet_map (Optional[Dict[str, str]]): Mapping of known speaker
                names to dialogue snippets provided by the user. Defaults to None.

        Returns:
            Tuple[str, Optional[str], SpeakerPreviewsList]:
                - Status message (string).
                - Path to the structured transcript JSON (string, or None on error).
                - List of speaker preview dictionaries, potentially pre-filled based
                  on snippet matching.
        """
        session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id: str = str(uuid.uuid4())[:8] # Short unique ID for the job
        base_dir_path: Path = Path(self.config.get("output_dir", "output"))
        # Create a unique directory for this job's output
        job_output_dir_path: Path = base_dir_path / f"job_{session_id}_{job_id}"
        # Create temp dir *inside* job dir for better isolation and cleanup
        temp_dir_path: Path = job_output_dir_path / "temp"
        log_path: Path = job_output_dir_path / LOG_FILE_NAME

        # Ensure directories exist
        job_output_dir_path.mkdir(parents=True, exist_ok=True)
        temp_dir_path.mkdir(exist_ok=True)

        log_file_handle: Optional[TextIO] = None
        structured_path_str: Optional[str] = None
        status_message: str = f"Starting job {job_id}..."
        error_message: str = "" # Initialize error message

        try:
            # --- Start logging ---
            log_file_handle = open(log_path, "w", encoding="utf-8")
            log_file_handle.write(f"📄 Processing Job ID: {job_id}\nSession ID: {session_id}\nStarted: {datetime.now()}\nInput: {input_source}\n\n")
            log_file_handle.flush()
            log_info(f"Job {job_id} started. Output dir: {job_output_dir_path}")

            # --- Step 1-4: Prepare Audio, Run WhisperX, Process Output ---
            log_info(f"Job {job_id}: Preparing audio input...")
            audio_path: Path = self._prepare_audio_input(input_source, temp_dir_path, log_file_handle, session_id)
            log_info(f"Job {job_id}: Checking audio duration...")
            self._run_ffprobe_duration_check(audio_path) # Optional pre-check
            log_info(f"Job {job_id}: Running transcription and diarization...")
            self.transcription.run_whisperx(str(audio_path), str(job_output_dir_path), log_file_handle, session_id)
            log_info(f"Job {job_id}: Locating and structuring WhisperX output...")
            json_output_path: Path = self._find_whisperx_output(job_output_dir_path, audio_path)
            structured_data: SegmentsList = self.transcription.convert_json_to_structured(str(json_output_path))

            # --- Step 5: Apply Snippet Matching (if snippets provided) ---
            initial_mapping: Dict[str, str] = self._match_snippets_to_speakers(
                structured_data, speaker_snippet_map or {} # Pass empty dict if None
            )

            # --- Step 6: Save Original Structured Transcript ---
            # This saves the transcript BEFORE final manual labeling, but AFTER initial emotion analysis.
            structured_path: Path = job_output_dir_path / STRUCTURED_TRANSCRIPT_NAME
            with open(structured_path, "w", encoding="utf-8") as f_json_out:
                json.dump(structured_data, f_json_out, indent=2, ensure_ascii=False)
            structured_path_str = str(structured_path) # Store path for the next step
            log_info(f"Job {job_id}: Initial structured transcript saved to {structured_path_str}")

            # --- Step 7: Get Speaker Previews for UI (pre-filled based on snippets) ---
            speaker_previews: SpeakerPreviewsList = self._get_speaker_previews(
                structured_data, initial_mapping # Pass the snippet mapping here
            )
            # Count unique generic speaker IDs found in the segments
            unique_speaker_ids = set(seg.get("speaker") for seg in structured_data if seg.get("speaker"))
            num_speakers: int = len(unique_speaker_ids)
            num_mapped_by_snippet: int = len(initial_mapping)
            log_info(f"Job {job_id}: Found {num_speakers} unique speaker IDs. Mapped {num_mapped_by_snippet} via snippets.")

            # --- Prepare success message ---
            status_message = (
                 f"✅ Job {job_id} processed. Found {num_speakers} speakers. "
                 f"Applied {num_mapped_by_snippet} labels from snippets. Ready for final labeling below."
            )
            return status_message, structured_path_str, speaker_previews

        # --- Error Handling ---
        except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError, IOError, shutil.Error, json.JSONDecodeError, TypeError) as specific_err:
             error_message = f"❌ Job {job_id} failed during initial processing: {specific_err}"
             log_error(error_message)
             # Log traceback only if log handle is valid
             if log_file_handle and not log_file_handle.closed:
                 try: log_file_handle.write(f"\n\n--- ERROR ---\n{error_message}\n{traceback.format_exc()}\n")
                 except Exception as log_e: print(f"Additionally failed to write error to log file: {log_e}")
             else: # Also log traceback to main log if handle failed
                 log_error(traceback.format_exc())
             # Return error status, None for path, empty list for previews
             return error_message, None, []
        except Exception as e: # Catch any other unexpected errors
            error_message = f"❌ Job {job_id} failed with unexpected error: {e}"
            log_error(error_message)
            # Log traceback
            if log_file_handle and not log_file_handle.closed:
                try: log_file_handle.write(f"\n\n--- ERROR ---\n{error_message}\n{traceback.format_exc()}\n")
                except Exception as log_e: print(f"Additionally failed to write error to log file: {log_e}")
            else:
                 log_error(traceback.format_exc())
            return error_message, None, []
        finally:
            # --- Cleanup ---
            log_info(f"Job {job_id}: Cleaning up temporary files...")
            if log_file_handle and not log_file_handle.closed:
                try: log_file_handle.close()
                except Exception: pass # Ignore errors closing log file
            # Cleanup temp dir associated with this job
            if temp_dir_path.exists():
                try:
                    shutil.rmtree(temp_dir_path)
                    log_info(f"Removed temporary directory: {temp_dir_path}")
                except OSError as e:
                    log_error(f"Error removing temp directory {temp_dir_path}: {e}")

    def relabel_and_finalize(
        self,
        structured_json_path_str: str,
        edited_dataframe_data: SpeakerLabelData # Expects Optional[List[Dict[str, Any]]]
    ) -> Tuple[Optional[str], str]:
        """
        Orchestrates the relabeling (based on UI edits) and final packaging steps.
        Loads the original structured transcript, applies the final labels derived
        from the potentially pre-filled and user-edited UI DataFrame, generates
        final summary/plots, and creates ZIP bundle.

        Args:
            structured_json_path_str (str): Path to the original structured transcript JSON
                                            generated by process_audio.
            edited_dataframe_data (SpeakerLabelData): Data from the UI DataFrame component
                                                      after user edits. Expected as a list
                                                      of dictionaries reflecting the final state.

        Returns:
            Tuple[Optional[str], str]:
                - Path to the final ZIP file (string, or None on error).
                - Status message (string).
        """
        if not structured_json_path_str:
             # Use log_error for internal errors before returning user message
             log_error("Relabeling failed: Missing structured JSON path input.")
             return None, "❌ Error: Missing structured JSON path from previous step."
        structured_json_path = Path(structured_json_path_str)
        if not structured_json_path.exists():
             error_message: str = f"Invalid or missing structured JSON path provided: {structured_json_path_str}"
             log_error(error_message)
             return None, f"❌ Error: {error_message}"

        log_info(f"Starting final labeling and finalization for: {structured_json_path}")
        output_dir_path: Path = structured_json_path.parent # Job-specific output dir
        job_name: str = output_dir_path.name # Get job name from directory
        final_zip_path: Optional[Path] = None
        error_message: str = "" # Initialize error message

        # Get log path for this job to include in the final zip
        job_log_path: Path = output_dir_path / LOG_FILE_NAME

        try:
            # --- Step 1: Create FINAL Speaker Mapping from edited UI data ---
            # This mapping reflects the final state of the UI DataFrame, including
            # snippet pre-fills and any user edits/overrides.
            final_speaker_mapping: Dict[str, str] = self._create_speaker_mapping(edited_dataframe_data)

            # --- Step 2: Load original structured data ---
            # We load the original data again to apply the FINAL mapping cleanly.
            data: SegmentsList
            try:
                with open(structured_json_path, "r", encoding="utf-8") as json_file_in:
                    data = json.load(json_file_in)
            except (FileNotFoundError, json.JSONDecodeError) as json_err:
                 log_error(f"Failed to load original structured JSON {structured_json_path}: {json_err}")
                 raise IOError(f"Could not load necessary file: {structured_json_path.name}") from json_err

            # --- Step 3: Apply FINAL mapping to segments ---
            relabeled_data, modified_count = self._apply_speaker_mapping(data, final_speaker_mapping)

            # --- Step 4: Save Relabeled Structured Transcript ---
            relabeled_json_path: Path = output_dir_path / f"{structured_json_path.stem}{RELABELED_SUFFIX}{structured_json_path.suffix}"
            log_info(f"Saving final relabeled transcript data to: {relabeled_json_path}")
            with open(relabeled_json_path, "w", encoding="utf-8") as f_relabeled:
                json.dump(relabeled_data, f_relabeled, indent=2, ensure_ascii=False)

            # --- Step 5: Calculate and Save Final Emotion Summary (based on relabeled data) ---
            log_info("Calculating final emotion summary...")
            final_summary: EmotionSummary = self._calculate_emotion_summary(relabeled_data, include_timeline=True)
            final_summary_suffix: str = job_name + RELABELED_SUFFIX # Suffix for summary files
            csv_path: Path; json_path: Path
            csv_path, json_path = self._save_emotion_summary(final_summary, output_dir_path, final_summary_suffix)
            log_info(f"Final emotion summary saved: {csv_path.name}, {json_path.name}")

            # --- Step 6: Generate Plots (based on final summary) ---
            log_info("Generating plots...")
            plot_filenames: List[Path] = []
            try:
                plot_suffix: str = final_summary_suffix # Use same suffix for plots
                plot_path_strs = generate_all_plots(final_summary, str(output_dir_path), plot_suffix)
                plot_filenames = [Path(p) for p in plot_path_strs]
                log_info(f"Generated {len(plot_filenames)} plot files.")
            except Exception as plot_err:
                # Log error but don't stop the whole process if plotting fails
                log_error(f"Warning: Failed during plot generation: {plot_err}\n{traceback.format_exc()}")

            # --- Step 7: Create Final ZIP Bundle ---
            # Define the final zip path (e.g., one level up from job dir)
            final_zip_path = output_dir_path.parent / f"{job_name}{FINAL_ZIP_SUFFIX}"
            # Define files to include and their names within the zip
            files_to_zip: Dict[Path, str] = {
                relabeled_json_path: relabeled_json_path.name, # Final transcript
                csv_path: csv_path.name,                     # Final summary CSV
                json_path: json_path.name,                   # Final summary JSON
                # Optionally include the original structured transcript for reference
                structured_json_path: f"{structured_json_path.stem}_original{structured_json_path.suffix}"
            }
            self._create_final_zip(final_zip_path, files_to_zip, plot_filenames, job_log_path)

            # --- Success ---
            status_message: str = (
                f"✅ Final labeling and processing complete for {job_name}. Results bundled: {final_zip_path.name}"
            )
            return str(final_zip_path) if final_zip_path else None, status_message

        # --- Error Handling for Finalization ---
        except (FileNotFoundError, IOError, zipfile.BadZipFile, csv.Error, json.JSONDecodeError, TypeError) as specific_err:
             error_message = f"❌ Finalization failed due to file/data error: {specific_err}"
             log_error(error_message)
             log_error(traceback.format_exc()) # Log full traceback for internal debugging
        except Exception as e:
             error_message = f"❌ Finalization failed with unexpected error: {e}"
             log_error(error_message)
             log_error(traceback.format_exc())

        # Ensure an error message is set if process finishes here
        if not error_message:
            error_message = "❌ Finalization failed with an unspecified error."
        # Return None for path and the error message for UI
        return None, error_message