# ui/main_gui.py
import os
import traceback
import re # Import re for parsing
from typing import Dict

import gradio as gr
import pandas as pd

from core.logging import log_error, log_warning, log_info # Import log_info if used
from core.pipeline import Pipeline

DATAFRAME_HEADERS = ["Speaker ID", "Dialogue Preview", "Enter Label Here"]

class UI:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Speech Transcription, Labeling, and Analysis")
            # ... description ...

            # State variables
            intermediate_json_path_state = gr.State(value=None)
            # Add state to hold the snippet mapping derived from text input
            speaker_snippet_map_state = gr.State(value=None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Step 1: Process Audio File or URL")
                    with gr.Group():
                        input_file = gr.File(label="Upload Audio File (.wav, .mp3, etc.)")
                        youtube_url = gr.Textbox(label="Or Enter YouTube URL")
                        # --- ADD SNIPPET INPUT ---
                        speaker_snippets_input = gr.Textbox(
                            label="Known Speakers & Dialogue Snippets (Optional)",
                            placeholder='Format: Speaker Name: Dialogue snippet\nExample:\nAlice Smith: As I mentioned earlier today...\nBob Johnson: The budget numbers look correct.',
                            lines=4,
                            info="Provide one known speaker and a unique short phrase they said per line."
                        )
                        process_btn = gr.Button("1. Process Audio & Apply Snippets ▶️", variant="primary")

                # ... (Column for Step 2: Label Speakers) ...
                with gr.Column(scale=2):
                    gr.Markdown("## Step 2: Refine Speaker Labels (Optional)")
                    with gr.Column(visible=False) as relabel_section:
                         gr.Markdown("AI labels applied from snippets (if any). Edit the 'Enter Label Here' column below to correct or add missing labels.")
                         speaker_label_df = gr.DataFrame( # ... dataframe config ...
                             headers=DATAFRAME_HEADERS, datatype=["str", "str", "str"],
                             row_count=(1,"dynamic"), col_count=(len(DATAFRAME_HEADERS),"fixed"),
                             interactive=[False, False, True], label="Speaker Labels",
                         )
                         relabel_btn = gr.Button("2. Apply Final Labels & Generate Output ✨", variant="primary")
                    # ... (Status & Results Column) ...
                    gr.Markdown("## Status & Results")
                    status_output = gr.Textbox(label="Current Status", interactive=False, lines=2)
                    download_output = gr.File(label="Download Results (ZIP)", interactive=False)


            # --- Wrapper Functions for Button Clicks ---

            def parse_speaker_snippets(snippet_text: str) -> Dict[str, str]:
                """Parses the multiline text input into a Dict[Name, Snippet]."""
                mapping = {}
                if not snippet_text or not snippet_text.strip():
                    return mapping
                lines = snippet_text.strip().split('\n')
                for line in lines:
                    match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
                    if match:
                        name = match.group(1).strip()
                        snippet = match.group(2).strip()
                        if name and snippet:
                            mapping[name] = snippet
                        else:
                            log_warning(f"Could not parse speaker snippet line effectively: '{line}'")
                    else:
                        log_warning(f"Ignoring invalid speaker snippet line format: '{line}'")
                log_info(f"Parsed speaker snippets: {mapping}")
                return mapping

            # --- UPDATE process_audio_wrapper ---
            def process_audio_wrapper(input_file_obj, url, speaker_snippets_text):
                """Handles Step 1: Process audio, apply snippet matching, update state and UI for Step 2."""
                input_source = None
                status_msg = "Starting..."
                json_path_update = None
                speaker_df_update = pd.DataFrame(columns=DATAFRAME_HEADERS)
                relabel_visible = False
                # Parse snippets immediately
                speaker_snippet_map = parse_speaker_snippets(speaker_snippets_text)
                # Store parsed map in state for potential future use if needed, though we pass it directly now
                snippet_map_state_update = speaker_snippet_map

                # ... (Input source determination logic - unchanged) ...
                if input_file_obj is not None:
                    input_source = input_file_obj.name
                    status_msg = f"Processing uploaded file: {os.path.basename(input_source)}..."
                elif url:
                    input_source = url
                    status_msg = f"Processing YouTube URL: {url}..."
                else:
                    status_msg = "ERROR: Please provide an audio file or a YouTube URL."
                    # Yield updates including the snippet map state
                    yield status_msg, None, speaker_df_update, gr.update(visible=False), snippet_map_state_update, gr.update(value=None)
                    return

                # Update status immediately and clear previous results
                yield status_msg, None, speaker_df_update, gr.update(visible=False), snippet_map_state_update, gr.update(value=None)

                if input_source:
                    try:
                        # Call pipeline's first stage, now passing the snippet map
                        status_msg, json_path, speaker_previews = self.pipeline.process_audio(
                            input_source,
                            speaker_snippet_map # Pass the parsed dictionary
                        )

                        # ... (Logic to handle pipeline output - unchanged, speaker_previews now contains pre-filled labels) ...
                        if json_path and speaker_previews:
                            json_path_update = json_path
                            speaker_df_update = pd.DataFrame(speaker_previews)
                            relabel_visible = True
                        elif json_path:
                             json_path_update = None # Keep path if needed for finalize w/o relabel? No, relabel always needs it.
                             relabel_visible = False
                        else: # Error
                             json_path_update = None
                             relabel_visible = False
                    except Exception as e:
                        status_msg = f"ERROR: An unexpected error occurred during processing: {e}"
                        log_error(f"UI Error during process_audio_wrapper: {e}\n{traceback.format_exc()}")
                        json_path_update = None
                        speaker_df_update = pd.DataFrame(columns=DATAFRAME_HEADERS)
                        relabel_visible = False

                # Final update for all outputs
                yield status_msg, json_path_update, speaker_df_update, gr.update(visible=relabel_visible), snippet_map_state_update, gr.update(value=None)

            # --- UPDATE relabel_finalize_wrapper ---
            # No change needed here, the relevant data (edited DataFrame) is passed directly
            def relabel_finalize_wrapper(json_path, edited_dataframe):
                """Handles Step 2: Apply final labels from DataFrame and finalize output."""
                # ... (Existing logic - unchanged) ...
                status_msg = "Starting relabeling..."
                final_zip_path = None
                if not json_path:
                    status_msg = "ERROR: No processed transcript found from Step 1. Cannot relabel."
                    yield status_msg, None
                    return

                yield status_msg, None

                try:
                    speaker_mapping_data = None
                    if isinstance(edited_dataframe, pd.DataFrame):
                        speaker_mapping_data = edited_dataframe.to_dict('records')
                    elif isinstance(edited_dataframe, list):
                         speaker_mapping_data = edited_dataframe
                    else:
                         log_warning(f"Unexpected type for edited_dataframe: {type(edited_dataframe)}. Attempting to proceed.")
                         speaker_mapping_data = edited_dataframe

                    # Call the pipeline's second stage. It uses the JSON path and the edited dataframe.
                    # The initial snippet mapping was already applied *before* this dataframe was generated/edited.
                    final_zip_path, status_msg = self.pipeline.relabel_and_finalize(
                        json_path, speaker_mapping_data
                    )
                except Exception as e:
                    status_msg = f"ERROR: An unexpected error occurred during finalization: {e}"
                    log_error(f"UI Error during relabel_finalize_wrapper: {e}\n{traceback.format_exc()}")
                    final_zip_path = None

                yield status_msg, final_zip_path

            # --- Connect Buttons to Functions ---
            # --- UPDATE process_btn click ---
            process_btn.click(
                 fn=process_audio_wrapper,
                 # Add speaker_snippets_input to inputs
                 inputs=[input_file, youtube_url, speaker_snippets_input],
                 outputs=[
                     status_output,
                     intermediate_json_path_state,
                     speaker_label_df,
                     relabel_section,
                     # Add speaker_snippet_map_state to outputs
                     speaker_snippet_map_state,
                     download_output # Ensure download output is cleared
                 ]
            )
            # --- relabel_btn click remains the same ---
            relabel_btn.click(
                 fn=relabel_finalize_wrapper,
                 inputs=[intermediate_json_path_state, speaker_label_df],
                 outputs=[status_output, download_output]
            )

        return demo

    def launch(self, **kwargs):
        """Launches the Gradio interface."""
        # Calls interface_ui() to get the Blocks object, then calls launch() on it
        self.interface_ui().launch(**kwargs)