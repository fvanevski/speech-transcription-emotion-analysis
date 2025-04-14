# ui/main_gui.py
import gradio as gr
import json # Import json for parsing mapping
from core.pipeline import Pipeline
# Assuming Config is needed indirectly via Pipeline
# from config.config import Config

class UI:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Speech Transcription, Labeling, and Analysis")

            # State variables to hold intermediate results between steps
            # Stores the path to the structured_transcript.json from step 1
            intermediate_json_path_state = gr.State(value=None)
            # Stores the list of speaker IDs found in step 1
            speaker_ids_state = gr.State(value=[])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Step 1: Process Audio File or URL")
                    input_file = gr.File(label="Upload Audio File (.wav, .mp3, etc.)")
                    youtube_url = gr.Textbox(label="Or Enter YouTube URL")
                    process_btn = gr.Button("1. Process Audio")

                with gr.Column(scale=2):
                    gr.Markdown("## Step 2: Label Speakers (Optional)")
                    # This section will be shown after step 1 completes successfully
                    with gr.Column(visible=False) as relabel_section:
                        speaker_id_display = gr.Markdown("Speakers found: (Will appear here)")
                        speaker_mapping_input = gr.Textbox(
                            label="Enter Speaker Mapping (JSON format)",
                            placeholder='Example: {"SPEAKER_00": "Name One", "SPEAKER_01": "Name Two"}',
                            lines=3
                        )
                        relabel_btn = gr.Button("2. Apply Labels & Generate Final Output")

                    gr.Markdown("## Status & Results")
                    status_output = gr.Textbox(label="Current Status", interactive=False)
                    # Final download link
                    download_output = gr.File(label="Download Results (ZIP)", interactive=False)


            # --- Wrapper Functions for Button Clicks ---

            def process_audio_wrapper(input_file_obj, url):
                """Handles Step 1: Process audio, update state and UI for Step 2."""
                input_source = None
                status_msg = "Starting..."
                json_path = None
                speaker_ids = []
                relabel_visible = False
                speaker_display_text = "No speakers found or processing failed."

                if input_file_obj is not None:
                    input_source = input_file_obj.name
                    status_msg = f"Processing uploaded file: {os.path.basename(input_source)}..."
                elif url:
                    input_source = url
                    status_msg = f"Processing YouTube URL: {url}..."
                else:
                    status_msg = "ERROR: Please provide an audio file or a YouTube URL."
                    # Return updates for all outputs defined in process_btn.click
                    return status_msg, None, [], speaker_display_text, gr.update(visible=False), None

                # Update status immediately before calling pipeline
                yield status_msg, None, [], speaker_display_text, gr.update(visible=False), None

                if input_source:
                    try:
                        # Call pipeline's first stage
                        status_msg, json_path, speaker_ids = self.pipeline.process_audio(input_source)

                        if json_path and speaker_ids: # Check for successful result
                            relabel_visible = True
                            speaker_display_text = f"**Speakers found:** `{', '.join(speaker_ids)}`.\nEnter mappings below if desired."
                        elif json_path: # Successful but maybe no speakers?
                             relabel_visible = False # Hide relabeling if no speakers
                             speaker_display_text = "Processing complete, but no distinct speakers found for labeling."
                        else: # Error occurred (json_path is None)
                             relabel_visible = False
                             speaker_display_text = "Processing failed. See status message."

                    except Exception as e:
                         status_msg = f"ERROR: An unexpected error occurred during processing: {e}"
                         print(f"UI Error during process_audio_wrapper: {e}")
                         json_path = None
                         speaker_ids = []
                         relabel_visible = False
                         speaker_display_text = "An unexpected error occurred."

                # Return final updates for all outputs
                yield status_msg, json_path, speaker_ids, speaker_display_text, gr.update(visible=relabel_visible), None


            def relabel_finalize_wrapper(json_path, mapping_text):
                """Handles Step 2: Apply labels and finalize output."""
                status_msg = "Starting relabeling..."
                final_zip_path = None

                if not json_path:
                    status_msg = "ERROR: No processed transcript found. Please complete Step 1 first."
                    return status_msg, None

                try:
                    # Parse the mapping input from the textbox
                    speaker_mapping = json.loads(mapping_text)
                    if not isinstance(speaker_mapping, dict):
                        raise ValueError("Mapping input must be a valid JSON object (dictionary).")

                    status_msg = "Applying labels and generating final results..."
                    # Update status before calling pipeline
                    yield status_msg, None

                    # Call the pipeline's second stage
                    final_zip_path, status_msg = self.pipeline.relabel_and_finalize(json_path, speaker_mapping)

                except json.JSONDecodeError:
                    status_msg = "ERROR: Invalid JSON format in speaker mapping input. Please correct it."
                except ValueError as e:
                     status_msg = f"ERROR: Invalid mapping input: {e}"
                except Exception as e:
                    status_msg = f"ERROR: An unexpected error occurred during finalization: {e}"
                    print(f"UI Error during relabel_finalize_wrapper: {e}")

                # Return updates for status and download link
                yield status_msg, final_zip_path


            # --- Connect Buttons to Functions ---

            process_btn.click(
                fn=process_audio_wrapper,
                inputs=[input_file, youtube_url],
                outputs=[
                    status_output,
                    intermediate_json_path_state, # Update state
                    speaker_ids_state,             # Update state
                    speaker_id_display,            # Update Markdown
                    relabel_section,               # Update visibility
                    download_output                # Clear/update download link
                ]
            )

            relabel_btn.click(
                fn=relabel_finalize_wrapper,
                inputs=[
                    intermediate_json_path_state, # Get path from state
                    speaker_mapping_input         # Get mapping from Textbox
                ],
                outputs=[
                    status_output,                # Update status
                    download_output               # Update final download link
                ]
            )

        return demo

    def launch(self):
        self.interface_ui().launch()