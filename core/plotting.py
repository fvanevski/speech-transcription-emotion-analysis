# ui/main_gui.py
import gradio as gr
import pandas as pd # Import pandas to help with Dataframe manipulation if needed
import os # For os.path.basename
# Assuming Config is needed indirectly via Pipeline
# from config.config import Config

DATAFRAME_HEADERS = ['Speaker ID', 'Dialogue Preview', 'Enter Label Here']

class UI:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo: # Added a soft theme
            gr.Markdown("# Speech Transcription, Labeling, and Analysis")
            gr.Markdown("Process an audio file or YouTube URL, then optionally label speakers and generate final analysis & plots.")

            # State variables to hold intermediate results
            intermediate_json_path_state = gr.State(value=None)
            # Using state for dataframe content might be complex, better to regenerate it

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Step 1: Process Audio File or URL")
                    with gr.Group():
                         input_file = gr.File(label="Upload Audio File (.wav, .mp3, etc.)")
                         youtube_url = gr.Textbox(label="Or Enter YouTube URL")
                         process_btn = gr.Button("1. Process Audio ▶️", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("## Step 2: Label Speakers (Optional)")
                    # Relabeling section - initially hidden
                    with gr.Column(visible=False) as relabel_section:
                        gr.Markdown("Edit the 'Enter Label Here' column with speaker names.")
                        speaker_label_df = gr.DataFrame(
                             headers=DATAFRAME_HEADERS,
                             datatype=["str", "str", "str"], # All strings
                             row_count=(1, "dynamic"), # Allow dynamic rows based on output
                             col_count=(len(DATAFRAME_HEADERS), "fixed"),
                             # Make only the last column editable
                             interactive=[False, False, True],
                             label="Speaker Labels"
                         )
                        relabel_btn = gr.Button("2. Apply Labels & Generate Final Output ✨", variant="primary")

                    gr.Markdown("## Status & Results")
                    status_output = gr.Textbox(label="Current Status", interactive=False, lines=2)
                    download_output = gr.File(label="Download Results (ZIP)", interactive=False)


            # --- Wrapper Functions for Button Clicks ---

            def process_audio_wrapper(input_file_obj, url):
                """Handles Step 1: Process audio, update state and UI for Step 2."""
                input_source = None
                status_msg = "Starting..."
                json_path_update = None
                speaker_df_update = pd.DataFrame(columns=DATAFRAME_HEADERS) # Empty dataframe initially
                relabel_visible = False

                # Determine input source
                if input_file_obj is not None:
                    input_source = input_file_obj.name
                    status_msg = f"Processing uploaded file: {os.path.basename(input_source)}..."
                elif url:
                    input_source = url
                    status_msg = f"Processing YouTube URL: {url}..."
                else:
                    status_msg = "ERROR: Please provide an audio file or a YouTube URL."
                    yield status_msg, None, speaker_df_update, gr.update(visible=False), gr.update(value=None)
                    return # Exit generator

                # Update status immediately and clear previous results
                yield status_msg, None, speaker_df_update, gr.update(visible=False), gr.update(value=None)

                if input_source:
                    try:
                        # Call pipeline's first stage
                        status_msg, json_path, speaker_previews = self.pipeline.process_audio(input_source)

                        if json_path and speaker_previews: # Success and speakers found
                            json_path_update = json_path # Store path for next step
                            # Convert list of dicts to DataFrame for display
                            speaker_df_update = pd.DataFrame(speaker_previews)
                            relabel_visible = True
                        elif json_path: # Success but no speakers found for labeling
                             json_path_update = None # Don't need path if no relabeling
                             relabel_visible = False
                             # status_msg already indicates completion
                        else: # Error occurred (json_path is None)
                             json_path_update = None
                             relabel_visible = False
                             # status_msg contains the error

                    except Exception as e:
                         status_msg = f"ERROR: An unexpected error occurred during processing: {e}"
                         print(f"UI Error during process_audio_wrapper: {e}\n{traceback.format_exc()}")
                         json_path_update = None
                         speaker_df_update = pd.DataFrame(columns=DATAFRAME_HEADERS)
                         relabel_visible = False

                # Final update for all outputs
                yield status_msg, json_path_update, speaker_df_update, gr.update(visible=relabel_visible), gr.update(value=None)

            def relabel_finalize_wrapper(json_path, edited_dataframe):
                """Handles Step 2: Apply labels and finalize output."""
                status_msg = "Starting relabeling..."
                final_zip_path = None

                if not json_path:
                    status_msg = "ERROR: No processed transcript found from Step 1. Cannot relabel."
                    yield status_msg, None # Update status and download link
                    return # Exit generator

                # Update status before calling pipeline
                yield status_msg, None

                try:
                    # Convert DataFrame back to list of dicts or preferred format if needed by pipeline
                    # Assuming pipeline's relabel_and_finalize can handle the pandas DataFrame directly
                    # or convert it (e.g., edited_dataframe.to_dict('records'))
                    speaker_mapping_data = edited_dataframe # Pass the dataframe data directly

                    # Call the pipeline's second stage
                    final_zip_path, status_msg = self.pipeline.relabel_and_finalize(json_path, speaker_mapping_data)

                except Exception as e:
                    status_msg = f"ERROR: An unexpected error occurred during finalization: {e}"
                    print(f"UI Error during relabel_finalize_wrapper: {e}\n{traceback.format_exc()}")
                    final_zip_path = None

                # Return final updates for status and download link
                yield status_msg, final_zip_path


            # --- Connect Buttons to Functions ---

            process_btn.click(
                fn=process_audio_wrapper,
                inputs=[input_file, youtube_url],
                outputs=[
                    status_output,
                    intermediate_json_path_state, # Store path in state
                    speaker_label_df,             # Update Dataframe content
                    relabel_section,              # Update visibility
                    download_output               # Clear download link
                ]
            )

            relabel_btn.click(
                fn=relabel_finalize_wrapper,
                inputs=[
                    intermediate_json_path_state, # Get path from state
                    speaker_label_df              # Get edited dataframe data
                ],
                outputs=[
                    status_output,               # Update status
                    download_output              # Update final download link
                ]
            )

        return demo

    def launch(self):
        self.interface_ui().launch()