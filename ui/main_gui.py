# ui/main_gui.py
import gradio as gr
from core.pipeline import Pipeline
# Assuming Config is needed for Pipeline initialization indirectly
# from config.config import Config

class UI:
    def __init__(self, config):
        self.config = config
        # Initialize the pipeline which now handles the full process
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Speech Transcription and Emotion Analysis")

            # Consolidate into a single main processing tab
            with gr.Tab("Process Audio"):
                gr.Markdown("Upload an audio file OR enter a YouTube URL to transcribe, diarize, and analyze emotion.")
                input_file = gr.File(label="Upload Audio File (.wav, .mp3, etc.)")
                youtube_url = gr.Textbox(label="Or Enter YouTube URL")
                process_btn = gr.Button("Process Audio") # Renamed button

                # Outputs: Status message and downloadable results file
                status_output = gr.Textbox(label="Processing Status")
                download_output = gr.File(label="Download Results (ZIP)")

                # Wrapper function to call the consolidated pipeline method
                def process_wrapper(input_file_obj, url):
                    input_source = None
                    if input_file_obj is not None:
                        # Use the temporary path provided by Gradio File component
                        input_source = input_file_obj.name
                        print(f"Processing uploaded file: {input_source}") # Debug print
                    elif url:
                        input_source = url
                        print(f"Processing YouTube URL: {input_source}") # Debug print
                    else:
                        return "Please provide either an audio file or a YouTube URL.", None # Return tuple for both outputs

                    if input_source:
                        try:
                            # Call the main processing method in the pipeline
                            zip_path, status_message = self.pipeline.process_audio(input_source)
                            # Return status message for Textbox, zip_path for File component
                            # If zip_path is None (due to error), File component will be empty
                            return status_message, zip_path
                        except Exception as e:
                             # Catch unexpected errors during the call itself
                             print(f"Error during pipeline processing: {e}") # Log error
                             return f"An unexpected error occurred: {e}", None
                    else:
                        # This case should have been caught above, but as a fallback:
                        return "No input provided.", None


                # Update button click handler
                process_btn.click(
                    fn=process_wrapper,
                    inputs=[input_file, youtube_url],
                    outputs=[status_output, download_output] # Map to both output components
                )

            # Removed Diarization Tab
            # Removed Emotion Analysis Tab

        return demo

    def launch(self):
        # Set share=True if you want a public link (useful for testing)
        self.interface_ui().launch()