# ui/postprocess_gui.py
import gradio as gr

from config.config import Config
from core.pipeline import Pipeline


class PostProcessUI:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Speaker Labeling & Emotion Summary")

            transcript_json = gr.File(label="Upload structured_transcript_*.json")
            label_csv = gr.File(label="Upload speaker_labels.csv")
            labeled_output = gr.File(label="Download Relabeled Files (ZIP)")
            apply_btn = gr.Button("Apply Speaker Labels")

            template_csv = gr.File(label="Download Speaker CSV Template")
            generate_btn = gr.Button("Generate Label Template")

            generate_btn.click(
                fn=self.pipeline.generate_csv_template,
                inputs=[transcript_json],
                outputs=[template_csv],
            )
            apply_btn.click(
                fn=self.pipeline.apply_labels_from_csv,
                inputs=[transcript_json, label_csv],
                outputs=[labeled_output],
            )

        return demo

    def launch(self):
        self.interface_ui().launch()
