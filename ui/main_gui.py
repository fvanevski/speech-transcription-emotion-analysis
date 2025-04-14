# ui/main_gui.py
import gradio as gr
from core.pipeline import Pipeline
from config.config import Config

class UI:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Speech Transcription and Emotion Analysis")

            with gr.Tab("Transcription"):
                input_file = gr.File(label="Upload Audio File")
                youtube_url = gr.Textbox(label="Enter YouTube URL")
                transcribe_btn = gr.Button("Transcribe")
                transcribe_output = gr.Textbox(label="Transcription Result")

                def transcribe_wrapper(input_file, youtube_url):
                    if input_file:
                        return self.pipeline.transcribe(input_file.name)
                    elif youtube_url:
                        audio_file = self.pipeline.download_audio_from_youtube(youtube_url)
                        return self.pipeline.transcribe(audio_file)
                    else:
                        return "Please provide either an audio file or a YouTube URL."

                transcribe_btn.click(fn=transcribe_wrapper, inputs=[input_file, youtube_url], outputs=[transcribe_output])

            with gr.Tab("Diarization"):
                input_file = gr.File(label="Upload Audio File")
                diarize_btn = gr.Button("Diarize")
                diarize_output = gr.Textbox(label="Diarization Result")
                diarize_btn.click(fn=self.pipeline.diarize, inputs=[input_file], outputs=[diarize_output])

            with gr.Tab("Emotion Analysis"):
                input_text = gr.Textbox(label="Enter Text")
                analyze_btn = gr.Button("Analyze Emotion")
                emotion_output = gr.Textbox(label="Emotion Result")
                analyze_btn.click(fn=self.pipeline.analyze_emotion, inputs=[input_text], outputs=[emotion_output])

        return demo

    def launch(self):
        self.interface_ui().launch()