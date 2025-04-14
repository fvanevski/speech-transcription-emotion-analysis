# speech-transcription-emotion-analysis
Speech Transcription and Emotion Analysis System

This project is a modular Python-based system for speech transcription, speaker diarization, and emotion analysis. It uses WhisperX for transcription and diarization, and BERT for emotion analysis. The system also includes a user-friendly interface built with Gradio.

Features
Speech Transcription: Converts audio input (YouTube links or uploaded files) into text.
Speaker Diarization: Identifies and separates speakers in the audio.
Emotion Analysis: Analyzes the emotional tone of each segment using BERT.
Batch Processing: Supports processing multiple audio files or URLs.
Speaker Relabeling: Allows users to relabel speakers in the transcript.
User Interface: Provides an intuitive Gradio-based UI for interaction.
Error Handling and Logging: Includes robust error handling and detailed logging.
Project Structure
project-root/
├── core/
│   ├── transcription.py        # Core transcription and diarization logic
│   ├── emotion_analysis.py     # Emotion analysis pipeline
│   └── utils.py                # Utility functions (e.g., file management, logging)
├── ui/
│   └── gradio_interface.py     # Gradio-based user interface
├── postprocessing/
│   └── speaker_relabeling.py   # Speaker relabeling logic
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

Installation

Clone the repository:

git clone https://github.com/your-username/speech-transcription-emotion-analysis.git
cd speech-transcription-emotion-analysis


Create a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Install additional tools (e.g., yt-dlp, ffmpeg):

yt-dlp: For downloading YouTube audio
pip install yt-dlp

ffmpeg: For audio conversion
On Ubuntu/Debian:
sudo apt install ffmpeg

On macOS (via Homebrew):
brew install ffmpeg

On Windows: Download FFmpeg
Usage
Running the Application

Launch the Gradio interface:

python ui/gradio_interface.py


Open the provided URL in your browser to access the interface.

Features in the Interface
Transcription Tab:
Input a YouTube URL or upload an audio file.
Process the audio for transcription, diarization, and emotion analysis.
Download the results as a ZIP file.
Speaker Relabeling Tab:
Upload a structured transcript JSON file.
Relabel speakers and download the updated transcript.
Batch Processing

To process multiple audio files or URLs, use the batch processing feature in the Transcription tab. Enter multiple YouTube URLs (one per line) or upload multiple audio files.

Logging

Logs are stored in the logs/ directory. Check these files for detailed information about each transcription job.

Contributing

Contributions are welcome! To contribute:

Fork the repository.
Create a new branch for your feature or bug fix:
git checkout -b feature-name

Commit your changes:
git commit -m "Add new feature"

Push to your branch:
git push origin feature-name

Open a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
WhisperX for transcription and diarization.
Hugging Face Transformers for emotion analysis.
Gradio for the user interface.

Feel free to reach out with any questions or suggestions!
