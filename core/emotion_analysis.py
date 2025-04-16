# core/emotion_analysis.py
from transformers import pipeline


class EmotionAnalysis:
    def __init__(self, config):
        self.config = config
        self.emotion_classifier = pipeline(
            "text-classification", model="j-hartmann/emotion-english-distilroberta-base"
        )

    def analyze_emotion(self, text):
        result = self.emotion_classifier(text)
        return result[0]["label"]
