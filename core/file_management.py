# core/file_management.py
import os
import shutil
from .utils import create_directory, get_temp_file

class FileManager:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get("output_dir")
        self.temp_dir = config.get("temp_dir")
        create_directory(self.output_dir)
        create_directory(self.temp_dir)

    def create_temp_file(self, suffix=""):
        return get_temp_file(suffix=suffix)

    def save_file(self, content, filename):
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def cleanup_temp_files(self):
        shutil.rmtree(self.temp_dir)
        create_directory(self.temp_dir)