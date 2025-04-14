# core/utils.py
import os
import tempfile

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_temp_file(suffix=""):
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name