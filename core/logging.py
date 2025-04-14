# core/logging.py
import logging
import os
from config.config import Config

def setup_logging(config):
    log_level = config.get("log_level")
    log_file = os.path.join(config.get("output_dir"), "app.log")
    logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)