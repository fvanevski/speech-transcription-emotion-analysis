import logging
import os
from config.config import Config

def setup_logging(config):
    log_level = config.get("log_level")
    log_dir = config.get("output_dir")
    log_file = os.path.join(log_dir, "app.log")

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)