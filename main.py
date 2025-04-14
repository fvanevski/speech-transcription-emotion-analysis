# main.py
from config.config import Config
from ui.main_gui import UI
from ui.postprocess_gui import PostProcessUI
from core.logging import setup_logging

def main():
    config = Config()
    setup_logging(config)
    app = UI(config)
    app.launch()

if __name__ == "__main__":
    main()