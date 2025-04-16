# config/config.py
import json
import os
from pathlib import Path

# --- ADD LOGGING IMPORTS ---
# Note: Logging might not be configured when Config is initialized,
# so we use print for initial load/validation messages.
# We can import log_error for use in save_config though.
try:
    # Use relative import if logging is within the same top-level package
    from core.logging import log_error
    # Or absolute if structure requires: from project_name.core.logging import log_error
except ImportError:
    # Fallback if core.logging is not available during isolated config use
    def log_error(message): print(f"ERROR: {message}") # Simple print fallback

class Config:
    def __init__(self, config_file="config.json"):
        # ... (keep prints in __init__, _load_config, _validate_config) ...
        self.config_file = Path(config_file)
        self.config = {}
        self._load_config()
        self._validate_config()

    def _load_defaults(self):
        # ... (no changes needed here) ...
        return {
            # ... defaults ...
        }

    def _load_config(self):
        """Loads configuration from file or sets defaults."""
        defaults = self._load_defaults()

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                self.config = {**defaults, **loaded_config}
                print(f"INFO: Loaded configuration from {self.config_file}") # KEEP PRINT (before logging setup)
            except json.JSONDecodeError:
                print(
                    f"WARN: Error decoding JSON from {self.config_file}. Using default configuration." # KEEP PRINT
                )
                self.config = defaults
            except Exception as e:
                print(
                    f"WARN: Error loading {self.config_file}: {e}. Using default configuration." # KEEP PRINT
                )
                self.config = defaults
        else:
            print(
                f"INFO: Configuration file {self.config_file} not found. Using default configuration." # KEEP PRINT
            )
            self.config = defaults

        # Override with environment variables
        env_device = os.getenv("DEVICE")
        if env_device:
            self.config["device"] = env_device
            print(
                f"INFO: Overriding 'device' with environment variable DEVICE: {env_device}" # KEEP PRINT
            )

        env_hf_token = os.getenv("HF_TOKEN")
        if env_hf_token:
            self.config["hf_token"] = env_hf_token
            print("INFO: Overriding 'hf_token' with environment variable HF_TOKEN.") # KEEP PRINT (token value not printed)

    def _validate_config(self):
        """Validates critical configuration settings after loading."""
        hf_token = self.config.get("hf_token")
        if not hf_token:
            # Keep critical error as print as it prevents app start before logging setup
            print(
                "CRITICAL ERROR: Hugging Face token ('hf_token') is missing. "
                "Please set the HF_TOKEN environment variable or add 'hf_token': 'your_token' to config.json. "
                "Diarization requires a valid Hugging Face token for Pyannote models."
            )
            # Raise error after printing
            raise ValueError("Hugging Face token ('hf_token') is missing.")
        print("INFO: Configuration validated successfully.") # KEEP PRINT

    def save_config(self):
        """Saves the current configuration to the config file."""
        try:
            config_dir = self.config_file.parent
            config_dir.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            # Logging might be configured now, but let's stick to print for consistency within Config
            print(f"INFO: Configuration saved to {self.config_file}") # Keep print here for now
        except Exception as e:
            # Use log_error if available, otherwise print fallback
            log_error(f"Failed to save configuration to {self.config_file}: {e}") # USE LOG_ERROR

    def get(self, key, default=None):
        # ... (no changes needed here) ...
        return self.config.get(key, default)

    def set(self, key, value):
        """Sets a configuration value by key and saves the config file"""
        self.config[key] = value
        self.save_config()