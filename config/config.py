# config/config.py
import os
import json

class Config:
    def __init__(self, config_file="config.json"):
        """
        Initializes the Config class.

        Args:
            config_file (str): The path to the configuration file.
        """
        self.config_file = config_file
        self.config = {} # Initialize empty config dict
        self._load_config() # Load existing or create default config
        self._validate_config() # Validate critical settings like HF_TOKEN

    def _load_defaults(self):
        """Returns a dictionary with default configuration values."""
        return {
            "output_dir": "output",
            "temp_dir": "temp",
            "batch_size": 16, # Default from whisperx, adjust if needed
            "log_level": "INFO",
            "device": "cpu", # Default device
            "hf_token": None, # Default to None, will be checked/overridden
            "min_diarization_duration": 5.0 # Example: add other params
        }

    def _load_config(self):
        """Loads configuration from file or sets defaults."""
        defaults = self._load_defaults()

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                # Merge defaults with loaded config (loaded values override defaults)
                self.config = {**defaults, **loaded_config}
                print(f"INFO: Loaded configuration from {self.config_file}")
            except json.JSONDecodeError:
                print(f"WARN: Error decoding JSON from {self.config_file}. Using default configuration.")
                self.config = defaults
            except Exception as e:
                print(f"WARN: Error loading {self.config_file}: {e}. Using default configuration.")
                self.config = defaults
        else:
            print(f"INFO: Configuration file {self.config_file} not found. Using default configuration.")
            self.config = defaults
            # Optionally save the default config file on first run
            # self.save_config() # Uncomment if you want to auto-create config.json

        # --- Override specific keys with environment variables (Environment takes precedence) ---
        # Device
        env_device = os.getenv("DEVICE")
        if env_device:
            self.config["device"] = env_device
            print(f"INFO: Overriding 'device' with environment variable DEVICE: {env_device}")

        # Hugging Face Token (Mandatory)
        env_hf_token = os.getenv("HF_TOKEN")
        if env_hf_token:
            self.config["hf_token"] = env_hf_token
            # Avoid printing the token itself for security
            print("INFO: Overriding 'hf_token' with environment variable HF_TOKEN.")
        # Note: Validation for hf_token happens in _validate_config()

    def _validate_config(self):
        """Validates critical configuration settings after loading."""
        # Validate Hugging Face Token (Option 1: Strict Requirement)
        hf_token = self.config.get("hf_token")
        if not hf_token: # Check if None or empty string
            raise ValueError(
                "CRITICAL ERROR: Hugging Face token ('hf_token') is missing. "
                "Please set the HF_TOKEN environment variable. "
                "Diarization requires a valid Hugging Face token."
            )
        # Optional: Add more validation for other keys (e.g., check if log_level is valid)
        print("INFO: Configuration validated successfully.")


    def save_config(self):
        """Saves the current configuration to the config file."""
        try:
            # Ensure output directory exists before saving config there (if config_file is relative)
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                 os.makedirs(config_dir)
                 
            with open(self.config_file, "w", encoding="utf-8") as f:
                # Use ensure_ascii=False for wider compatibility if needed
                json.dump(self.config, f, indent=2, ensure_ascii=False) 
            print(f"INFO: Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"ERROR: Failed to save configuration to {self.config_file}: {e}")


    def get(self, key, default=None):
        """
        Gets a configuration value by key.

        Args:
            key (str): The configuration key.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default.
        """
        return self.config.get(key, default)

    def set(self, key, value):
         """ Sets a configuration value by key and saves the config file """
         self.config[key] = value
         self.save_config() # Save after setting a value