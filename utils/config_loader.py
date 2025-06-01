# utils/config_loader.py
import os
import yaml
from utils.logger import get_logger

logger = get_logger(__name__, "INFO")
CONFIG_PATH = os.path.join("config", "config.yml")
_config_cache = None

def get_config():
    """
    Load and return the configuration dictionary from config.yml.
    Uses a global cache to avoid repeated disk reads.

    Returns:
    - dict: Parsed YAML configuration
    """
    global _config_cache
    if _config_cache is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"[!] config.yml not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f:
            _config_cache = yaml.safe_load(f)
        logger.info("Configuration loaded from: %s", CONFIG_PATH)
    return _config_cache
