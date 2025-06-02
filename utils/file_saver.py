# utils/file_saver.py

import os
import glob
import joblib
import json
from datetime import datetime
from scapy.all import wrpcap
from tensorflow.keras.models import save_model

from utils.logger import get_logger

logger = get_logger(__name__, "INFO")


def ensure_dir(path):
    """
    Ensures the directory exists for a given file or directory path.
    If a file path is passed, its parent directory is created.
    """
    dir_path = path if os.path.isdir(path) else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.debug("Created directory: %s", dir_path)


def save_pcap(pcap, path):
    ensure_dir(path)
    wrpcap(path, pcap)
    logger.info("PCAP saved: %s", path)


def save_pickle(obj, path):
    ensure_dir(path)
    joblib.dump(obj, path)
    logger.info("Pickle saved: %s", path)


def save_json(obj, path):
    ensure_dir(path)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    logger.info("JSON saved: %s", path)


def save_text(text, path):
    ensure_dir(path)
    with open(path, 'w') as f:
        f.write(str(text))
    logger.info("Text saved: %s", path)


def save_dataframe(df, path):
    ensure_dir(path)
    df.to_csv(path, index=False)
    logger.info("CSV saved: %s", path)


def save_keras_model(model, path):
    ensure_dir(path)
    save_model(model, path)
    logger.info("Keras model saved: %s", path)


def generate_incremented_path(base_path, extension=None):
    """
    Generates a timestamped, incremented output path.

    Parameters:
    - base_path (str): Path like 'data/output/myfile.csv' or 'data/models/model'
    - extension (str): Optional override extension (e.g., '.csv', '.pcap')

    Returns:
    - str: Path like 'data/output/myfile_20250525_151045_1.csv'
    """
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    ext = extension or os.path.splitext(base_path)[-1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pattern = os.path.join(base_dir, f"{base_name}_{timestamp}_*{ext}")
    existing = glob.glob(pattern)
    next_id = len(existing) + 1

    ensure_dir(os.path.join(base_dir, 'placeholder.tmp'))  # ensure dir exists
    return os.path.join(base_dir, f"{base_name}_{timestamp}_{next_id}{ext}")


def safe_save_path(base_path, extension=None):
    """
    Ensures file is saved safely. If file exists, appends timestamped increment.

    Parameters:
    - base_path (str): Suggested full file path
    - extension (str): Optional extension override

    Returns:
    - str: Safe, unique file path (original or incremented)
    """
    if os.path.exists(base_path):
        logger.warning("[!] File already exists: %s. Generating new path...", base_path)
        return generate_incremented_path(base_path, extension)
    return base_path


def get_base_filename(path):
    """
    Returns the base filename (without extension) from a given file path.

    Example:
        "data/output/my_dataset_20250525_153045.csv" → "my_dataset_20250525_153045"
    """
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]