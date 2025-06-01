# --- This block can be removed without affecting pipeline in any way ---

# Suppress Scapy runtime warnings
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# Suppress deprecation warnings from Scapy-imported cryptography library
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Suppress Pandas deprecation warning about silent downcasting (fillna)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Suppress Tensorflow/Keras suggestion to swap to .legacy.Adam optimiser on M1/M2 chips
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# --- End of warning suppression block ---

# cli/main.py

import argparse

import __version__
from core.capture import run_capture, run_live_capture
from core.preprocessor import run_preprocessor
from core.dataset_utils import run_dataset_utils
from core.public_datasets import handle_public_dataset
from models.trainer import run_train_model
from models.detector import run_detection, run_live_detection
from utils.registry import list_pretrained_models, list_public_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Anomaly-Based Threat Detection CLI (v{__version__})"
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config file (default: config/config.yml)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"anomaly-detect v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Capture
    capture_parser = subparsers.add_parser("capture", help="Capture network traffic")
    capture_parser.add_argument("--live", action="store_true", help="Enable live packet monitoring")
    capture_parser.add_argument("--interface", help="Network interface to capture/monitor with")
    capture_parser.add_argument("--duration", type=int, help="Capture duration (seconds)")
    capture_parser.add_argument("--packet-count", type=int, help="Max packets to capture")
    capture_parser.add_argument("--output", help="Optional output PCAP filepath (incompatible with live monitoring)")

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Convert PCAP to CSV features")
    preprocess_parser.add_argument("--input", help="Input PCAP file")
    preprocess_parser.add_argument("--label", help="Optional label to assign to all extracted packets")
    preprocess_parser.add_argument("--output", help="Optional override directory to save output CSV file")

    # Dataset Utils
    dataset_parser = subparsers.add_parser("dataset", help="Dataset management and transformation utilities")
    dataset_parser.add_argument("--combine", nargs='+', help="List CSVs or folder paths to combine")
    dataset_parser.add_argument("--balance", action="store_true", help="Balance class labels after combining")
    dataset_parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test")
    dataset_parser.add_argument("--output", help="Base output filename for dataset results")

    # Public Datasets
    public_parser = subparsers.add_parser("public-dataset", help="Download or preprocess a public dataset")
    public_parser.add_argument("--name", required=True, help="Dataset name: nsl, unsw, cicids")
    public_parser.add_argument("--download", action="store_true", help="Download public dataset")
    public_parser.add_argument("--prepare", action="store_true", help="Preprocess public dataset")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a machine learning model")
    train_parser.add_argument("--model", required=True, help="Model type (random_forest, autoencoder, svm)")
    train_parser.add_argument("--input", required=True, help="Training CSV file")
    train_parser.add_argument("--output", help="Optional override path to save model")

    # Detect
    detect_parser = subparsers.add_parser("detect", help="Run detection using a trained model")
    detect_parser.add_argument("--model", required=True, help="Model type (random_forest, autoencoder, svm)")
    detect_parser.add_argument("--model-path", required=True, help="Path to model file")
    detect_parser.add_argument("--live", action="store_true", help="Enable live packet detection")
    detect_parser.add_argument("--interface", help="Network interface to monitor with")
    detect_parser.add_argument("--input", help="Input PCAP filepath for detection (incompatible with live detection)")
    detect_parser.add_argument("--output", help="Optional output CSV filepath (incompatible with live detection)")

    # List Pretrained Models
    subparsers.add_parser("list-pretrained", help="List local pretrained models")

    # List Pre-loaded Datasets
    subparsers.add_parser("list-datasets", help="List local public datasets")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "capture":
        if args.live:
            run_live_capture(args)
        else:
            run_capture(args)

    elif args.command == "preprocess":
        run_preprocessor(args)

    elif args.command == "dataset":
        run_dataset_utils(args)

    elif args.command == "public-dataset":
        handle_public_dataset(args)

    elif args.command == "train":
        run_train_model(args)

    elif args.command == "detect":
        if args.live:
            run_live_detection(args)
        else:
            run_detection(args)

    elif args.command == "list-pretrained":
        print("Available pretrained models:")
        for model in list_pretrained_models():
            print(f"- {model}")

# === make dynamic === 
    elif args.command == "list-datasets":
        print("Available public datasets:")
        for dataset in list_public_datasets():
            print(f"- {dataset}")


if __name__ == "__main__":
    main()
