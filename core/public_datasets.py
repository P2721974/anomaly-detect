# core/public_datasets.py

import os
import zipfile
import requests
import pandas as pd
import numpy as np
import shutil

from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))

DATA_PATH = config['public']['save_path']

DATASETS = {
    "nsl": {
        "name": "nsl-kdd",
        "url": config['public']['nsl_url'],
        "filename": "nsl_kdd.zip",
        "manual": False
    },
    "unsw": {
        "name": "unsw-nb15",
        "url": config['public']['unsw_url'],
        "filename": None,
        "manual": True
    },
    "cicids": {
        "name": "cicids2017",
        "url": config['public']['cicids_url'],
        "filename": None,
        "manual": True
    }
}


def download_dataset(key):
    dataset = DATASETS.get(key.lower())
    if not dataset:
        logger.error("Unknown dataset key: %s", key)
        return

    if dataset.get("manual"):
        logger.warning("%s requires manual download:/n%s", dataset["name"], dataset["url"])
        return

    url = dataset["url"]
    filename = dataset["filename"]
    target_path = os.path.join(DATA_PATH, dataset['name'])

    os.makedirs(target_path, exist_ok=True)
    zip_path = os.path.join(target_path, filename)

    logger.info("Downloading %s dataset...", dataset["name"])
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info("Downloaded to %s", zip_path)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_path)
        logger.info("Extracted dataset to %s", target_path)

    except Exception as e:
        logger.error("Failed to download %s: %s", dataset["name"], e)


def prepare_nsl_kdd(input_path, output_path):
    logger.info("Preparing NSL-KDD from: %s", input_path)

    try:
        col_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]

        df = pd.read_csv(input_path, names=col_names)

        # Drop unused column
        df.drop(columns=["difficulty"], inplace=True)

        # Binary classification: normal = 0, attack = 1
        df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

        df.to_csv(output_path, index=False)
        logger.info("NSL-KDD prepared and saved to: %s", output_path)

    except Exception as e:
        logger.error("Failed to prepare NSL-KDD: %s", e)

def cleanup_nsl(path):
    # Remove extracted folder
    extracted_folder = os.path.join(os.path.dirname(path), "nsl-kdd/NSL_KDD-master")
    print(extracted_folder)
    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)
        logger.info("Deleted extracted folder: %s", extracted_folder)

    # Remove downloaded zip
    zip_path = os.path.join(os.path.dirname(path), "nsl-kdd/nsl_kdd.zip")
    print(zip_path)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        logger.info("Deleted archive file: %s", zip_path)


def prepare_unsw(input_path, output_path):
    """
    Prepares the labeled UNSW-NB15 training/testing sets:
    - Assumes proper headers are included
    - One-hot encodes 'proto', 'state', 'service'
    - Converts all other fields to numeric safely
    - Drops rows with missing or malformed values
    - Saves a clean labeled dataset
    """
    try:
        df = pd.read_csv(input_path, low_memory=False)

        # Drop metadata columns
        drop_cols = ["id", "srcip", "sport", "dstip", "dsport"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Normalize attack_cat (optional)
        if "attack_cat" in df.columns:
            df["attack_cat"] = df["attack_cat"].fillna("unknown").str.lower()

        # One-hot encode categorical
        df = pd.get_dummies(df, columns=["proto", "service", "state"])

        # Convert all remaining fields to numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        missing_before = df.isna().sum().sum()
        df.fillna(0, inplace=True)
        logger.info("Replaced %d missing values with 0", missing_before)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("UNSW-NB15 prepared and saved to: %s", output_path)

    except Exception as e:
        logger.error("Failed to prepare UNSW-NB15: %s", e)


def prepare_cicids(input_path, output_path):
    """
    Cleans and labels a CICIDS2017 ISCX CSV file into a usable dataset.
    """
    try:
        logger.info("Loading CICIDS2017 file: %s", input_path)
        df = pd.read_csv(input_path, low_memory=False)
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            logger.error("No 'Label' column found in CSV.")
            return

        logger.info("Converting 'Label' column to binary...")
        df["label"] = df["Label"].apply(lambda x: 0 if x.strip().upper() == "BENIGN" else 1)

        logger.info("Dropping non-numeric features and cleaning data...")
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("CICIDS2017 ICSX CSV prepared and saved to: %s", output_path)
    
    except Exception as e:
        logger.error("Failed to prepare CICIDS2017: %s", e)


def handle_public_dataset(args):
    name = args.name.lower()
    save_dir = config['public']['save_path']

    if args.download:
        download_dataset(name)
    
    if args.prepare:
        if args.name == "nsl":
            raw_path = os.path.join(save_dir, "nsl-kdd/NSL_KDD-master/KDDTrain+.txt")
            output_path = os.path.join(save_dir, "nsl-kdd/nsl_train_clean.csv")
            prepare_nsl_kdd(raw_path, output_path)

            raw_path = os.path.join(save_dir, "nsl-kdd/NSL_KDD-master/KDDTest+.txt")
            output_path = os.path.join(save_dir, "nsl-kdd/nsl_test_clean.csv")
            prepare_nsl_kdd(raw_path, output_path)

            raw_path = os.path.join(save_dir, "nsl-kdd/NSL_KDD-master/Small Training Set.csv")
            output_path = os.path.join(save_dir, "nsl-kdd/nsl_small_clean.csv")
            prepare_nsl_kdd(raw_path, output_path)

            cleanup_nsl(save_dir)
        
        elif args.name == "unsw":
            raw_path = os.path.join(save_dir, "unsw-nb15/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv")
            output_path = os.path.join(save_dir, "unsw-nb15/unsw_train_clean.csv")
            prepare_unsw(raw_path, output_path)

            raw_path = os.path.join(save_dir, "unsw-nb15/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv")
            output_path = os.path.join(save_dir, "unsw-nb15/unsw_test_clean.csv")
            prepare_unsw(raw_path, output_path)

        elif args.name == "cicids":
            raw_path = os.path.join(save_dir, "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_1.csv")
            prepare_cicids(raw_path, output_path)

            raw_path = os.path.join(save_dir, "cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_2.csv")
            prepare_cicids(raw_path, output_path)

            raw_path = os.path.join(save_dir, "cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_3.csv")
            prepare_cicids(raw_path, output_path)

            raw_path = os.path.join(save_dir, "cicids2017/Monday-WorkingHours.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_4.csv")
            prepare_cicids(raw_path, output_path)

            raw_path = os.path.join(save_dir, "cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_5.csv")
            prepare_cicids(raw_path, output_path)

            raw_path = os.path.join(save_dir, "cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_6.csv")
            prepare_cicids(raw_path, output_path)

            raw_path = os.path.join(save_dir, "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_7.csv")
            prepare_cicids(raw_path, output_path)

            raw_path = os.path.join(save_dir, "cicids2017/Wednesday-workingHours.pcap_ISCX.csv")
            output_path = os.path.join(save_dir, "cicids2017/clean/cicids_clean_8.csv")
            prepare_cicids(raw_path, output_path)

        else:
            logger.error("[!] Dataset '%s' is not yet supported for preparation", name)