# core/dataset_utils.py

import os
import glob
import pandas as pd
from datetime import datetime
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from utils.progress import tqdm_bar
from utils.file_saver import safe_save_path
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def prepare_dataset_path(base_path=None):
    base_path = base_path or config['dataset_utils']['output_path']
    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path).replace('.csv', '')
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing = glob.glob(os.path.join(base_dir, f"{base_name}_{timestamp}_*.csv"))
    next_id = len(existing) + 1
    return os.path.join(base_dir, f"{base_name}_{timestamp}_{next_id}.csv")


def load_dataset(path):
    if os.path.isfile(path):
        return pd.read_csv(path)
    elif os.path.isdir(path):
        all_csvs = glob.glob(os.path.join(path, '*.csv'))
        return pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def build_combined_dataset(sources, output_path, deduplicate=True):
    dataframes = []

    for src in tqdm_bar(sources, desc="Building dataset", unit="file"):
        try:
            df = load_dataset(src)
            df['source'] = os.path.basename(src)
            dataframes.append(df)
        except Exception as e:
            logger.warning("Skipped %s due to error: %s", src, e)

    if dataframes:
        combined = pd.concat(dataframes, ignore_index=True)
        logger.info("Combined dataset shape before cleaning: %s", combined.shape)
        
        # Drop duplicates
        combined.drop_duplicates(inplace=True)

        # Drop the 'source' column
        combined.drop(columns=['source'], inplace=True, errors='ignore')

        # Ensure all values are numeric
        try:
            combined = combined.apply(pd.to_numeric, errors='raise')
        except Exception as e:
            logger.error("Combined dataset contains non-numeric values: %s", e)
            raise

        logger.info("Cleaned combined dataset shape: %s", combined.shape)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_csv(output_path, index=False)
        logger.info("Combined dataset saved to: %s", output_path)
        return combined
    else:
        logger.warning("No valid datasets found to combine.")
        return pd.DataFrame()


def split_dataset(df, label_col='label', train_size=0.8, test_size=0.2, stratify=True, random_state=42):
    """
    Splits dataset into train, validation, and test sets.

    Returns:
    - train_df, val_df, test_df
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    if not (0.99 < (train_size + test_size) < 1.01):
        raise ValueError("Split ratios must sum to 1.")

    stratify_labels = df[label_col] if stratify else None

    # Split: train + test
    df_train, df_test = train_test_split(
        df, train_size=train_size, stratify=stratify_labels, random_state=random_state
    )

    logger.info("Split complete: train=%d, test=%d", len(df_train), len(df_test))
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def balance_labels(df, label_col='label', random_state=42):
    """
    Undersamples the majority class to create a 50/50 balanced dataset.

    Parameters:
    - df: Input DataFrame with a label column
    - label_col: Column name for the binary class label (default = 'label')
    - random_state: Seed for reproducibility

    Returns:
    - A balanced DataFrame
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    class_counts = df[label_col].value_counts()
    if len(class_counts) != 2:
        raise ValueError("Label column must have exactly two classes to balance.")

    logger.info("Class distribution before balancing:\n%s", class_counts)

    # Identify majority and minority
    class_0, class_1 = class_counts.index
    df_minority = df[df[label_col] == class_0]
    df_majority = df[df[label_col] == class_1]

    if len(df_majority) < len(df_minority):
        df_minority, df_majority = df_majority, df_minority  # swap

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=random_state
    )

    df_balanced = pd.concat([df_minority, df_majority_downsampled])
    logger.info("Balanced dataset created with %d samples per class", len(df_minority))

    return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def run_dataset_utils(args):
    """
    Dispatcher for dataset utility actions:
    - combine
    - balance
    - split
    Uses safe saving and config defaults for output paths.
    """
    default_output = config['dataset_utils']['output_path']
    output_path = args.output or safe_save_path(default_output)

    df = None

    if args.combine:
        logger.info("Combining datasets from: %s", args.combine)
        build_combined_dataset(args.combine, output_path)
        df = pd.read_csv(output_path)

    if args.balance:
        if df is None:
            df = pd.read_csv(output_path)
        df = balance_labels(df)
        output_path = safe_save_path(output_path.replace(".csv", "_balanced.csv"))
        df.to_csv(output_path, index=False)
        logger.info("Saved balanced dataset to: %s", output_path)

    if args.split:
        if df is None:
            df = pd.read_csv(output_path)
        train_df, test_df = split_dataset(df)
        base = output_path.replace(".csv", "")
        train_path = safe_save_path(f"{base}_train.csv")
        test_path = safe_save_path(f"{base}_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Saved split datasets:\n- %s\n- %s", train_path, test_path)

    if not (args.combine or args.balance or args.split):
        logger.warning("No dataset operation specified.")
    else:
        logger.info("Dataset transformation complete.")
