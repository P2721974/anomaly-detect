# models/trainer.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from models.loader import instantiate_model
from utils.metrics_utils import pretty_print_metadata
from utils.file_saver import safe_save_path
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def train_autoencoder(input_path, output_path=None):
    logger.info("Starting Autoencoder training on: %s", input_path)

    if not os.path.exists(input_path):
        logger.error("CSV input file not found at %s", input_path)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("Loaded CSV is empty: %s", input_path)
        return

    logger.info("Loaded training data with shape: %s", df.shape)

    # If labelled data, drop labels
    if "label" in df.columns:
        X = df.drop(columns=["label"]).values
    else:
        X = df

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    model = instantiate_model("autoencoder", input_dim=X.shape[1])
    model.train(X_train, X_val=X_val)

    # Evaluate + save
    metrics = model.evaluate(X_val)
    model_dir = output_path or config['training']['save_dir'] + "autoencoder/autoencoder_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nAutoencoder evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    return metrics


def train_random_forest(input_path, output_path=None):
    logger.info("Starting Random Forest training on: %s", input_path)

    if not os.path.exists(input_path):
        logger.error("CSV input file not found at %s", input_path)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("Loaded CSV is empty: %s", input_path)
        return

    if "label" not in df.columns:
        logger.error("Missing 'label' column for supervised training.")
        return

    logger.info("Loaded training data with shape: %s", df.shape)

    y = df["label"].values
    X = df.drop(columns=["label"]).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = instantiate_model("random_forest", input_dim=X.shape[1])
    model.train(X_train, y=y_train)

    metrics = model.evaluate(X_val, y_val, log_metrics=False)
    model_dir = output_path or config['training']['save_dir'] + "random_forest/random_forest_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nRandom Forest evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    # Save evaluation plots
    plot_path = os.path.join(model_dir, "evaluation_report.png")
    model.plot(X_val, y_val, output_path=plot_path)

    return metrics


def train_svm(input_path, output_path=None):
    logger.info("Starting SVM training on: %s", input_path)

    if not os.path.exists(input_path):
        logger.error("CSV input file not found at %s", input_path)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("Loaded CSV is empty: %s", input_path)
        return

    if "label" not in df.columns:
        logger.error("Missing 'label' column for supervised training.")
        return

    logger.info("Loaded training data with shape: %s", df.shape)

    y = df["label"].values
    X = df.drop(columns=["label"]).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = instantiate_model("svm", input_dim=X.shape[1])
    model.train(X_train, y=y_train)

    metrics = model.evaluate(X_val, y_val, log_metrics=False)
    model_dir = output_path or config['training']['save_dir'] + "svm/svm_model"
    model_dir = safe_save_path(model_dir, extension="")
    model.save(model_dir, metrics=metrics)

    logger.info("\nSVM evaluation metrics:")
    pretty_print_metadata(model.get_metadata(model_dir))

    # Save evaluation plots
    plot_path = os.path.join(model_dir, "evaluation_report.png")
    model.plot(X_val, y_val, output_path=plot_path)

    return metrics


def run_train_model(args):
    model_type = args.model or config['training']['model_type']
    input_path = args.input or config['training']['input']
    output_path = args.output or None

    dispatch = {
        "autoencoder": train_autoencoder,
        "random_forest": train_random_forest,
        "svm": train_svm
    }

    if model_type not in dispatch:
        logger.error("Unknown model type: %s", model_type)
        return

    logger.info("Dispatching training for model: %s", model_type)
    dispatch[model_type](input_path, output_path)
