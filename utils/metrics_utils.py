# utils/metrics_utils.py

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def pretty_print_metadata(metadata: dict, indent: int = 2):
    """
    Makes model metadata and metrics pretty.

    Parameters:
    - metadata: Dictionary containing model info
    - indent: Indentation level for JSON formatting
    """
    if not metadata:
        logger.warning("[!] No metadata found.")
        return

    print(json.dumps(metadata, indent=indent))

def plot_classification_report(metrics: dict, y_true, y_pred, title: str = "Model Evaluation", output_path: str = None):
    """
    Generate a side-by-side visualization of:
    - Classification metrics bar chart
    - Confusion matrix heatmap
    """
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    keys = ["accuracy", "precision", "recall", "f1_score"]
    values = [metrics.get(k, 0) for k in keys]
    sns.barplot(x=keys, y=values, palette="pastel", ax=axes[0])
    axes[0].set_title("Classification Metrics")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=axes[1])
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()