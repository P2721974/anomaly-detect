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


def pretty_print_metadata(metadata: dict, indent: int = 4):
    """
    Makes model metadata and metrics pretty (formatted JSON).

    Parameters:
        metadata (dict): Dictionary containing model metadata and metrics.
        indent (int): Number of spaces for indentation (default is 4).
    """
    if not metadata:
        logger.warning("[!] No metadata found.")
        return
    print(json.dumps(metadata, indent=indent))

def plot_classification_report(metrics: dict, y_true, y_pred, title: str = "Model Evaluation", output_path: str = None):
    """
    Visualises model metrics using a bar chart and confusion matrix.

    Parameters:
        metrics (dict): Dictionary with 'accuracy', 'precision', 'recall', and 'f1_score'.
        y_true (list or np.ndarray): Ground truth labels.
        y_pred (list or np.ndarray): Predicted labels.
        title (str, optional): title for the entire plot.
        output_path (str, optional): Destination file path to save to, instead of displaying plot.
    """
    sns.set_theme(style="whitegrid")
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