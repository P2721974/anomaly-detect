# models/random_forest.py

import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from models.base_model import BaseModel
from utils.metrics_utils import plot_classification_report
from utils.file_saver import save_pickle, save_json, ensure_dir
from utils.progress import tqdm_bar
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))

class RandomForestModel(BaseModel):
    """
    Random Forest classifier implementation of BaseModel.

    Suitable for supervised learning on labelled datasets.
    """

    def __init__(self, **kwargs):
        """
        Initialises the Random Forest model with filtered keyword arguments.

        Parameters:
            **kwargs: Hyperparameters for RandomForestClassifier. Unrelated keys are ignored.
        """
        # Strip out unrelated kwargs like 'input_dim'
        rf_kwargs = {k: v for k, v in kwargs.items() if k in RandomForestClassifier().get_params()}
        self.model = RandomForestClassifier(**rf_kwargs)
        self.input_dim = None
        self.metadata = {}
        logger.info("Initialised Random Forest model with params: %s", rf_kwargs)

    def train(self, X, y=None, **kwargs):
        """
        Trains the Random Forest model on labelled data.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.
            y (np.ndarray or pd.Series): Class labels.
            **kwargs: Additional hyperparameters for RandomForestClassifier.
        """
        if y is None:
            raise ValueError("Supervised training requires labels (y).")
        
        self.input_dim = X.shape[1]

        n_estimators = config['training']['n_estimators']

        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            warm_start=True,
            **{k: v for k, v in kwargs.items() if k in RandomForestClassifier().get_params()}
            )
        
        logger.info("Training %s trees on data with %d samples", n_estimators, len(X))

        for i in tqdm_bar(range(1, n_estimators + 1), desc="Training Trees", unit="tree"):
            self.model.n_estimators = i
            self.model.fit(X, y)

        logger.info("Training complete.")

    def predict(self, X):
        """
        Generates class predictions from the trained model.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        logger.info("Predicting using trained Random Forest model on %d samples", len(X))
        return self.model.predict(X)

    def evaluate(self, X, y_true, log_metrics=False):
        """
        Evaluates the classifier on labelled data.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input features.
            y_true (np.ndarray or pd.Series): Ground truth labels.
            log_metrics (bool): Whether to log classification report.

        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1 score.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        y_pred = self.model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        logger.info("Random Forest model evaluation complete.")

        if log_metrics:
            logger.info("Random Forest model evaluation:\n%s", classification_report(y_true, y_pred, zero_division=0))

        return metrics
    
    def plot(self, X_val, y_val, output_path=None, title=None):
        """
        Generates and saves a classification report plot.

        Parameters:
            X_val (np.ndarray): Validation feature set.
            y_val (np.ndarray): Validation labels.
            output_path (str, optional): File path to save plot.
            title (str, optional): Title for the plot.
        """
        y_pred = self.model.predict(X_val)
        metrics = self.evaluate(X_val, y_val, log_metrics=True)
        title = title or f"{self.metadata.get('model_type', 'Model')} Evaluation"

        plot_classification_report(metrics, y_val, y_pred, title=title, output_path=output_path)

    def save(self, path,  metrics=None):
        """
        Saves the model, metadata, and optionally evaluation metrics.

        Parameters:
            path (str): Directory to save model files.
            metrics (dict, optional): Evaluation metrics to store in metadata.
        """
        ensure_dir(path)

        model_path = os.path.join(path, "model.pkl")
        save_pickle(self.model, model_path)

        self.metadata = {
            "model_type": "random_forest",
            "model_path": model_path,
            "input_dim": self.input_dim,
            "evaluation_metrics": metrics or {}
        }

        metadata_path = os.path.join(path, 'metadata.json')
        save_json(self.metadata, metadata_path)

        logger.info("Random Forest model, metadata, and metrics plot saved to: %s", path)

    def load(self, path):
        """
        Loads the model and metadata from disk.

        Parameters:
            path (str): Directory to load model files from.
        """
        model_path = os.path.join(path, 'model.pkl')
        metadata_path = os.path.join(path, 'metadata.json')

        self.model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.input_dim = self.metadata.get("input_dim")

        logger.info("Random Forest model loaded from: %s", path)

    def get_metadata(self, path) -> dict:
        """
        Returns stored model metadata.

        Parameters:
            path (str): Base path to construct fallback metadata if none exists.

        Returns:
            dict: Metadata dictionary.
        """
        return self.metadata or {
            "model_type": "random_forest",
            "model_path": os.path.join(path, "model.pkl"),
            "input_dim": self.input_dim,
            "evaluation_metrics": {}
        }
