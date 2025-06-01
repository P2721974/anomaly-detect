# models/base_model.py

from abc import ABC, abstractmethod
from typing import Any, Union, Optional
import numpy as np
import pandas as pd
import logging
import joblib


class BaseModel(ABC):
    """
    Abstract base class for all supported ML models in the anomaly detection pipeline.
    All custom models must implement these methods.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Optional base constructor for shared configuration and logger.

        Parameters:
        - config: Dictionary of optional parameters such as thresholds or paths        
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None, X_val: Union[np.ndarray, pd.DataFrame, None] = None) -> None:
        """
        Train the model using the provided data.

        Parameters:
        - X: Features (array-like or DataFrame)
        - y: Labels (optional, required for supervised models)
        - X_val: Optional validation data (used in some models like autoencoders)
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Run inference using the trained model.

        Parameters:
        - X: Features (array-like or DataFrame)

        Returns:
        - Array of predictions or anomaly scores
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to a specified file path.

        Parameters:
        - path: Destination file path for the model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a model from the specified file path.

        Parameters:
        - path: File path of the saved model
        """
        pass

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None) -> np.ndarray:
        """
        Train and immediately predict on the input data.

        Parameters:
        - X: Features
        - y: Labels (optional)

        Returns:
        - Predictions
        """
        self.train(X, y)
        return self.predict(X)

    def get_metadata(self) -> dict:
        """
        Return optional metadata for the model (e.g., threshold, input shape).
        Override in subclasses as needed.

        Returns:
        - Dictionary of metadata
        """
        return {}

    def validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Basic input validation to ensure correct type and dimensionality.

        Parameters:
        - X: Input data to validate

        Raises:
        - ValueError: If input is not a valid array or DataFrame
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input must be a numpy array or pandas DataFrame.")
        if len(X.shape) != 2:
            raise ValueError("Input must be 2-dimensional (samples, features).")

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Any) -> dict:
        """
        Optional method for evaluation. Subclasses may override to return metrics.

        Parameters:
        - X: Features
        - y: True labels

        Returns:
        - Dictionary of evaluation metrics
        """
        raise NotImplementedError("Evaluation not implemented for this model.")

    def save_with_joblib(self, obj: Any, path: str) -> None:
        """
        Save an object using joblib.

        Parameters:
        - obj: Object to serialize
        - path: Path to save to
        """
        joblib.dump(obj, path)
        self.logger.info(f"Model saved to {path}")

    def load_with_joblib(self, path: str) -> Any:
        """
        Load an object using joblib.

        Parameters:
        - path: Path of saved object

        Returns:
        - Deserialized object
        """
        self.logger.info(f"Loading model from {path}")
        return joblib.load(path)
