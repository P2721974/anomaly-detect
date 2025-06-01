# models/loader.py

from models.random_forest import RandomForestModel
from models.autoencoder import AutoencoderModel
from models.svm import SVMModel
from models.base_model import BaseModel
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "autoencoder": AutoencoderModel,
    "svm": SVMModel
}

def get_model_class(name):
    """
    Retrieve the model class from the registry based on name.

    Parameters:
    - name: Name of the model (case-insensitive)

    Returns:
    - A class object inheriting from BaseModel
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not supported. Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[name]
    if not issubclass(cls, BaseModel):
        raise TypeError(f"Model '{name}' does not implement BaseModel.")
    return cls

def instantiate_model(name, **kwargs):
    """
    Instantiate a model with optional constructor arguments.

    Parameters:
    - name: Name of the model
    - kwargs: Parameters passed to the model's constructor

    Returns:
    - An instance of the selected model
    """
    logger.info("Instantiating model: %s", name)
    return get_model_class(name)(**kwargs)

def load_model_instance(name: str, model_path: str) -> BaseModel:
    """
    Load a previously saved model from the specified path.

    Parameters:
    - name: Name of the model type (for correct loader)
    - model_path: Path to the model prefix (excluding _model.h5, etc.)

    Returns:
    - Loaded model instance
    """
    model = instantiate_model(name)
    model.load(model_path)
    logger.info("Model '%s' loaded from %s", name, model_path)
    return model
