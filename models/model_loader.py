# models/model_loader.py

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
    Retrieves a model class from the registry by name.

    Parameters:
        name (str): Name of the model (case-insensitive).

    Returns:
        Type[BaseModel]: Model class inheriting from BaseModel.
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
        name (str): Name of the model.
        **kwargs: Keyword arguments passed to the model constructor.

    Returns:
        BaseModel: Instantiated model object.
    """
    logger.info("Instantiating model: %s", name)
    return get_model_class(name)(**kwargs)
