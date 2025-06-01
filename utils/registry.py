# utils/registry.py

from utils.config_loader import get_config

config = get_config()

def list_pretrained_models():
    return list(config['registry']['models'])

def get_model_metadata(name):
    return config['registry']['models'].get(name)

def list_public_datasets():
    return list(config['registry']['datasets'])

def get_dataset_metadata(name):
    return config['registry']['datasets'].get(name)