# models/model_setup.py

import os
from shutil import copyfile

from utils.config_loader import get_config

config = get_config()

PRETRAINED_MODELS = {
    "autoencoder": config['registry']['models']['autoencoder_cicids.keras']['path'],
    "random_forest": config['registry']['models']['rf_unsw.pkl']['path'],
    "svm": config['registry']['models']['svm_nsl.pkl']['path']
}

def run_model_setup(args):
    if args.info:
        print("Available pretrained models:")
        for key, path in PRETRAINED_MODELS.items():
            print(f"  {key}: {path}")

    if args.use:
        model_name = args.use.lower()
        if model_name not in PRETRAINED_MODELS:
            print(f"[!] Unknown model: {model_name}. Available options: {list(PRETRAINED_MODELS.keys())}")
            return

        model_path = PRETRAINED_MODELS[model_name]
        ext = os.path.splitext(model_path)[-1]
        target = os.path.join(config['training']['save_dir'], f"{model_name}_model_active{ext}")
        os.makedirs(os.path.dirname(target), exist_ok=True)        

        if os.path.exists(model_path):
            copyfile(model_path, target)
            print(f"[+] Activated pretrained model '{model_name}' at: {target}")
        else:
            print(f"[!] Model file not found: {model_path}")
