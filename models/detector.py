# models/detector.py

import os
import numpy as np
import pandas as pd

from core.capture import live_packet_monitor
from core.preprocessor import preprocess_file, extract_packet_features, clean_dataframe
from models.model_loader import instantiate_model
from siem.wazuh_forwarder import forward_alert
from utils.packet_utils import print_packet_summary, drop_columns
from utils.file_saver import safe_save_path, save_dataframe
from utils.progress import tqdm_bar
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def detection(model, df, model_type, model_name, model_path, output_path=None, live=False):
    """
    Run anomaly detection on a preprocessed DataFrame using a loaded model.

    Parameters:
        model: Loaded model for prediction.
        df (pd.DataFrame): Input DataFrame.
        model_type (str): Model category (e.g., "autoencoder").
        model_name (str): Model filename.
        model_path (str): Path to model directory.
        output_path (str, optional): Path to save prediction results.
        live (bool): Indicates if detection is in live mode.

    Returns:
        pd.DataFrame: DataFrame with prediction and optionally anomaly scores.
    """
    if df.empty:
        logger.debug("Empty DataFrame passed. Skipping.")
        return df

    # Drop string-based ips to avoid dimension mismatch
    df = drop_columns(df, ['src', 'dst'])

    # If csv contains labels - remove label column before detection
    df = drop_columns(df, ['label'])
        
    X = df.values

    metadata = model.get_metadata(model_path)

    predictions = model.predict(X)

    # Autoencoder-specific postprocessing
    if model_type == "autoencoder":
        reconstructions = model.model.predict(model.scaler.transform(X), verbose=0)
        mse = np.mean(np.square(model.scaler.transform(X) - reconstructions), axis=1)
        df['anomaly_score'] = mse

        threshold = metadata.get("threshold", config['detection']['threshold'])
        predictions = (mse > threshold).astype(int)

    df['prediction'] = predictions

    if not live:
        # Simulate per-row progress when running detection on a file
        for _, row in tqdm_bar(df.iterrows(), total=len(df), desc="Running Detection", unit="row"):
            if row['prediction'] == 1:
                logger.debug("Anomaly detected at timestamp %s", row.get("timestamp"))

        save_dataframe(df, output_path)
        logger.info("Predictions saved to: %s", output_path)

    for _, row in df[df['prediction'] == 1].iterrows():
        alert = {
            "source_ip": row.get("source_ip", "unknown"),
            "destination_ip": row.get("destination_ip", "unknown"),
            "source_port": row.get("source_port", "unknown"),
            "destination_port": row.get("destination_port", "unknown"),
            "protocol": row.get("protocol", "unknown"),
            "anomaly_score": row.get("anomaly_score", None),
            "classification": "anomaly",
            "model": model_name,
            "model_type": model_type,
            "timestamp": row.get("timestamp")
        }
        forward_alert(alert)

    logger.info("%d alert(s) forwarded to Wazuh.", len(df[df['prediction'] == 1]))

    return df    


# need to figure out way to avoid defining within a function if possible
# and move to packet utils too if possible
def create_packet_handler(model, model_type, model_name, model_path):
    """
    Returns a packet handler function with access to the detection context.
    
    This lets me run the detection module through a Scapy sniff as the 'prn', 
    and I only define it here as to avoid defining a function within the detection dispatcher.
    """
    def handler(pkt):
        try:
            df = extract_packet_features(pkt)
            if df is None or df.empty:
                    return  # skip unprocessable packets
        
            df = clean_dataframe(df)
            if df.empty:
                return  # skip packets with no usable features

            detection(model, df, model_type, model_name, model_path, live=True)
            print_packet_summary(pkt)

        except Exception as e:
            logger.warning("Live packet processing failed: %s", e)
        
    return handler


def run_detection(args):
    """
    Command-line interface handler for anomaly-based detection on network data.

    Parameters:
        args: Parsed command-line arguments containing 'model_type', 'model_path', 'live', 'interface', 'input', and 'output'
        options.

    This dispatcher handles the flow of the following detection operations:
        - Detection of real-time network traffix using a pre-trained model.
        - Detection of network traffic from an input file ('.pcap' or '.csv') using a pre-trained model.

    Uses config defaults and safe file naming if needed.
    """
    model_type = args.model or config['detection']['model_type']
    model_path = args.model_path or config['detection']['model_path']
    model_name = os.path.basename(model_path)

    logger.info("Loading model: %s", model_name)
    model = instantiate_model(model_type)
    model.load(model_path)

    if args.live:
        interface = args.interface or config['capture']['interface']
        live_packet_handler = create_packet_handler(model, model_type, model_name, model_path)

        live_packet_monitor(interface, live_packet_handler, count=0, timeout=None)

    else:
        batch_size = config['preprocessing']['batch_size']
        input_path = args.input or config['detection']['input_path']
        default_output = args.output or config['detection']['csv_output']
        safe_output = safe_save_path(default_output)

        logger.info("Loading data from: %s", input_path)
        try:
            if input_path.endswith(".pcap"):
                logger.info("Detected PCAP input - preprocessing file before detection.")
                df = preprocess_file(input_path, batch_size, label=None)

            elif input_path.endswith(".csv"):
                logger.info("Detected CSV input - loading data directly from file.")
                df = pd.read_csv(input_path)

            else:
                logger.error("Unsupported input file format. Use '.pcap' or '.csv'.")
                return
            
            if df.empty:
                logger.warning("No usable packets to scan after loading.")
                return

            detection(model, df, model_type, model_name, model_path, safe_output)
        
        except Exception as e:
            logger.error("Detection input data is invalid: %s", e)