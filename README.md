# Anomaly Detect

A modular, machine learning-based threat detection pipeline for identifying anomalies in network traffic. 

Designed for use by security analysts, researchers, and engineers in need of a production-grade framework for real-time or file-based detection of network threats.

---

## Features

- **Anomaly-based detection** using Autoencoder, Random Forest, and Support Vector Machine models
- **Live or offline packet capture** using Scapy
- **Custom preprocessing** with detailed packet-level feature extraction
- **Modular training pipeline** with support for pluggable models
- **Visual evaluation tools** (ROC, confusion matrix, reconstruction loss)
- **Wazuh SIEM integration** for alert forwarding via file or syslog
- Fully customisable via YAML config
- Unified logging and versioning support
- Extensive unit test suite and CLI interface

---

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate anomaly-detect
```

### `pip` (Alternative)
```bash
pip install .
```

---

## Usage

```bash
anomaly-detect --help
```

### Example Commands
```bash
# Capture 30 seconds of live traffic on eth0 network adapter
python -m cli.main capture --interface eth0 --duration 30 --output data/captures/sample.pcap

# Extract features from PCAP to CSV
python -m cli.main preprocess --pcap data/captures/sample.pcap --output data/preprocessed/sample.csv

# Train a model
python -m cli.main train --model autoencoder --input data/preprocessed/sample.csv --output models/my_autoencoder

# Detect anomalies using a trained model
python -m cli.main detect --model models/my_autoencoder --model-type autoencoder --input data/preprocessed/sample.csv --output data/detection/predicted.csv

```

---

## Project Structure
```
anomaly-detect/
├── cli/                  # CLI entry point
├── config/               # YAML configuration file
├── core/                 # Capture, preprocessing, and dataset tools
├── models/               # ML model logic, base interface, loader, trainer, detector
│   └── pretrained/       # Pretrained demonstration models
├── siem/                 # Wazuh alert forwarding (file/syslog integration)
├── utils/                # Logging, config loading, file saving, etc.
├── tests/                # Full pipeline testing coverage using unit test framework
├── data/                 # Captures, processed feature CSVs, public datasets, models, etc.
```

---

### Testing
```
python -m unittest discover tests
```

---

### SIEM Integration

Alerts are exported in Wazuh-compatible format for ingestion into your SIEM. Logs can be forwarded to:
- Local file (default: `data/logs/ml_alerts.log`)
- Syslog over UDP
- Both (configurable)

---

## License

MIT License - see more details [here](./LICENSE)

---

## Author

Developed by William Jecks as a part of a final year thesis project.

Supervised by Prof. Fatemeh Raji (2024 to March 2025), Jarrad Morden (March 2025 to end).

Cyber Security, De Montfort University  
p2721974@my365.dmu.ac.uk

---
