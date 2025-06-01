# siem/wazuh_forwarder.py
"""
Forward alerts to Wazuh via file log or syslog, depending on config settings.
"""
import json
import os
import socket
from datetime import datetime
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))

DEFAULT_MODE = config['siem']['mode']
DEFAULT_LOG_PATH = config['siem']['log_path']
DEFAULT_SYSLOG_ADDR = config['siem']['syslog_host']
DEFAULT_SYSLOG_PORT = config['siem']['syslog_port']

def forward_alert(alert: dict, dry_run: bool = False) -> bool:
    """
    Forward alert to Wazuh based on config: file, syslog, or both.

    Parameters:
    - alert: dictionary representing the alert payload
    - dry_run: if True, do not actually send, just log action

    Returns:
    - True if at least one delivery was successful, False otherwise
    """
    alert['timestamp'] = alert.get('timestamp') or datetime.utcnow().isoformat()
    alert_json = json.dumps(alert)

    success = False

    if DEFAULT_MODE in ('file', 'both'):
        try:
            logger.debug("Writing alert to file: %s", DEFAULT_LOG_PATH)
            if not dry_run:
                os.makedirs(os.path.dirname(DEFAULT_LOG_PATH), exist_ok=True)
                with open(DEFAULT_LOG_PATH, 'a') as f:
                    f.write(alert_json + '\n')
            success = True
        except Exception as e:
            logger.error("Failed to write alert to log file: %s", e)

    if DEFAULT_MODE in ('syslog', 'both'):
        try:
            logger.debug("Sending alert via syslog to %s:%d", DEFAULT_SYSLOG_ADDR, DEFAULT_SYSLOG_PORT)
            if not dry_run:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(alert_json.encode(), (DEFAULT_SYSLOG_ADDR, DEFAULT_SYSLOG_PORT))
                sock.close()
            success = True
        except Exception as e:
            logger.error("Failed to send alert via syslog: %s", e)

    return success
