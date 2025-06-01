# utils/logger.py

import logging
import sys

def get_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
    """
    Returns a module-specific logger with consistent formatting and stream output.

    Parameters:
    - name: logger name, typically use __name__
    - level: string log level (e.g., 'DEBUG', 'INFO')

    Returns:
    - Configured Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
    return logger
