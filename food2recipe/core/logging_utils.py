# File: food2recipe/core/logging_utils.py
import logging
import sys
from pathlib import Path

def setup_logger(name: str = "food2recipe", level: int = logging.INFO) -> logging.Logger:
    """
    Configures a simple logger that outputs to stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
