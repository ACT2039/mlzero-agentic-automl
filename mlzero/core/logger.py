"""
Structured logging module for MLZero.
"""

import logging
from pathlib import Path

from mlzero.core.config import settings


def setup_logger(name: str) -> logging.Logger:
    """
    Setup a structured logger.

    Args:
        name (str): The name of the logger (typically __name__).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Map string level to logging level
    level_name = settings.logging.level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    
    formatter = logging.Formatter(settings.logging.format)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if settings.logging.file:
        log_file_path = Path(settings.logging.file)
        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
