"""
Logging Configuration
Supports DEBUG, INFO, WARNING, ERROR modes
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

from config.settings import LOGS_DIR, LOG_LEVEL

def setup_logging(name: str = "project_sam", level: str = None) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Override log level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger
    """
    level = level or LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Default logger instance
log = setup_logging()
