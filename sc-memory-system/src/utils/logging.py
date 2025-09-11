"""
Logging configuration for SC Memory System.

This module provides centralized logging configuration with proper
formatters, handlers, and log levels for different environments.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..core.config import Settings, get_settings


def setup_logging(
    settings: Optional[Settings] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Setup application logging configuration.
    
    Args:
        settings: Application settings
        log_file: Optional log file path
    """
    if settings is None:
        settings = get_settings()
    
    # Configure log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Create formatters
    if settings.debug:
        format_str = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_str)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={settings.log_level}")


__all__ = ["setup_logging"]