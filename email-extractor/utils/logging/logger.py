import logging
import sys
from pathlib import Path

def setup_logger(name=None, level=None, log_file=None):
    """
    Setup standardized logger for the application
    
    Args:
        name: Logger name (default: root logger)
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level or logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level or logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level or logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=None):
    """Get or create logger with standard configuration"""
    from utils.config import get_config
    
    config = get_config()
    
    log_level = getattr(logging, config.get('logging.level', 'INFO'))
    log_file = config.get('logging.log_file') if config.get('logging.log_to_file') else None
    
    return setup_logger(name, log_level, log_file)
