import os
import sys
from loguru import logger

def setup_logger(log_file: str = "logs/app.log", level: str = "INFO") -> None:
    """
    Configures Loguru logger to output to both console and a rotating log file.
    
    Args:
        log_file (str): Path to the log file.
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Remove all default handlers (specifically stderr console handler)
    logger.remove()

    # Add console output handler with nice colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        enqueue=True
    )

    # Add rotating file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:7} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        enqueue=True
    )
    
    logger.info(f"Logger initialized with level: {level}, saving to: {log_file}")
