import logging
from pathlib import Path

# Define the base directory for logs
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Define log message format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
exampleFormatter = logging.Formatter(LOG_FORMAT)


def logger_config(file_name: str) -> logging.Logger:
    """
    Configures and returns a logger with file and stream handlers.

    Parameters
    ----------
    file_name : str
        The base name of the log file (without extension).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(file_name)  # Use file name as logger name
    logger.propagate = False

    # Create file handler for logging
    file_handler = logging.FileHandler(LOG_DIR / f"{file_name}.log", mode="w")
    file_handler.setFormatter(exampleFormatter)

    # Create stream handler for logging
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(exampleFormatter)

    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    return logger
