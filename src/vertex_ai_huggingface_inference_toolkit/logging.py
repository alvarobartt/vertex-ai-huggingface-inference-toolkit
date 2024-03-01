import logging
import os
import sys
import warnings

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Gets the `logging.Logger` for the current package with a custom
    configuration. Also uses `rich` for better formatting.
    """

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
        stream=sys.stdout,
    )
    # Remove `datasets` logger to only log on `critical` mode
    # as it produces `PyTorch` messages to update on `info`
    logging.getLogger("datasets").setLevel(logging.CRITICAL)

    log_level = os.environ.get("INFERENCE_LOG_LEVEL", "INFO")
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        warnings.warn(
            f"Invalid log level '{log_level}', using default 'INFO' instead.",
            stacklevel=2,
        )
        log_level = "INFO"

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger
