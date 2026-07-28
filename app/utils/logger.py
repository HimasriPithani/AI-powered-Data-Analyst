"""
Simple structured logging setup used across the app for observability.
Logs both to console and to a rotating file so tool calls, errors, and
LLM interactions can be audited after the fact.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from app.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on Streamlit re-runs

    logger.setLevel(settings.log_level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    try:
        log_dir = os.path.dirname(settings.log_file) or "."
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.log_file, maxBytes=2_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        # File logging is best-effort (e.g. read-only filesystem in some envs)
        pass

    return logger
