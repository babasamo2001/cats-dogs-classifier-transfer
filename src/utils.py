import logging
import os


def setup_logger(
        logger_name,
        log_file
):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    logger = logging.getLogger(
        logger_name
    )

    logger.setLevel(
        logging.INFO
    )

    handler = logging.FileHandler(
        log_file
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(
        formatter
    )

    logger.addHandler(
        handler
    )

    return logger