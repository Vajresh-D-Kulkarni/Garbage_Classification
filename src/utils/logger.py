import logging
import os


def get_logger(log_dir="outputs/logs", log_file="train.log"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("GarbageClassifier")
    logger.setLevel(logging.INFO)

    log_path = os.path.join(log_dir, log_file)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
