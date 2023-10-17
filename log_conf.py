from datetime import datetime
import logging
import datetime
import os


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def init_logger(save_folder):
    # log_file = f"./logs/{file_name}"
    log_file = f"{save_folder}/debug.log"

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)
