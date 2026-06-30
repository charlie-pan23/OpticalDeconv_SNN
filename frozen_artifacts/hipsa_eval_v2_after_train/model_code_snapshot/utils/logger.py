import os
import logging
from datetime import datetime


def setup_logger(name=None):
    """
    Configure and return a unified globally formatted Logger.
    Provides console output (INFO+) and file output (DEBUG+) automatically.
    """
    # If a specific name is provided, use it to create a sub-logger, otherwise use root
    logger = logging.getLogger(name if name else "OpticalDeconv_SNN")

    # Prevent duplicate handlers if the logger is already initialized
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Unified Log Format: [Timestamp] - [Level] - [Filename:LineNo] - Message
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Console Handler (Outputs INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 2. File Handler (Automatically creates 'logs' directory at project root, saves DEBUG and above)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    log_dir = os.path.join(project_root, 'logs')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'run_{current_time}.log')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


# Expose a default global logger instance for convenience
logger = setup_logger()

if __name__ == "__main__":
    print("--- Starting Global Logger Test ---")
    test_logger = setup_logger("LoggerTest")

    test_logger.debug("This is a DEBUG log (Only saved to the file).")
    test_logger.info("This is an INFO log (Printed to console and saved to file).")
    test_logger.warning("This is a WARNING log.")
    test_logger.error("This is an ERROR log.")
    test_logger.critical("This is a CRITICAL log.")
    print("--- Logger Test Complete. Please check the 'logs/' directory at your project root. ---")