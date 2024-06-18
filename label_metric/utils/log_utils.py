import logging


def setup_logger(logger, log_fn=None):
    # Set the global log level for the logger
    logger.setLevel(logging.DEBUG)

    # Create a console handler with INFO level or above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                       datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)

    # Add console handler to the logger
    logger.addHandler(console_handler)

    # If log_fn is provided, add a file handler
    if log_fn:
        file_handler = logging.FileHandler(log_fn)
        file_handler.setLevel(logging.DEBUG)  # File handler logs all levels
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
