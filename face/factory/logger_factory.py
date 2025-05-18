import logging


def get_logger(module_name="", lvl=logging.DEBUG):
    logger = logging.getLogger(module_name)
    logger.setLevel(lvl)
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    logger = get_logger("Test")

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')