import logging



class SimpleLogger:
    def __init__(self, module_name=""):
        self._logger = logging.getLogger(module_name)


    def config(self, lvl=logging.DEBUG):
        self._logger.setLevel(lvl)

    def add_stream_handler(self, ch=None ):
        if ch is None:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s')
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)
        else:
            self._logger.addHandler(ch)
            

    def debug (self, msg):
        self._logger.debug(msg)

    def info (self, msg):
        self._logger.info(msg)

    def warning (self, msg):
        self._logger.warning(msg)

    def error (self, msg):
        self._logger.error(msg)

    def critical (self, msg):
        self._logger.critical(msg)


if __name__ == "__main__":
    logger = SimpleLogger("Test")
    logger.add_stream_handler()

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')