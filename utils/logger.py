import os
import logging


def get_logger(filename=None, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    if filename is not None:
        fh = logging.FileHandler(filename, "w")
        fh.setLevel(level_dict[verbosity])
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(level_dict[verbosity])
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
