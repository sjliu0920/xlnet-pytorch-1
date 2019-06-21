import logging


DEFAULT_OPTIONS = ("headless", "disable-gpu")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt=None,
    style="%"
)

ch.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.propagate = False
logger.addHandler(ch)
