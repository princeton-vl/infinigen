import logging

__version__ = "1.1.0"

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)