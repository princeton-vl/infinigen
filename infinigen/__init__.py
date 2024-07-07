import logging
from pathlib import Path

__version__ = "1.5.0"


def repo_root():
    return Path(__file__).parent.parent
