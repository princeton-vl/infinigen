import logging
from pathlib import Path

__version__ = "1.6.0-dev"


def repo_root():
    return Path(__file__).parent.parent
