import pkg_resources
from pathlib import Path
__version__ = pkg_resources.get_distribution(Path(__file__).parent.name).version