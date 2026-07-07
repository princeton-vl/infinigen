import logging

try:
    from .core import Terrain, hidden_in_viewport
except ImportError:
    Terrain = None
    hidden_in_viewport = []
    logging.warning(
        "Terrain failed import! {e=} terrain will likely crash if it is used"
    )
