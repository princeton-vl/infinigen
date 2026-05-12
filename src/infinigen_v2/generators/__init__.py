from pathlib import Path

from pandas import read_json as _read_json

GENERATORS_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
GENERATORS_MANIFEST = _read_json(GENERATORS_MANIFEST_PATH)
