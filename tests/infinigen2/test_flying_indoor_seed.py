import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "examples/flying_indoor/render.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("flying_indoor_render", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


def test_stereo_seed_accepts_decimal_and_hex() -> None:
    assert RENDER._parse_seed("303770610") == 303770610
    assert RENDER._parse_seed("0x121b2bf2") == 303770610
