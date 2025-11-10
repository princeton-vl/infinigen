import logging
import sys
from pathlib import Path

import gin
import pytest

from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Set up command line options for pytest."""
    parser.addoption(
        "--asset",
        nargs="*",
        default=["all"],
        help="Specify one or more assets to test. Defaults to 'all'.",
    )

    parser.addoption(
        "--nr",
        help="Number of objects to test.",
    )


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    gin.clear_config()


@pytest.fixture(scope="session")
def assets_to_test(request):
    """
    Get command line options for assets to test.
    """
    return request.config.getoption("--asset")


@pytest.fixture(scope="session")
def nr_assets(request):
    return request.config.getoption("--nr")


@pytest.fixture(scope="session")
def cached_assets(request):
    """
    Spawn all assets once per test session and cache them.
    Returns a dict: {asset_name: [(seed, obj, mesh), ...]}
    """
    nr_assets = int(request.config.getoption("--nr"))
    assets_to_spawn = request.config.getoption("--asset")
    if "all" in assets_to_spawn:
        assets_to_spawn = list(OBJECT_CLASS_MAP.keys())

    cache = {}
    for asset_name in assets_to_spawn:
        if asset_name not in OBJECT_CLASS_MAP:
            pytest.fail(f"Asset '{asset_name}' not found in OBJECT_CLASS_MAP.")

        logger.info(f"Spawning asset: {asset_name}")

    return cache
