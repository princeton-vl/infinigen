import pytest

from infinigen.core.sim import sim_factory as sf

ASSETS = ["door", "lamp", "dishwasher", "multifridge", "multidoublefridge", "toaster"]
FILE_FORMATS = ["mjcf", "urdf", "usd"]


@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("format", FILE_FORMATS)
def test_sim_export(asset, format):
    export_path, semantic_mapping = sf.spawn_simready(
        name=asset,
        seed=100,
        exporter=format,
        visual_only=True,
    )
