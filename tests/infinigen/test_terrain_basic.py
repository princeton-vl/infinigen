# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import pathlib

import gin
import pytest

import infinigen.terrain
from infinigen.core.init import configure_blender
from infinigen.core.util.organization import Task
from infinigen.core.util.test_utils import setup_gin
from infinigen.terrain import Terrain

_ELEMENTS = pathlib.Path(infinigen.terrain.__file__).parent / "lib" / "cpu" / "elements"


@pytest.mark.skip_for_ci
@pytest.mark.skipif(
    not _ELEMENTS.is_dir() or not any(_ELEMENTS.glob("*.so")),
    reason="needs a terrain build (install with INFINIGEN_INSTALL_TERRAIN=True)",
)
def test_terrain_runs(tmp_path):
    setup_gin(
        "infinigen_examples/configs_nature",
        configs=["base_nature.gin", "fast_terrain_assets"],
        overrides=[
            "scene.caves_chance=1",
            "scene.landtiles_chance=1",
            "scene.ground_chance=1",
            "scene.warped_rocks_chance=1",
            "scene.voronoi_rocks_chance=1",
            "scene.voronoi_grains_chance=0",
            "scene.upsidedown_mountains_chance=1",
            "scene.waterbody_chance=1",
            "scene.volcanos_chance=0",
            "scene.ground_ice_chance=0",
        ],
    )

    configure_blender()

    terrain = Terrain(
        0,
        task=Task.Coarse,
        # a fixed /tmp path may already exist and belong to another user
        on_the_fly_asset_folder=str(tmp_path / "terrain_tests"),
    )
    terrain.coarse_terrain()

    gin.clear_config()
    gin.unlock_config()
