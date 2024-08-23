# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import pytest

from infinigen.core.init import configure_blender
from infinigen.core.surface import registry
from infinigen.core.util.organization import Task
from infinigen.core.util.test_utils import setup_gin
from infinigen.terrain import Terrain


@pytest.mark.skip
@pytest.mark.nature
def test_terrain_runs():
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
        0, registry, task=Task.Coarse, on_the_fly_asset_folder="/tmp/terrain_tests"
    )
    terrain.coarse_terrain()

    gin.clear_config()
    gin.unlock_config()
