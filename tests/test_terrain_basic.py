# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

from pathlib import Path

import bpy
import gin
import pytest

from infinigen.core.surface import registry
from infinigen.core.util.organization import Task
from infinigen.terrain import Terrain
from infinigen_examples.util.test_utils import setup_gin


@pytest.mark.skip_for_ci
@pytest.mark.nature
def test_terrain_runs():
    setup_gin(
        "infinigen_examples/configs_nature",
        configs=["fast_terrain_assets"],
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

    bpy.ops.preferences.addon_enable(module="add_mesh_extra_objects")
    bpy.ops.preferences.addon_enable(module="ant_landscape")

    terrain = Terrain(0, registry, task=Task.Coarse, on_the_fly_asset_folder="/tmp/terrain_tests")
    terrain.coarse_terrain()

    gin.clear_config()
    gin.unlock_config()
