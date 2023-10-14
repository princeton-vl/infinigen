from pathlib import Path

import pytest
import bpy
import gin
from infinigen.terrain import Terrain
from infinigen.core.surface import registry
from infinigen.core.util.organization import Task

from utils import (
    setup_gin, 
)

def test_terrain_runs():

    setup_gin(
        configs=['fast_terrain_assets'],
        overrides=[
            'scene.caves_chance=1',
            'scene.landtiles_chance=1',
            'scene.ground_chance=1',
            'scene.warped_rocks_chance=1',
            'scene.voronoi_rocks_chance=1',
            'scene.voronoi_grains_chance=0',
            'scene.upsidedown_mountains_chance=1',
            'scene.waterbody_chance=1',
            'scene.volcanos_chance=0',
            'scene.ground_ice_chance=0',
        ],
    )

    bpy.ops.preferences.addon_enable(module='add_mesh_extra_objects')
    bpy.ops.preferences.addon_enable(module='ant_landscape')

    terrain = Terrain(0, registry, task=Task.Coarse, on_the_fly_asset_folder="/tmp/terrain_tests")
    terrain.coarse_terrain()

    gin.clear_config()