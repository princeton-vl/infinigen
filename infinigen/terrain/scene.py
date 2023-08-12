# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
from infinigen.terrain.elements.caves import Caves
from infinigen.terrain.elements.ground import Ground
from infinigen.terrain.elements.landtiles import LandTiles, Volcanos, FloatingIce
from infinigen.terrain.elements.upsidedown_mountains import UpsidedownMountains
from infinigen.terrain.elements.voronoi_rocks import VoronoiRocks, VoronoiGrains
from infinigen.terrain.elements.warped_rocks import WarpedRocks
from infinigen.terrain.elements.waterbody import Waterbody
from infinigen.terrain.elements.atmosphere import Atmosphere

from infinigen.terrain.utils import chance
from infinigen.core.util.organization import ElementNames, Assets
from infinigen.core.util.math import FixedSeed, int_hash

@gin.configurable
def scene(
    seed,
    on_the_fly_asset_folder,
    reused_asset_folder,
    device,
    caves_chance=0.5,
    landtiles_chance=1,
    ground_chance=1,
    warped_rocks_chance=0.3,
    voronoi_rocks_chance=0.5,
    voronoi_grains_chance=0,
    upsidedown_mountains_chance=0,
    waterbody_chance=0.5,
    volcanos_chance=0,
    ground_ice_chance=0,
):
    elements = {}
    scene_infos = {}

    with FixedSeed(int_hash([seed, "caves"])):
        if chance(caves_chance):
            caves = Caves(on_the_fly_asset_folder / Assets.Caves, reused_asset_folder / Assets.Caves)
        else:
            caves = None

    last_ground_element = None
    
    with FixedSeed(int_hash([seed, "ground"])):
        if chance(ground_chance):
            elements[ElementNames.Ground] = Ground(device, caves)
            last_ground_element = elements[ElementNames.Ground]
    
    with FixedSeed(int_hash([seed, "landtiles"])):
        if chance(landtiles_chance):
            elements[ElementNames.LandTiles] = LandTiles(device, caves, on_the_fly_asset_folder, reused_asset_folder)
            last_ground_element = elements[ElementNames.LandTiles]

    assert(last_ground_element is not None)

    with FixedSeed(int_hash([seed, "warped_rocks"])):
        if chance(warped_rocks_chance):
            elements[ElementNames.WarpedRocks] = WarpedRocks(device, caves)

    with FixedSeed(int_hash([seed, "voronoi_rocks"])):
        if chance(voronoi_rocks_chance):
            elements[ElementNames.VoronoiRocks] = VoronoiRocks(device, last_ground_element, caves)

    with FixedSeed(int_hash([seed, "voronoi_grains"])):
        if chance(voronoi_grains_chance):
            elements[ElementNames.VoronoiGrains] = VoronoiGrains(device, last_ground_element, caves)
    
    with FixedSeed(int_hash([seed, "upsidedown_mountains"])):
        if chance(upsidedown_mountains_chance):
            elements[ElementNames.UpsidedownMountains] = UpsidedownMountains(
                device, on_the_fly_asset_folder / Assets.UpsidedownMountains, reused_asset_folder / Assets.UpsidedownMountains
            )
    
    with FixedSeed(int_hash([seed, "volcanos"])):
        if chance(volcanos_chance):
            elements[ElementNames.Volcanos] = Volcanos(
                device, None, on_the_fly_asset_folder, reused_asset_folder
            )
    
    with FixedSeed(int_hash([seed, "ground_ice"])):
        if chance(ground_ice_chance):
            elements[ElementNames.FloatingIce] = FloatingIce(
                device, None, on_the_fly_asset_folder, reused_asset_folder
            )

    scene_infos["water_plane"] = -1e5
    waterbody = None
    
    with FixedSeed(int_hash([seed, "waterbody"])):
        if chance(waterbody_chance):
            waterbody = Waterbody(device, elements.get(ElementNames.LandTiles, None))
            elements[ElementNames.Liquid] = waterbody
            scene_infos["water_plane"] = waterbody.height

    elements[ElementNames.Atmosphere] = Atmosphere(device, waterbody=waterbody)

    return elements, scene_infos

def transfer_scene_info(terrain, scene_info):
    for key in scene_info:
        setattr(terrain, key, scene_info[key])