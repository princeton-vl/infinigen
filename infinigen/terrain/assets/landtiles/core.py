# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import json

import cv2
import gin
import numpy as np
from infinigen.terrain.utils import boundary_smooth, read, smooth
from infinigen.core.util.organization import AssetFile, LandTile, Process

from .ant_landscape import ant_landscape_asset
from .custom import coast_asset, multi_mountains_asset, coast_params, multi_mountains_params


@gin.configurable
def tile_sizes(
    MultiMountains=600,
    Coast=600,
    Mesa=50,
    Canyon=200,
    Canyons=200,
    Cliff=50,
    Mountain=50,
    River=50,
    Volcano=50,
):
    """_summary_

    This function specifies tile size of each land tile asset type
    """
    return {
        LandTile.MultiMountains: MultiMountains,
        LandTile.Coast: Coast,
        LandTile.Mesa: Mesa,
        LandTile.Canyon: Canyon,
        LandTile.Canyons: Canyons,
        LandTile.Cliff: Cliff,
        LandTile.Mountain: Mountain,
        LandTile.River: River,
        LandTile.Volcano: Volcano,
    }


@gin.configurable
def tile_directions(
    MultiMountains="random",
    Coast="dependent",
    Mesa="random",
    Canyon="random",
    Canyons="random",
    Cliff="dependent",
    Mountain="random",
    River="initial",
    Volcano="random",
):
    """_summary_

    This function specifies direction mode of each land tile asset type,
    "random" means it will be put in random direction in the scene;
    "dependent" means it will be put from low to high in the x direction;
    "initial" means it will not be rotated;

    """
    return {
        LandTile.MultiMountains: MultiMountains,
        LandTile.Coast: Coast,
        LandTile.Mesa: Mesa,
        LandTile.Canyon: Canyon,
        LandTile.Canyons: Canyons,
        LandTile.Cliff: Cliff,
        LandTile.Mountain: Mountain,
        LandTile.River: River,
        LandTile.Volcano: Volcano,
    }


def assets_to_data(
    folder, land_process,
    N=2048,
    do_smooth=False,
):
    preset_name = str(folder).split("/")[-2]
    data = {}
    if land_process is None: path = folder/f"{AssetFile.Heightmap}.exr"
    elif land_process == Process.Snowfall: path = folder/f"{Process.Snowfall}.{AssetFile.Heightmap}.exr"
    elif land_process == Process.Erosion: path = folder/f"{Process.Erosion}.{AssetFile.Heightmap}.exr"
    heightmap = read(path)
    assert(heightmap.shape[0] == N)
    if do_smooth: heightmap = smooth(heightmap, 3)

    if land_process is None:
        mask = np.zeros(N * N)
    else:
        if land_process == Process.Snowfall: path = folder/f"{Process.Snowfall}.{AssetFile.Mask}.exr"
        elif land_process == Process.Erosion: path = folder/f"{Process.Erosion}.{AssetFile.Mask}.exr"
        mask = read(path)

    mask = mask.reshape(-1)
    data["mask"] = mask
    
    # compute direction of directional tiles (must be done before smoothing it)
    direction = tile_directions()[preset_name]
    if direction == "dependent":
        data["direction"] = np.arctan2(
            np.mean(heightmap[:, -1] - heightmap[:, 0]),
            np.mean(heightmap[-1] - heightmap[0])
        ).reshape(-1)
    elif direction == "initial":
        data["direction"] = np.array([0.0])
    
    if direction != "dependent": heightmap = boundary_smooth(heightmap)
    data["heightmap"] = heightmap.reshape(-1)
    L = float(np.loadtxt(folder/f"{AssetFile.TileSize}.txt"))
    if preset_name == LandTile.MultiMountains and (folder/f"{AssetFile.Params}.txt").exists():
        with open(folder/f"{AssetFile.Params}.txt", "r") as file:
            params = json.load(file)
            assert params == multi_mountains_params(raw=1), "asset should not be reused if you changed settings"
    if preset_name == LandTile.Coast and (folder/f"{AssetFile.Params}.txt").exists():
        with open(folder/f"{AssetFile.Params}.txt", "r") as file:
            params = json.load(file)
            assert params == {"multi_mountains_params": multi_mountains_params(raw=1), "coast_params": coast_params(raw=1)}, "asset should not be reused if you changed settings"
    
    return L, N, data


def landtile_asset(
    folder,
    preset_name,
    resolution=2048,
    device=None,
):
    tile_size = tile_sizes()[preset_name]
    if preset_name == LandTile.MultiMountains:
        multi_mountains_asset(folder, tile_size, resolution, device)
    elif preset_name == LandTile.Coast:
        coast_asset(folder, tile_size, resolution, device)
    else:
        ant_landscape_asset(folder, preset_name, tile_size, resolution)
    (folder / AssetFile.Finish).touch()
