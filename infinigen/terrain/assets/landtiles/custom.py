# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import json
from pathlib import Path

import cv2
import gin
import numpy as np
from infinigen.terrain.elements.core import Element
from infinigen.terrain.elements.mountains import Mountains
from infinigen.terrain.land_process.erosion import run_erosion
from infinigen.terrain.land_process.snowfall import run_snowfall
from infinigen.terrain.utils import grid_distance, perlin_noise, random_int
from infinigen.core.util.organization import AssetFile
from infinigen.core.util.random import random_general as rg


coast_params_ = {}
multi_mountains_params_ = {}

@gin.configurable
def coast_params(
    coast_freq=("uniform", 0.00005, 0.00015),
    beach_size=("uniform", 50, 100),
    beach_slope=("uniform", 0.07, 0.12),
    steep_slope_size=("uniform", 5, 10),
    sea_depth=("uniform", 8, 12),
    raw=False,
):
    """_summary_

        coast_freq: base frequency of coast line
        beach_size: size of beach
        beach_slope: slope of beach
        steep_slope_size: size of the steep part between beach and sea floor
        sea_depth: sea depth
    """
    if raw:
        d = {
            "coast_freq": coast_freq,
            "beach_size": beach_size,
            "beach_slope": beach_slope,
            "steep_slope_size": steep_slope_size,
            "sea_depth": sea_depth,
        }
        return {x: list(d[x]) if type(d[x]) is tuple else d[x] for x in d}
    global coast_params_
    if coast_params_ == {}:
        coast_params_ = {
            "coast_freq": rg(coast_freq),
            "beach_size": rg(beach_size),
            "beach_slope": rg(beach_slope),
            "steep_slope_size": rg(steep_slope_size),
            "sea_depth": rg(sea_depth),
        }
    return coast_params_
    

@gin.configurable
def multi_mountains_params(
    min_freq=("uniform", 0.0005, 0.002),
    max_freq=("uniform", 0.003, 0.004),
    height=("uniform", 60, 90),
    coverage=("uniform", 0.4, 0.6),
    slope_freq=("uniform", 0.001, 0.004),
    slope_height=("uniform", 4, 6),
    raw=False,
):
    """_summary_

        min_freq: min base frequency of all mountains
        max_freq: max base frequency of all mountains
        height: mountain height
        coverage: mountain coverage
        slope_freq: base frequency of the slope the mountains sit on
        slope_height: height of such slope
    """
    if raw:
        d = {
            "min_freq": min_freq,
            "max_freq": max_freq,
            "height": height,
            "coverage": coverage,
            "slope_freq": slope_freq,
            "slope_height": slope_height,
        }
        return {x: list(d[x]) if type(d[x]) is tuple else d[x] for x in d}
    global multi_mountains_params_
    if multi_mountains_params_ == {}:
        multi_mountains_params_ = {
            "min_freq": rg(min_freq),
            "max_freq": rg(max_freq),
            "height": rg(height),
            "coverage": rg(coverage),
            "slope_freq": rg(slope_freq),
            "slope_height": rg(slope_height),
        }
    return multi_mountains_params_


def coast_heightmapping(heightmap):
    mapped = heightmap.copy()
    params = coast_params()
    beach_size = params["beach_size"]
    beach_slope = params["beach_slope"]
    steep_slope_size = params["steep_slope_size"]
    sea_depth = params["sea_depth"]
    seafloor_loc = beach_size / 2 + steep_slope_size
    mapped[(heightmap > -beach_size/2) & (heightmap < beach_size/2)] *= beach_slope
    mapped[heightmap > beach_size/2] = beach_size/2 * beach_slope
    steep_slope = (sea_depth - beach_size/2 * beach_slope) / (seafloor_loc - beach_size/2)
    steep_mask = (heightmap < -beach_size/2) & (heightmap > -seafloor_loc)
    mapped[steep_mask] = (-beach_size/2 * beach_slope - (-beach_size/2 - heightmap) * steep_slope)[steep_mask]
    mapped[heightmap < -seafloor_loc] = -sea_depth
    return mapped


def multi_mountains_asset(
    folder,
    tile_size,
    resolution,
    device,
    erosion=True,
    snowfall=True,
):
    Path(folder).mkdir(parents=True, exist_ok=True)
    x = np.linspace(-tile_size / 2, tile_size / 2, resolution)
    y = np.linspace(-tile_size / 2, tile_size / 2, resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")
    params = multi_mountains_params()
    mountains = Mountains(
        device=device,
        min_freq=params["min_freq"],
        max_freq=params["max_freq"],
        height=params["height"],
        coverage=params["coverage"],
        slope_freq=params["slope_freq"],
        slope_height=params["slope_height"],
    )
    heightmap = mountains.get_heightmap(X, Y)
    mountains.cleanup()
    Element.called_time.pop("mountains")
    cv2.imwrite(str(folder / f'{AssetFile.Heightmap}.exr'), heightmap)
    with open(folder/f'{AssetFile.TileSize}.txt', "w") as f:
        f.write(f"{tile_size}\n")
    with open(folder/f'{AssetFile.Params}.txt', "w") as f:
        json.dump(multi_mountains_params(raw=1), f)
    if erosion: run_erosion(folder)
    if snowfall: run_snowfall(folder)
    

def coast_asset(
    folder,
    tile_size,
    resolution,
    device,
    erosion=True,
    snowfall=True,
):
    Path(folder).mkdir(parents=True, exist_ok=True)
    x = np.linspace(-tile_size / 2, tile_size / 2, resolution)
    y = np.linspace(-tile_size / 2, tile_size / 2, resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")
    params1 = multi_mountains_params()
    mountains = Mountains(
        device=device,
        min_freq=params1["min_freq"],
        max_freq=params1["max_freq"],
        height=params1["height"],
        coverage=params1["coverage"],
        slope_freq=params1["slope_freq"],
        slope_height=params1["slope_height"],
    )
    heightmap = mountains.get_heightmap(X, Y)
    mountains.cleanup()
    Element.called_time.pop("mountains")

    params2 = coast_params()
    positions = np.stack((X.reshape(-1), Y.reshape(-1), np.zeros(resolution * resolution)), -1)
    coast_mask = perlin_noise(
        device=device,
        positions=positions,
        seed=random_int(),
        freq=params2["coast_freq"],
        octaves=9,
    ).reshape((resolution, resolution)) > 0
    coast_distance = (grid_distance(~coast_mask, downsample=512) - grid_distance(coast_mask, downsample=512)) * tile_size
    mask = np.clip((coast_distance - 0.2 * params2["beach_size"]) / (0.4 * params2["beach_size"]), a_min=0, a_max=1)
    coast_heightmap = coast_heightmapping(coast_distance)
    heightmap = (coast_heightmap + heightmap * mask).astype(np.float32)
    cv2.imwrite(str(folder / f'{AssetFile.Heightmap}.exr'), heightmap)
    with open(folder/f'{AssetFile.TileSize}.txt', "w") as f:
        f.write(f"{tile_size}\n")
    with open(folder/f'{AssetFile.Params}.txt', "w") as f:
        json.dump({"multi_mountains_params": multi_mountains_params(raw=1), "coast_params": coast_params(raw=1)}, f)
    if erosion: run_erosion(folder, mask_height_range=(0, 0.1 * params2["beach_size"] * params2["beach_slope"]))
    if snowfall: run_snowfall(folder)
