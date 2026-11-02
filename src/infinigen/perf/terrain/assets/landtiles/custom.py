# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen custom land tiles (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/assets/landtiles/custom.py)
# - David Yan, Jia Deng: optimized implementation

import json
from pathlib import Path

import cv2
import numpy as np

from infinigen.core.util.organization import AssetFile
from infinigen.perf.terrain.land_process.erosion import plan_erosion
from infinigen.perf.terrain.land_process.planning import run_land_process_steps
from infinigen.perf.terrain.land_process.snowfall import plan_snowfall
from infinigen.terrain.assets.landtiles.custom import (
    coast_heightmapping,
    coast_params,
    multi_mountains_params,
)
from infinigen.terrain.elements.core import Element
from infinigen.terrain.elements.mountains import Mountains
from infinigen.terrain.utils import grid_distance, perlin_noise, random_int


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
    cv2.imwrite(str(folder / f"{AssetFile.Heightmap}.exr"), heightmap)

    with open(folder / f"{AssetFile.TileSize}.txt", "w") as f:
        f.write(f"{tile_size}\n")
    with open(folder / f"{AssetFile.Params}.txt", "w") as f:
        json.dump(multi_mountains_params(raw=1), f)

    steps = []
    if erosion:
        steps.append(plan_erosion(folder))
    if snowfall:
        steps.append(plan_snowfall(folder))
    run_land_process_steps(folder, steps)


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
    positions = np.stack(
        (X.reshape(-1), Y.reshape(-1), np.zeros(resolution * resolution)), -1
    )
    coast_mask = (
        perlin_noise(
            device=device,
            positions=positions,
            seed=random_int(),
            freq=params2["coast_freq"],
            octaves=9,
        ).reshape((resolution, resolution))
        > 0
    )
    coast_distance = (
        grid_distance(~coast_mask, downsample=512)
        - grid_distance(coast_mask, downsample=512)
    ) * tile_size
    mask = np.clip(
        (coast_distance - 0.2 * params2["beach_size"]) / (0.4 * params2["beach_size"]),
        a_min=0,
        a_max=1,
    )
    coast_heightmap = coast_heightmapping(coast_distance)
    heightmap = (coast_heightmap + heightmap * mask).astype(np.float32)
    cv2.imwrite(str(folder / f"{AssetFile.Heightmap}.exr"), heightmap)
    with open(folder / f"{AssetFile.TileSize}.txt", "w") as f:
        f.write(f"{tile_size}\n")
    with open(folder / f"{AssetFile.Params}.txt", "w") as f:
        json.dump(
            {
                "multi_mountains_params": multi_mountains_params(raw=1),
                "coast_params": coast_params(raw=1),
            },
            f,
        )
    steps = []
    if erosion:
        steps.append(
            plan_erosion(
                folder,
                mask_height_range=(
                    0,
                    0.1 * params2["beach_size"] * params2["beach_slope"],
                ),
            )
        )
    if snowfall:
        steps.append(plan_snowfall(folder))
    run_land_process_steps(folder, steps)
