# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import cv2
import gin
import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowDirectorSteepest, TransportLengthHillslopeDiffuser

from infinigen.terrain.utils import get_normal, read, smooth
from tqdm import tqdm
from infinigen.core.util.organization import AssetFile, Process
from infinigen.core.util.random import random_general as rg

snowfall_params_ = {}
@gin.configurable
def snowfall_params(
    normal_params=[
        ((np.cos(np.pi/6), 0, np.sin(np.pi/6)), (0.80, 0.801)),
        ((0, 0, 1), (0.90, 0.901)),
    ],
    detailed_normal_params=[((0, 0, 1), (0.80, 0.801))],
    on_rock_normal_params=[((0, 0, 1), (0.50, 0.501))],
):
    global snowfall_params_
    if snowfall_params_ == {}:
        snowfall_params_ = {
            "normal_params": rg(normal_params),
            "detailed_normal_params": rg(detailed_normal_params),
            "on_rock_normal_params": rg(on_rock_normal_params),
        }
    return snowfall_params_

@gin.configurable
def run_snowfall(
    folder,
    blending_params=[0, 0.5],
    diffussion_params=[(256, 10, 9), (1024, 10, 5)],
    verbose=0,
):
    heightmap_path = f"{folder}/{Process.Erosion}.{AssetFile.Heightmap}.exr"
    tile_size = float(np.loadtxt(f"{folder}/{AssetFile.TileSize}.txt"))
    rocks = read(heightmap_path)
    M = rocks.shape[0]

    snows = np.zeros_like(rocks) - 1e9
    for N, n_iters, smoothing_kernel in diffussion_params:
        snow = rocks.copy()
        snow = cv2.resize(snow, (N, N))
        mg = RasterModelGrid((N, N))
        mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
        _ = mg.add_field("topographic__elevation", snow, at="node")
        fdir = FlowDirectorSteepest(mg)
        tl_diff = TransportLengthHillslopeDiffuser(mg, erodibility=0.001, slope_crit=0.6)
        if verbose: range_t = tqdm(range(n_iters))
        else: range_t = range(n_iters)
        for t in range_t:
            fdir.run_one_step()
            tl_diff.run_one_step(1.)
        snow = mg.at_node['topographic__elevation']
        snow = snow.reshape((N, N))
        snow = cv2.resize(snow, (M, M))
        snow = smooth(snow, smoothing_kernel)
        snows = np.maximum(snows, snow)

    mask = np.zeros_like(rocks)
    for blending in blending_params:
        for normal_preference, (th0, th1) in snowfall_params()["normal_params"]:
            reference_snow = rocks * blending + snows * (1 - blending)
            normal_map = get_normal(reference_snow, tile_size / snows.shape[0])
            mask_sharpening = 1 / (th1 - th0)
            mask += np.clip(((normal_map * np.array(normal_preference).reshape((1, 1, 3))).sum(axis=-1) - th0) * mask_sharpening, a_min=0, a_max=1)
            mask -= np.clip(((-normal_map * np.array(normal_preference).reshape((1, 1, 3))).sum(axis=-1) - th0) * mask_sharpening, a_min=0, a_max=1)
            mask = np.clip(mask, a_min=0, a_max=1)
    heightmap = snows * mask + rocks * (1 - mask)
    cv2.imwrite(str(folder/f'{Process.Snowfall}.{AssetFile.Heightmap}.exr'), heightmap.astype(np.float32))
    cv2.imwrite(str(folder/f'{Process.Snowfall}.{AssetFile.Mask}.exr'), mask.astype(np.float32))