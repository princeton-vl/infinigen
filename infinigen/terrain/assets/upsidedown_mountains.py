# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from pathlib import Path

import cv2
import gin
import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowDirectorSteepest, TransportLengthHillslopeDiffuser
from numpy import ascontiguousarray as AC
from skimage.measure import label
from infinigen.terrain.elements.core import Element
from infinigen.terrain.elements.mountains import Mountains
from infinigen.terrain.utils import read
from tqdm import tqdm
from infinigen.core.util.organization import AssetFile
from infinigen.core.util.random import random_general as rg


@gin.configurable
def upsidedown_mountains_asset(
    folder,
    device,
    min_freq=("uniform", 0.005, 0.015),
    max_freq=("uniform", 0.025, 0.035),
    height=("uniform", 20, 30),
    coverage=0.5,
    tile_size=150,
    resolution=256,
    verbose=0,
):
    """_summary_
        min_freq: min base frequency of all upsidedown mountains
        max_freq: max base frequency of all upsidedown mountains
        height: upsidedown mountain height
        coverage: upsidedown mountain coverage
        tile_size: size of the upsidedown mountain tile

    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    N = resolution
    x = np.linspace(-tile_size / 2, tile_size / 2, N)
    y = np.linspace(-tile_size / 2, tile_size / 2, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    min_freq = rg(min_freq)
    max_freq = rg(max_freq)
    height = rg(height)
    coverage = rg(coverage)
    mountains1 = Mountains(
        device=device,
        slope_height=0,
        min_freq=min_freq,
        max_freq=max_freq,
        height=height,
        coverage=coverage,
    )
    mountains2 = Mountains(
        device=device,
        slope_height=0,
        min_freq=min_freq,
        max_freq=max_freq,
        height=height,
        coverage=coverage,
    )
    heightmap = mountains1.get_heightmap(X, Y)
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
    radius = (x ** 2 + y ** 2) ** 0.5
    heightmap *= 1 - np.clip((radius - 0.8) * 5, a_min=0, a_max=1)
    mg = RasterModelGrid((N, N))
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
    _ = mg.add_field("topographic__elevation", heightmap.astype(float), at="node")
    fdir = FlowDirectorSteepest(mg)
    tl_diff = TransportLengthHillslopeDiffuser(mg, erodibility=0.001, slope_crit=0.6)
    if verbose: range_t = tqdm(range(150))
    else: range_t = range(150)
    for t in range_t:
        fdir.run_one_step()
        tl_diff.run_one_step(1.)
    res = mg.at_node['topographic__elevation']
    heightmap = res.reshape((N, N)) - 2
    peak = np.zeros((N, N))
    mask = (heightmap > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    component_label = label(mask).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    n_label = component_label.max()
    for l in range(1, n_label + 1):
        if (component_label == l).any():
            peak[component_label == l] = heightmap[component_label == l].max()
    downside = heightmap - 1

    heightmap = mountains2.get_heightmap(X, Y)
    upside = peak.copy() + np.maximum(downside, 0) / 2
    upside[upside > 0] += (heightmap.reshape((N, N)))[upside > 0]
    mg = RasterModelGrid((N, N))
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
    _ = mg.add_field("topographic__elevation", upside.astype(float), at="node")
    fdir = FlowDirectorSteepest(mg)
    tl_diff = TransportLengthHillslopeDiffuser(mg, erodibility=0.001, slope_crit=0.6)
    if verbose: range_t = tqdm(range(150))
    else: range_t = range(150)
    for t in range_t:
        fdir.run_one_step()
        tl_diff.run_one_step(1.)
    res = mg.at_node['topographic__elevation']
    upside = res.reshape((N, N))
    
    cv2.imwrite(str(folder/'upside.exr'), upside.astype(np.float32))
    cv2.imwrite(str(folder/'peak.exr'), peak.astype(np.float32))
    cv2.imwrite(str(folder/'downside.exr'), downside.astype(np.float32))
    with open(folder/f'{AssetFile.TileSize}.txt', "w") as f:
        f.write(f"{tile_size}\n")

    mountains1.cleanup()
    mountains2.cleanup()
    Element.called_time.pop("mountains")
    (folder / AssetFile.Finish).touch()



def assets_to_data(folder):
    data = {}
    upside = read(str(folder/'upside.exr'))
    N = upside.shape[0]
    data["upside"] = AC(upside.reshape(-1))
    data["downside"] = AC(read(str(folder/'downside.exr')).reshape(-1))
    data["peak"] = AC(read(str(folder/'peak.exr')).reshape(-1))
    L = float(np.loadtxt(f"{folder}/{AssetFile.TileSize}.txt"))
    return L, N, data
