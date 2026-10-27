# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen upside-down mountains (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/assets/upsidedown_mountains.py)
# - David Yan, Jia Deng: optimized implementation

from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label

from infinigen.core.util.organization import AssetFile
from infinigen.core.util.random import random_general as rg
from infinigen.perf.terrain.land_process.snowfall import transport_length_diffusion
from infinigen.terrain.elements.core import Element
from infinigen.terrain.elements.mountains import Mountains


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
    """Upside, peak and downside heightmaps of a `tile_size` upside-down mountain tile,
    written to `folder`. `min_freq`, `max_freq`, `height` and `coverage` set the
    mountains' base frequency range, height and coverage."""

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
    radius = (x**2 + y**2) ** 0.5
    heightmap *= 1 - np.clip((radius - 0.8) * 5, a_min=0, a_max=1)
    # bitwise-identical numpy replica of the default's landlab diffusion
    heightmap = (
        transport_length_diffusion(
            heightmap.astype(float), 150, erodibility=0.001, slope_crit=0.6
        )
        - 2
    )
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
    upside = transport_length_diffusion(
        upside.astype(float), 150, erodibility=0.001, slope_crit=0.6
    )

    cv2.imwrite(str(folder / "upside.exr"), upside.astype(np.float32))
    cv2.imwrite(str(folder / "peak.exr"), peak.astype(np.float32))
    cv2.imwrite(str(folder / "downside.exr"), downside.astype(np.float32))
    with open(folder / f"{AssetFile.TileSize}.txt", "w") as f:
        f.write(f"{tile_size}\n")

    mountains1.cleanup()
    mountains2.cleanup()
    Element.called_time.pop("mountains")
    (folder / AssetFile.Finish).touch()
