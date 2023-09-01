# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import POINTER, c_char_p, c_float, c_int32

import cv2
import gin
import numpy as np
from numpy import ascontiguousarray as AC

from infinigen.terrain.utils import ASFLOAT, load_cdll, read, smooth
from infinigen.core.util.organization import AssetFile, Process
from infinigen.core.init import repo_root


@gin.configurable
def run_erosion(
    folder,
    Ns=[512, 2048],
    n_iters = [int(1e4), int(5e5)],
    mask_height_range=None,
    spatial=1,
    mask_range=(4, 47),
    ground_depth=25,
    sinking_rate=0.05,
):
    dll = load_cdll("terrain/lib/cpu/soil_machine/SoilMachine.so")
    func = dll.run
    func.argtypes = [
        POINTER(c_float), POINTER(c_float), POINTER(c_float),
        c_int32, c_int32, c_int32, c_int32, c_int32, c_float, c_char_p
    ]
    func.restype = None

    heightmap = read(str(folder/f'{AssetFile.Heightmap}.exr')).astype(np.float32)
    tile_size = float(np.loadtxt(f"{folder}/{AssetFile.TileSize}.txt"))

    soil_config_path = repo_root()/"infinigen/terrain/source/cpu/soil_machine/soil/sand.soil"

    for i, N, n_iter in zip(list(range(len(Ns))), Ns, n_iters):
        M = heightmap.shape[0]
        heightmap = cv2.resize(heightmap, (N, N))
        if N > M: heightmap = smooth(heightmap, 3)
        original_heightmap = heightmap.copy()
        ground_level = heightmap.min() - ground_depth
        heightmap = AC((heightmap - ground_level).astype(np.float32))
        result_heightmap = np.zeros_like(heightmap)
        watertrack = np.zeros_like(heightmap)
        func(
            ASFLOAT(heightmap),  ASFLOAT(result_heightmap),  ASFLOAT(watertrack),
            N, N, 0, n_iter, 0, spatial * tile_size, str(soil_config_path).encode('utf-8'),
        )
        heightmap = result_heightmap + ground_level
        watertrack = watertrack.reshape((N, N))
        watertrack = np.clip((watertrack - mask_range[0]) / (mask_range[1] - mask_range[0]), a_min=0, a_max=1)
        watertrack = watertrack ** 0.2
        if mask_height_range is not None:
            mask = np.clip((heightmap - mask_height_range[0]) / (mask_height_range[1] - mask_height_range[0]), a_min=0, a_max=1)
        else:
            mask = np.ones_like(heightmap)
        if i == 0 and len(Ns) > 1: heightmap -= watertrack * sinking_rate
        heightmap = heightmap * mask + original_heightmap * (1 - mask)
    if mask_height_range is not None:
        kernel = np.ones((5, 5), np.float32) / 25
        original_heightmap = heightmap.copy()
        for i in range(5):
            heightmap = cv2.filter2D(heightmap, -1, kernel)
        heightmap = heightmap * (1 - mask) + original_heightmap * mask

    cv2.imwrite(str(folder/f'{Process.Erosion}.{AssetFile.Heightmap}.exr'), heightmap)
    cv2.imwrite(str(folder/f'{Process.Erosion}.{AssetFile.Mask}.exr'), watertrack)

    del dll
