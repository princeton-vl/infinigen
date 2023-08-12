# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import os

import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.assets.caves import assets_to_data, caves_asset
from infinigen.terrain.utils import random_int, random_int_large
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.random import random_general as rg
from infinigen.core.util.organization import AssetFile
from .core import Element


# this is a special element that only exists as an operation to other elements
@gin.configurable
class Caves(Element):
    def __init__(
        self,
        on_the_fly_asset_folder,
        reused_asset_folder,
        n_lattice=3,
        is_horizontal=0,
        frequency=0.03,
        randomness=1,
        height_offset=0,
        deepest_level=-10,
        scale_increase=("uniform", 1, 2),
        noise_octaves=9,
        noise_scale=5,
        noise_freq=("log_uniform", 0.1, 0.3),
        smoothness=0.5,
    ):
        self.on_the_fly_asset_folder = on_the_fly_asset_folder
        self.reused_asset_folder = reused_asset_folder
        nonpython_seed = random_int()
        self.assets_seed = random_int_large()
        noise_freq = rg(noise_freq)
        n_instances, N, float_data = self.load_assets()
        self.int_params = AC(np.array([
            nonpython_seed, n_lattice, is_horizontal, n_instances, N,
        ]).astype(np.int32))
        self.float_params = AC(np.concatenate((np.array([
            randomness, frequency, deepest_level, rg(scale_increase),
            noise_octaves, noise_freq, rg(noise_scale), height_offset, smoothness,
        ]), float_data)).astype(np.float32))

    

    @gin.configurable
    def load_assets(
        self,
        on_the_fly_instances=5,
        reused_instances=0,
    ):
        asset_paths = []
        if on_the_fly_instances > 0:
            for i in range(on_the_fly_instances):
                if not (self.on_the_fly_asset_folder / str(i) / AssetFile.Finish).exists():
                    with FixedSeed(int_hash(("Caves", self.assets_seed, i))):
                        caves_asset(self.on_the_fly_asset_folder / f"{i}")
        for i in range(on_the_fly_instances):
            asset_paths.append(self.on_the_fly_asset_folder / f"{i}")
        if reused_instances > 0:
            assert(self.reused_asset_folder is not None and self.reused_asset_folder.exists())
            all_instances = len([x for x in os.listdir(str(self.reused_asset_folder)) if x[0] != '.'])
            sample = np.random.choice(all_instances, reused_instances, replace=reused_instances > all_instances)
            for i in range(reused_instances):
                asset_paths.append(self.reused_asset_folder / f"{sample[i]}")
        
        datas = {}
        for asset_path in asset_paths:
            N, data = assets_to_data(asset_path)
            for key in data:
                if key in datas:
                    datas[key].append(data[key])
                else:
                    datas[key] = [data[key]]
        for key in datas:
            datas[key] = np.concatenate(datas[key])
        float_params = np.concatenate((datas["bounding_box"], datas["occupancy"])).astype(np.float32)
        return on_the_fly_instances + reused_instances, N, float_params