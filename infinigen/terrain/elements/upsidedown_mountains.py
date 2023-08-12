# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import os

import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.assets.upsidedown_mountains import assets_to_data, upsidedown_mountains_asset
from infinigen.terrain.utils import random_int, random_int_large
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import Materials, Transparency, ElementNames, ElementTag, Tags, AssetFile

from .core import Element

@gin.configurable
class UpsidedownMountains(Element):
    name = ElementNames.UpsidedownMountains
    def __init__(
        self,
        device,
        on_the_fly_asset_folder,
        reused_asset_folder,
        floating_height=5,
        randomness=0,
        frequency=0.005,
        perturb_octaves=9,
        perturb_freq=1,
        perturb_scale=0.2,
        material=Materials.MountainCollection,
        transparency=Transparency.Opaque,
    ):
        self.device = device
        self.on_the_fly_asset_folder = on_the_fly_asset_folder
        self.reused_asset_folder = reused_asset_folder
        nonpython_seed = random_int()
        self.assets_seed = random_int_large()
        self.aux_names = [Tags.UpsidedownMountainsLowerPart]
        n_instances, L, N, float_data = self.load_assets()
        self.int_params = AC(np.concatenate((np.array([nonpython_seed, n_instances, N]),)).astype(np.int32))
        self.float_params = AC(np.concatenate((np.array([L, floating_height, randomness, frequency, perturb_octaves, perturb_freq, perturb_scale]), float_data)).astype(np.float32))
        
        Element.__init__(self, "upsidedown_mountains", material, transparency)
        self.tag = ElementTag.UpsidedownMountains

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
                    with FixedSeed(int_hash(("UpsidedownMountains", self.assets_seed, i))):
                        upsidedown_mountains_asset(self.on_the_fly_asset_folder / f"{i}", device=self.device)
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
            L, N, data = assets_to_data(asset_path)
            for key in data:
                if key in datas:
                    datas[key].append(data[key])
                else:
                    datas[key] = [data[key]]
        for key in datas:
            datas[key] = np.concatenate(datas[key])
        float_params = np.concatenate((datas["upside"], datas["downside"], datas["peak"])).astype(np.float32)
        return on_the_fly_instances + reused_instances, L, N, float_params