# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import os

import gin
import numpy as np
from numpy import ascontiguousarray as AC

from infinigen.terrain.assets.landtiles import assets_to_data, landtile_asset
from infinigen.terrain.utils import random_int, random_int_large
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import Tags, Materials, LandTile, Process, Transparency, ElementNames, ElementTag, AssetFile
from infinigen.core.util.random import random_general as rg

from .core import Element


def none_to_0(x):
    if x is None: return 0
    return x

@gin.configurable
class LandTiles(Element):
    name = ElementNames.LandTiles
    def __init__(
        self,
        device,
        caves,
        on_the_fly_asset_folder, # for tiledlandscape the folder is the containing folder not specific type folder
        reused_asset_folder,
        n_lattice=1,
        tiles=[LandTile.MultiMountains],
        tile_density=1,
        randomness=0,
        attribute_probability=1,
        attribute_distance_range=1e9,
        island_probability=0,
        tile_heights=[-0.1],
        land_process=("choice", [Process.Erosion, None], [0.65, 0.35]),
        height_modification_start=None, height_modification_end=None,
        attribute_modification_start_height=None,
        attribute_modification_end_height=None,
        attribute_modification_distort_freq=1,
        attribute_modification_distort_mag=2,
        empty_below=-1e9,
        y_tilt=0,
        y_tilt_clip=0,
        material=Materials.MountainCollection,
        transparency=Transparency.Opaque,
        use_cblerp=False,
        smooth=False,
    ):
        self.device = device
        self.on_the_fly_asset_folder = on_the_fly_asset_folder
        self.reused_asset_folder = reused_asset_folder
        nonpython_seed = random_int()
        self.assets_seed = random_int_large()
        self.tiles = tiles
        self.attribute_modification_start_height = attribute_modification_start_height = rg(attribute_modification_start_height)
        self.attribute_modification_end_height = attribute_modification_end_height = rg(attribute_modification_end_height)
        self.smooth = smooth
        self.aux_names = []
        land_process = rg(land_process)
        sharpen = 0
        mask_random_freq = 0
        if land_process == Process.Snowfall:
            self.aux_names.append(Materials.Snow)
            sharpen = 0.9
            mask_random_freq = 5
        elif land_process == Process.Erosion:
            self.aux_names.append(Materials.Eroded)
        elif land_process == Process.Eruption:
            land_process = Process.Erosion
            self.aux_names.append(Materials.Lava)
        elif land_process == Process.IceErosion:
            land_process = Process.Erosion
            self.aux_names.append(None)
        else:
            self.aux_names.append(None)
        self.land_process = land_process
        if attribute_modification_start_height is not None:
            self.aux_names.append(Materials.Beach)
        else:
            self.aux_names.append(None)
        if caves is None:
            self.aux_names.append(None)
        else:
            self.aux_names.append(Tags.Cave)
            self.int_params2 = caves.int_params
            self.float_params2 = caves.float_params

        n_instances, tile_size, N, float_data = self.load_assets()

        frequency = 1 / (tile_size * 0.67) * tile_density

        self.int_params = AC(np.concatenate((np.array([
            nonpython_seed, n_lattice, len(tiles), height_modification_start is not None,
            attribute_modification_start_height is not None, n_instances, N, use_cblerp,
        ]), )).astype(np.int32))
        self.float_params = AC(np.concatenate((np.array([
            randomness, frequency, attribute_probability, attribute_distance_range, island_probability, tile_size,
            none_to_0(height_modification_start), none_to_0(height_modification_end),
            none_to_0(attribute_modification_start_height), none_to_0(attribute_modification_end_height),
            attribute_modification_distort_freq, attribute_modification_distort_mag, empty_below, y_tilt, y_tilt_clip, sharpen, mask_random_freq,
            *tile_heights,
        ]), float_data)).astype(np.float32))
    
        self.meta_params = [caves is not None]
        Element.__init__(self, "landtiles", material, transparency)
        self.tag = ElementTag.Terrain

    @gin.configurable
    def load_assets(
        self,
        on_the_fly_instances=5,
        reused_instances=0,
    ):
        asset_paths = []
        if on_the_fly_instances > 0:
            for t, tile in enumerate(self.tiles):
                for i in range(on_the_fly_instances):
                    if not (self.on_the_fly_asset_folder / tile / str(i) / AssetFile.Finish).exists():
                        with FixedSeed(int_hash(("LandTiles", self.assets_seed, t, i))):
                            landtile_asset(self.on_the_fly_asset_folder / tile / f"{i}", tile, device=self.device)
        for tile in self.tiles:
            for i in range(on_the_fly_instances):
                asset_paths.append(self.on_the_fly_asset_folder / tile / f"{i}")
            if reused_instances > 0:
                assert self.reused_asset_folder is not None
                assert (self.reused_asset_folder / tile).exists(), f"{self.reused_asset_folder / tile} does not exists"
                all_instances = len([x for x in os.listdir(str(self.reused_asset_folder / tile)) if x[0] != '.'])
                sample = np.random.choice(all_instances, reused_instances, replace=reused_instances > all_instances)
                for i in range(reused_instances):
                    asset_paths.append(self.reused_asset_folder / tile / f"{sample[i]}")

        datas = {"direction": [np.zeros(0)]}
        for asset_path in asset_paths:
            tile_size, N, data = assets_to_data(asset_path, self.land_process, do_smooth=self.smooth)
            for key in data:
                if key in datas:
                    datas[key].append(data[key])
                else:
                    datas[key] = [data[key]]
        for key in datas:
            datas[key] = np.concatenate(datas[key])
        float_params = np.concatenate((datas["heightmap"], datas["mask"], datas["direction"])).astype(np.float32)
        return on_the_fly_instances + reused_instances, tile_size, N, float_params


class Volcanos(LandTiles):
    name = ElementNames.Volcanos
    def __init__(
        self,
        device,
        caves,
        on_the_fly_asset_folder,
        reused_asset_folder,
        tile_density=0.25,
    ):
        LandTiles.__init__(
            self,
            device,
            caves,
            on_the_fly_asset_folder,
            reused_asset_folder,
            tiles=[LandTile.Volcano],
            tile_heights=[-2],
            tile_density=tile_density,
            attribute_probability=0.5,
            attribute_distance_range=150,
            land_process=Process.Eruption,
            height_modification_start=-0.5, height_modification_end=-1.5,
            attribute_modification_start_height=None,
            attribute_modification_end_height=None,
            randomness=1,
        )
        self.tag = ElementTag.Volcanos

class FloatingIce(LandTiles):
    name = ElementNames.FloatingIce
    def __init__(
        self,
        device,
        caves,
        on_the_fly_asset_folder,
        reused_asset_folder,
        tile_density=1,
    ):
        LandTiles.__init__(
            self,
            device,
            caves,
            on_the_fly_asset_folder,
            reused_asset_folder,
            tiles=[LandTile.Mesa],
            tile_density=tile_density,
            tile_heights=[-12.15],
            land_process=Process.IceErosion,
            transparency=Transparency.CollectiveTransparent,
            height_modification_start=None, height_modification_end=None,
            attribute_modification_start_height=None,
            attribute_modification_end_height=None,
            empty_below=-0.4,
            randomness=1,
        )
        self.tag = ElementTag.FloatingIce