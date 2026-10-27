# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen land tile terrain element (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/elements/landtiles.py)
# - David Yan, Jia Deng: optimized implementation

import logging
import os

import numpy as np

from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import AssetFile
from infinigen.perf.terrain.assets.landtiles.core import landtile_asset_for
from infinigen.perf.terrain.land_process.pool import land_process_pool
from infinigen.terrain.assets.landtiles import assets_to_data
from infinigen.terrain.elements import landtiles as _default


class LandTiles(_default.LandTiles):
    def load_assets(
        self,
        on_the_fly_instances=5,
        reused_instances=0,
    ):
        asset_paths = []
        if on_the_fly_instances > 0:
            # heightmaps are sampled here in order, since they consume the RNG; with
            # the pool enabled, the tiles' erosion and snowfall run concurrently
            with land_process_pool():
                created = set()
                for t, tile in enumerate(self.tiles):
                    for i in range(on_the_fly_instances):
                        folder = self.on_the_fly_asset_folder / tile / str(i)
                        # a tile name listed twice is generated once, as in the
                        # default; with the pool its finish marker may not exist
                        # yet on the second visit
                        if folder in created:
                            continue
                        created.add(folder)
                        if not (folder / AssetFile.Finish).exists():
                            logging.info(f"creating {tile} #{i}")
                            with FixedSeed(
                                int_hash(("LandTiles", self.assets_seed, t, i))
                            ):
                                landtile_asset_for(
                                    self.land_process,
                                    self.on_the_fly_asset_folder / tile / f"{i}",
                                    tile,
                                    device=self.device,
                                )
        for tile in self.tiles:
            for i in range(on_the_fly_instances):
                asset_paths.append(self.on_the_fly_asset_folder / tile / f"{i}")
            if reused_instances > 0:
                assert self.reused_asset_folder is not None
                assert (self.reused_asset_folder / tile).exists(), (
                    f"{self.reused_asset_folder / tile} does not exists"
                )
                all_instances = len(
                    [
                        x
                        for x in os.listdir(str(self.reused_asset_folder / tile))
                        if x[0] != "."
                    ]
                )
                sample = np.random.choice(
                    all_instances,
                    reused_instances,
                    replace=reused_instances > all_instances,
                )
                for i in range(reused_instances):
                    asset_paths.append(self.reused_asset_folder / tile / f"{sample[i]}")

        datas = {"direction": [np.zeros(0)]}
        for asset_path in asset_paths:
            tile_size, N, data = assets_to_data(
                asset_path, self.land_process, do_smooth=self.smooth
            )
            for key in data:
                if key in datas:
                    datas[key].append(data[key])
                else:
                    datas[key] = [data[key]]
        for key in datas:
            datas[key] = np.concatenate(datas[key])
        float_params = np.concatenate(
            (datas["heightmap"], datas["mask"], datas["direction"])
        ).astype(np.float32)
        return on_the_fly_instances + reused_instances, tile_size, N, float_params
