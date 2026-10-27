# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen cave terrain element (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/elements/caves.py)
# - David Yan, Jia Deng: optimized implementation

import logging
import os

import numpy as np

from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import AssetFile
from infinigen.perf.terrain.assets.caves.core import plan_caves_asset
from infinigen.perf.terrain.land_process.planning import run_land_process_steps
from infinigen.perf.terrain.land_process.pool import land_process_pool
from infinigen.terrain.assets.caves import assets_to_data
from infinigen.terrain.elements import caves as _default


class Caves(_default.Caves):
    def load_assets(
        self,
        on_the_fly_instances=5,
        reused_instances=0,
    ):
        asset_paths = []
        if on_the_fly_instances > 0:
            # each cave is built from its own fixed seed; with the pool enabled,
            # the caves are built concurrently
            with land_process_pool():
                for i in range(on_the_fly_instances):
                    folder = self.on_the_fly_asset_folder / f"{i}"
                    if not (folder / AssetFile.Finish).exists():
                        seed = int_hash(("Caves", self.assets_seed, i))
                        with FixedSeed(seed):
                            logging.info(f"creating cave #{i}")
                            step = plan_caves_asset(folder, seed)
                        run_land_process_steps(folder, [step])
        for i in range(on_the_fly_instances):
            asset_paths.append(self.on_the_fly_asset_folder / f"{i}")
        if reused_instances > 0:
            assert (
                self.reused_asset_folder is not None
                and self.reused_asset_folder.exists()
            )
            all_instances = len(
                [x for x in os.listdir(str(self.reused_asset_folder)) if x[0] != "."]
            )
            sample = np.random.choice(
                all_instances,
                reused_instances,
                replace=reused_instances > all_instances,
            )
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
        float_params = np.concatenate(
            (datas["bounding_box"], datas["occupancy"])
        ).astype(np.float32)
        return on_the_fly_instances + reused_instances, N, float_params
