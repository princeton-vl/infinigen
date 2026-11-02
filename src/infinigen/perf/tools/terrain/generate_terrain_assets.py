# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen terrain asset generation (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/tools/terrain/generate_terrain_assets.py)
# - David Yan, Jia Deng: optimized implementation

"""Asset-library generation that can run land processes and caves in pool workers."""

from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import AssetFile, Assets, LandTile
from infinigen.perf.terrain.assets.caves.core import plan_caves_asset
from infinigen.perf.terrain.land_process.planning import run_land_process_steps
from infinigen.perf.terrain.land_process.pool import land_process_pool
from infinigen.terrain.assets.landtiles import landtile_asset
from infinigen.terrain.assets.upsidedown_mountains import upsidedown_mountains_asset


def asset_generation(
    output_folder,
    assets,
    instance_ids,
    seed,
    device,
    check_only=False,
):
    with land_process_pool():
        for i in instance_ids:
            for asset in assets:
                if asset in [
                    LandTile.Mesa,
                    LandTile.Canyon,
                    LandTile.Canyons,
                    LandTile.Cliff,
                    LandTile.Mountain,
                    LandTile.River,
                    LandTile.Volcano,
                    LandTile.MultiMountains,
                    LandTile.Coast,
                ]:
                    if not (output_folder / asset / f"{i}" / AssetFile.Finish).exists():
                        print(asset, i)
                        if not check_only:
                            with FixedSeed(int_hash([asset, seed, i])):
                                landtile_asset(
                                    output_folder / asset / f"{i}", asset, device=device
                                )
                if asset == Assets.UpsidedownMountains:
                    if not (output_folder / asset / f"{i}" / AssetFile.Finish).exists():
                        print(asset, i)
                        if not check_only:
                            with FixedSeed(int_hash([asset, seed, i])):
                                upsidedown_mountains_asset(
                                    output_folder / Assets.UpsidedownMountains / f"{i}",
                                    device=device,
                                )
                if asset == Assets.Caves:
                    if not (output_folder / asset / f"{i}" / AssetFile.Finish).exists():
                        print(asset, i)
                        if not check_only:
                            cave_seed = int_hash([asset, seed, i])
                            folder = output_folder / Assets.Caves / f"{i}"
                            with FixedSeed(cave_seed):
                                step = plan_caves_asset(folder, cave_seed)
                            run_land_process_steps(folder, [step])
