# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


'''
fileheader placeholder
'''

import os
import sys
import argparse
from pathlib import Path

import bpy
from infinigen.terrain.assets.caves import caves_asset
from infinigen.terrain.assets.landtiles import landtile_asset
from infinigen.terrain.assets.upsidedown_mountains import upsidedown_mountains_asset
from infinigen.core.util import blender as butil
from infinigen.core.util.math import int_hash, FixedSeed
from infinigen.core.util.organization import Assets, LandTile, AssetFile
from infinigen.core import init

def asset_generation(
    output_folder,
    assets,
    instance_ids,
    seed,
    device,
    check_only=False,
):
    for i in instance_ids:
        for asset in assets:
            if asset in [LandTile.Mesa, LandTile.Canyon, LandTile.Canyons, LandTile.Cliff, LandTile.Mountain, LandTile.River, LandTile.Volcano, LandTile.MultiMountains, LandTile.Coast]:
                if not (output_folder/asset/f"{i}"/AssetFile.Finish).exists():
                    print(asset, i)
                    if not check_only:
                        with FixedSeed(int_hash([asset, seed, i])):
                            landtile_asset(output_folder/asset/f"{i}", asset, device=device)
            if asset == Assets.UpsidedownMountains:
                if not (output_folder/asset/f"{i}"/AssetFile.Finish).exists():
                    print(asset, i)
                    if not check_only:
                        with FixedSeed(int_hash([asset, seed, i])):
                            upsidedown_mountains_asset(output_folder/Assets.UpsidedownMountains/f"{i}", device=device)
            if asset == Assets.Caves:
                if not (output_folder/asset/f"{i}"/AssetFile.Finish).exists():
                    print(asset, i)
                    if not check_only:
                        with FixedSeed(int_hash([asset, seed, i])):
                            caves_asset(output_folder/Assets.Caves/f"{i}")


if __name__ == "__main__":
    # by default infinigen does on-the-fly terrain asset generation, but if you want to pre-generate a pool of assets, run this code
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--assets', nargs='+', default=[
        LandTile.MultiMountains,
        LandTile.Coast,
        LandTile.Mesa,
        LandTile.Canyon,
        LandTile.Canyons,
        LandTile.Cliff,
        LandTile.Mountain,
        LandTile.River,
        LandTile.Volcano,
        Assets.UpsidedownMountains,
        Assets.Caves,
    ])
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, default=1)
    parser.add_argument('-f', '--folder')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check_only', type=int, default=0)
    parser.add_argument('--device', type=str, default="cpu")
    args = init.parse_args_blender(parser)

    bpy.ops.preferences.addon_enable(module='add_mesh_extra_objects')
    bpy.ops.preferences.addon_enable(module='ant_landscape')
    butil.clear_scene(targets=[bpy.data.objects])
    asset_generation(Path(args.folder), args.assets, list(range(args.start, args.end)), args.seed, args.device, check_only=args.check_only)
