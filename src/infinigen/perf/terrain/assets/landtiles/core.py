# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen land tile assets (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/assets/landtiles/core.py)
# - David Yan, Jia Deng: optimized implementation

from pathlib import Path

from infinigen.core.util.organization import AssetFile, LandTile, Process
from infinigen.perf import options
from infinigen.perf.terrain.land_process import planning
from infinigen.perf.terrain.land_process.planning import mark_finished
from infinigen.terrain.assets.landtiles import core as _default
from infinigen.terrain.assets.landtiles.ant_landscape import ant_landscape_asset
from infinigen.terrain.assets.landtiles.core import tile_sizes
from infinigen.terrain.assets.landtiles.custom import coast_asset, multi_mountains_asset


def landtile_asset(
    folder,
    preset_name,
    resolution=2048,
    device=None,
):
    """Generate one land tile asset in `folder`, skipping simulations the scene never
    reads when called through `landtile_asset_for`. Erosion and snowfall run here or
    in the enabled land-process pool, and the finish marker is written after them."""
    folder = Path(folder)
    tile_size = tile_sizes()[preset_name]
    kwargs = {}
    needed_process = planning.needed_process()
    if (
        options.only_needed_land_processes()
        and needed_process is not planning.NOT_GIVEN
    ):
        if needed_process not in [Process.Erosion, Process.Snowfall]:
            kwargs["erosion"] = False
        if needed_process != Process.Snowfall:
            kwargs["snowfall"] = False
    if preset_name == LandTile.MultiMountains:
        multi_mountains_asset(folder, tile_size, resolution, device, **kwargs)
    elif preset_name == LandTile.Coast:
        coast_asset(folder, tile_size, resolution, device, **kwargs)
    else:
        ant_landscape_asset(folder, preset_name, tile_size, resolution, **kwargs)
    mark_finished(folder, folder / AssetFile.Finish)


def landtile_asset_for(needed_process, *args, **kwargs):
    """`landtile_asset` for a scene that reads `needed_process`'s outputs (None,
    `Process.Erosion` or `Process.Snowfall`), simulating only what those need unless
    `only_needed_land_processes` is off. Asset-library generation calls
    `landtile_asset` directly and gets every output."""
    with planning.needed_processes(needed_process):
        return _default.landtile_asset(*args, **kwargs)
