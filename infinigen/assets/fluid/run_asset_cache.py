# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import argparse
import time

import gin
import numpy as np

from infinigen.assets.fluid import cached_factory_wrappers
from infinigen.assets.fluid.asset_cache import FireCachingSystem
from infinigen.core import init
from infinigen.core.util.test_utils import setup_gin

if __name__ == "__main__":
    time.sleep(np.random.uniform(0, 3))

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--asset_folder", type=str)
    parser.add_argument("-a", "--asset")
    parser.add_argument("-s", "--start_frame", type=int, default=-20)
    parser.add_argument("-d", "--simulation_duration", type=int, default=30)
    parser.add_argument("-e", "--estimate_domain", action="store_true")
    parser.add_argument("-r", "--resolution", type=int)
    parser.add_argument("--dissolve_speed", type=int, default=25)
    parser.add_argument("--dom_scale", type=float, default=1)

    args = init.parse_args_blender(parser)
    setup_gin("infinigen_examples/configs_nature", configs=["base_nature.gin"])

    # Use gin.unlock_config() when getting the factory class
    with gin.unlock_config():
        factory = getattr(cached_factory_wrappers, args.asset, None)
        if factory is None:
            raise ModuleNotFoundError(
                f"{args.asset} not found in cached_factory_wrappers."
            )

        cache_system = FireCachingSystem(asset_folder=args.asset_folder, create=True)
        cache_system.create_cached_assets(factory, args)
