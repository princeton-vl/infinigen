# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import argparse
import importlib
import os
import sys
import time

import gin
import numpy as np

sys.path.append(os.getcwd())

from infinigen.assets.fluid.asset_cache import FireCachingSystem
from infinigen.core import init, surface

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
    init.apply_gin_configs(
        configs=[], overrides=[], config_folders=["infinigen_examples/configs_nature"]
    )
    surface.registry.initialize_from_gin()

    factory_name = args.asset
    factory = None
    for subdir in os.listdir("assets"):
        with gin.unlock_config():
            module = importlib.import_module(f'assets.{subdir.split(".")[0]}')
        if hasattr(module, factory_name):
            factory = getattr(module, factory_name)
            break
    if factory is None:
        raise ModuleNotFoundError(f"{factory_name} not Found.")

    cache_system = FireCachingSystem(asset_folder=args.asset_folder, create=True)
    cache_system.create_cached_assets(factory, args)
