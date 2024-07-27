# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan

import os
import random

import bpy

from infinigen.assets.static_assets.base import StaticAssetFactory
from infinigen.core.util.math import FixedSeed


def static_category_factory(category) -> StaticAssetFactory:
    class StaticCategoryFactory(StaticAssetFactory):
        def __init__(self, factory_seed, coarse=False):
            super().__init__(factory_seed, coarse)
            with FixedSeed(factory_seed):
                self.category = category
                self.asset_dir = os.path.join(self.root_asset_dir, category)
                asset_files = [
                    f
                    for f in os.listdir(self.asset_dir)
                    if f.lower().endswith(tuple(self.import_map.keys()))
                ]
                if not asset_files or len(asset_files) == 0:
                    raise ValueError(f"No valid asset files found in {self.asset_dir}")
                self.asset_file = random.choice(asset_files)

        def create_asset(self, **params) -> bpy.types.Object:
            file_path = os.path.join(self.asset_dir, self.asset_file)
            imported_obj = self.import_file(file_path)

            if imported_obj:
                return imported_obj
            else:
                raise ValueError(f"Failed to import asset: {self.asset_file}")

    return StaticCategoryFactory


# Create factory instances for different categories
StaticSofaFactory = static_category_factory("Sofa")
StaticTableFactory = static_category_factory("Table")
StaticShelfFactory = static_category_factory("Shelf")
StaticVendingMachineFactory = static_category_factory("VendingMachine")
