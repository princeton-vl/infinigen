# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan

import os
import random

import bpy

from infinigen.assets.static_assets.base import StaticAssetFactory
from infinigen.core.tagging import tag_support_surfaces
from infinigen.core.util.math import FixedSeed


def static_category_factory(
    path_to_assets: str,
    tag_support=False,
    x_dim: float = None,
    y_dim: float = None,
    z_dim: float = None,
    rotation_euler: tuple[float] = None,
) -> StaticAssetFactory:
    """
    Create a factory for external asset import.
    tag_support: tag the planes of the object that are parallel to xy plane as support surfaces (e.g. shelves)
    x_dim, y_dim, z_dim: specify ONLY ONE dimension for the imported object. The object will be scaled accordingly.
    rotation_euler: sets the rotation of the object in euler angles. The object will not be rotated if not specified.
    """

    class StaticCategoryFactory(StaticAssetFactory):
        def __init__(self, factory_seed, coarse=False):
            super().__init__(factory_seed, coarse)
            with FixedSeed(factory_seed):
                self.path_to_assets = path_to_assets
                self.tag_support = tag_support
                self.asset_dir = path_to_assets
                self.x_dim, self.y_dim, self.z_dim = x_dim, y_dim, z_dim
                self.rotation_euler = rotation_euler
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
            if (
                self.x_dim is not None
                or self.y_dim is not None
                or self.z_dim is not None
            ):
                # check only one dimension is provided
                if (
                    sum(
                        [
                            1
                            for dim in [self.x_dim, self.y_dim, self.z_dim]
                            if dim is not None
                        ]
                    )
                    != 1
                ):
                    raise ValueError("Only one dimension can be provided")
                if self.x_dim is not None:
                    scale = self.x_dim / imported_obj.dimensions[0]
                elif self.y_dim is not None:
                    scale = self.y_dim / imported_obj.dimensions[1]
                else:
                    scale = self.z_dim / imported_obj.dimensions[2]
                imported_obj.scale = (scale, scale, scale)
            if self.tag_support:
                tag_support_surfaces(imported_obj)

            if imported_obj:
                return imported_obj
            else:
                raise ValueError(f"Failed to import asset: {self.asset_file}")

    return StaticCategoryFactory


# Create factory instances for different categories
StaticSofaFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Sofa"
)
StaticTableFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Table"
)
StaticShelfFactory = static_category_factory(
    "infinigen/assets/static_assets/source/Shelf", tag_support=True, z_dim=2
)
