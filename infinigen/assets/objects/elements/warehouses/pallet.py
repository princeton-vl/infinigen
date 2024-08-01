# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import wood
from infinigen.assets.utils.decorate import read_normal
from infinigen.assets.utils.object import join_objects, new_bbox, new_cube
from infinigen.core import tags as t
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj


class PalletFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PalletFactory, self).__init__(factory_seed, coarse)
        self.depth = uniform(1.2, 1.4)
        self.width = uniform(1.2, 1.4)
        self.thickness = uniform(0.01, 0.015)
        self.tile_width = uniform(0.06, 0.1)
        self.tile_slackness = uniform(1.5, 2)
        self.height = uniform(0.2, 0.25)
        self.surface = wood

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        bbox = new_bbox(0, self.width, 0, self.depth, 0, self.height)
        write_attr_data(
            bbox,
            f"{PREFIX}{t.Subpart.SupportSurface.value}",
            read_normal(bbox)[:, -1] > 0.5,
            "INT",
            "FACE",
        )
        return bbox

    def create_asset(self, **params) -> bpy.types.Object:
        vertical = self.make_vertical()
        vertical.location[-1] = self.thickness
        vertical_ = deep_clone_obj(vertical)
        vertical_.location[-1] = self.height - self.thickness
        horizontal = self.make_horizontal()
        horizontal_ = deep_clone_obj(horizontal)
        horizontal_.location[-1] = self.height - 2 * self.thickness
        support = self.make_support()
        support.location[-1] = 2 * self.thickness
        obj = join_objects([horizontal, horizontal_, vertical, vertical_, support])
        return obj

    def make_vertical(self):
        obj = new_cube()
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.tile_width / 2, self.depth / 2, self.thickness / 2
        butil.apply_transform(obj)
        count = (
            int(
                np.floor(
                    (self.width - self.tile_width)
                    / self.tile_width
                    / self.tile_slackness
                )
                / 2
            )
            * 2
        )
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=((self.width - self.tile_width) / count, 0, 0),
            count=count + 1,
        )
        return obj

    def make_horizontal(self):
        obj = new_cube()
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.width / 2, self.tile_width / 2, self.thickness / 2
        butil.apply_transform(obj)
        count = (
            int(
                np.floor(
                    (self.depth - self.tile_width)
                    / self.tile_width
                    / self.tile_slackness
                )
                / 2
            )
            * 2
        )
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, (self.depth - self.tile_width) / count, 0),
            count=count + 1,
        )
        return obj

    def make_support(self):
        obj = new_cube()
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = (
            self.tile_width / 2,
            self.tile_width / 2,
            self.height / 2 - 2 * self.thickness,
        )
        butil.apply_transform(obj)
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=((self.width - self.tile_width) / 2, 0, 0),
            count=3,
        )
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, (self.depth - self.tile_width) / 2, 0),
            count=3,
        )
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets)
