# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials.art import ArtFabric
from infinigen.assets.utils.decorate import read_co, select_vertices, write_co
from infinigen.assets.utils.object import new_grid
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform


class BlanketFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BlanketFactory, self).__init__(factory_seed, coarse)
        self.width = log_uniform(0.9, 1.2)
        self.size = self.width * log_uniform(0.4, 0.7)
        self.thickness = log_uniform(0.004, 0.008)

        materials = AssetList["BlanketFactory"]()
        self.surface = materials["surface"].assign_material()
        if self.surface == ArtFabric:
            self.surface = self.surface(self.factory_seed)

    def create_asset(self, **params) -> bpy.types.Object:
        obj = new_grid(
            x_subdivisions=64, y_subdivisions=int(self.size / self.width * 64)
        )
        obj.scale = self.width / 2, self.size / 2, 1
        butil.apply_transform(obj)
        unwrap_faces(obj)
        self.surface.apply(obj)
        return obj

    def fold(self, obj):
        theta = uniform(-np.pi / 6, np.pi / 6)
        y_margin = self.size * (0.5 - uniform(0.1, 0.3))
        obj.rotation_euler[-1] = theta
        obj.location[1] -= y_margin
        butil.apply_transform(obj, True)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(0, 1, 0))
        x, y, z = read_co(obj).T
        co = np.stack([x, np.where(y > 0, -y, y), np.where(y > 0, 0.05 - z, z)], -1)
        write_co(obj, co)
        obj.location[1] += y_margin
        butil.apply_transform(obj, True)
        obj.rotation_euler[-1] = -theta
        butil.apply_transform(obj)


class ComforterFactory(BlanketFactory):
    def create_asset(self, **params) -> bpy.types.Object:
        obj = super().create_asset(**params)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=0.01)
        return obj


class BoxComforterFactory(ComforterFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BoxComforterFactory, self).__init__(factory_seed, coarse)
        self.margin = uniform(0.3, 0.4)

    def create_asset(self, **params) -> bpy.types.Object:
        obj = super().create_asset(**params)
        x, y, _ = read_co(obj).T
        _x = (
            np.abs(x / self.margin - np.round(x / self.margin)) * self.margin
            < self.width / 64 / 2
        )
        _y = (
            np.abs(y / self.margin - np.round(y / self.margin)) * self.margin
            < self.width / 64 / 2
        )
        with butil.ViewportMode(obj, "EDIT"):
            select_vertices(obj, _x | _y)
            bpy.ops.mesh.remove_doubles(threshold=0.02)
        return obj
