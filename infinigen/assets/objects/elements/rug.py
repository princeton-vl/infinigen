# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials.art import ArtRug
from infinigen.assets.utils.object import new_base_circle, new_bbox, new_plane
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import clip_gaussian


class RugFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(RugFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.width = clip_gaussian(3, 1, 2, 6)
            self.length = self.width * uniform(1, 1.5)
            self.rug_shape = np.random.choice(
                ["rectangle", "circle", "rounded", "ellipse"]
            )
            if self.rug_shape == "circle":
                self.length = self.width
            self.rounded_buffer = self.width * uniform(0.1, 0.5)
            self.thickness = uniform(0.01, 0.02)
            material_assignments = AssetList["RugFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            if self.surface == ArtRug:
                self.surface = self.surface(self.factory_seed)

    def build_shape(self):
        match self.rug_shape:
            case "rectangle":
                obj = new_plane()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
            case "rounded":
                obj = new_plane()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
                butil.modify_mesh(obj, "BEVEL", width=self.rounded_buffer, segments=16)
            case _:
                obj = new_base_circle(vertices=128)
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.edge_face_add()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.length / 2,
            self.length / 2,
            -self.width / 2,
            self.width / 2,
            0,
            self.thickness,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.build_shape()
        wrap_sides(obj, self.surface, "z", "x", "y")
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=1)
        return obj
