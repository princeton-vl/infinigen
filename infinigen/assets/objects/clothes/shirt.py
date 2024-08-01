# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials.art import ArtFabric
from infinigen.assets.utils.decorate import (
    read_center,
    read_normal,
    remove_faces,
    subsurf,
    write_co,
)
from infinigen.assets.utils.draw import remesh_fill
from infinigen.assets.utils.object import new_circle
from infinigen.assets.utils.uv import wrap_front_back
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform


class ShirtFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ShirtFactory, self).__init__(factory_seed, coarse)
        self.width = log_uniform(0.45, 0.55)
        self.size = self.width + uniform(0.25, 0.3)
        self.size_neck = uniform(0.1, 0.15) * self.size
        self.type = np.random.choice(["short", "long"])
        match self.type:
            case "short":
                self.sleeve_length = self.size / 2 + uniform(-0.35, -0.3)
            case _:
                self.sleeve_length = self.size / 2 + uniform(-0.05, 0.0)
        self.sleeve_width = uniform(0.14, 0.18)
        self.sleeve_angle = uniform(np.pi / 6, np.pi / 4)
        self.thickness = log_uniform(0.02, 0.03)
        materials = AssetList["ShirtFactory"]()
        self.surface = materials["surface"].assign_material()
        if self.surface == ArtFabric:
            self.surface = self.surface(self.factory_seed)

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = (
            0,
            self.width / 2,
            self.width / 2,
            self.width / 2 + self.sleeve_length * np.sin(self.sleeve_angle),
            self.width / 2
            + self.sleeve_length * np.sin(self.sleeve_angle)
            + self.sleeve_width * np.cos(self.sleeve_angle),
            self.width / 2,
            self.width / 4,
            0,
        )

        y_anchors = (
            0,
            0,
            self.size - self.sleeve_width / np.sin(self.sleeve_angle),
            self.size
            - self.sleeve_width / np.sin(self.sleeve_angle)
            - self.sleeve_length * np.cos(self.sleeve_angle),
            self.size
            - self.sleeve_width / np.sin(self.sleeve_angle)
            - self.sleeve_length * np.cos(self.sleeve_angle)
            + self.sleeve_width * np.sin(self.sleeve_angle),
            self.size,
            self.size + self.size_neck,
            self.size + self.size_neck * uniform(0.3, 0.7),
        )

        obj = new_circle(vertices=len(x_anchors))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.edge_face_add()
            bpy.ops.mesh.flip_normals()
        write_co(obj, np.stack([x_anchors, y_anchors, np.zeros_like(x_anchors)], -1))
        butil.modify_mesh(obj, "MIRROR", use_axis=(True, False, False))
        remesh_fill(obj, 0.02)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        x, y, z = read_center(obj).T
        x_, y_, z_ = read_normal(obj).T
        remove_faces(obj, (y_ < -0.5) | ((y_ > 0.5) & (x_ * x < 0)))
        with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles(threshold=1e-3)
        butil.modify_mesh(obj, "BEVEL", width=self.sleeve_width * uniform(0.1, 0.15))
        subsurf(obj, 1)
        wrap_front_back(obj, self.surface)
        return obj
