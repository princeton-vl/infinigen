# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring, subsurf
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects, new_cylinder
from infinigen.assets.utils.uv import wrap_front_back
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class BottleFactory(AssetFactory):
    z_neck_offset = 0.05
    z_waist_offset = 0.15

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.z_length = uniform(0.15, 0.25)
            self.x_length = self.z_length * uniform(0.15, 0.25)
            self.x_cap = uniform(0.3, 0.35)
            self.bottle_type = np.random.choice(
                ["beer", "bordeaux", "champagne", "coke", "vintage"]
            )
            self.bottle_width = uniform(0.002, 0.005)
            self.z_waist = 0
            match self.bottle_type:
                case "beer":
                    self.z_neck = uniform(0.5, 0.6)
                    self.z_cap = uniform(0.05, 0.08)
                    neck_size = uniform(0.06, 0.1)
                    neck_ratio = uniform(0.4, 0.5)
                    self.x_anchors = [
                        0,
                        1,
                        1,
                        (neck_ratio + 1) / 2 + (1 - neck_ratio) / 2 * self.x_cap,
                        neck_ratio + (1 - neck_ratio) * self.x_cap,
                        self.x_cap,
                        self.x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        self.z_neck,
                        self.z_neck + uniform(0.6, 0.7) * neck_size,
                        self.z_neck + neck_size,
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 1, 0, 1, 1, 1, 0]
                case "bordeaux":
                    self.z_neck = uniform(0.6, 0.7)
                    self.z_cap = uniform(0.1, 0.15)
                    neck_size = uniform(0.1, 0.15)
                    self.x_anchors = (
                        0,
                        1,
                        1,
                        (1 + self.x_cap) / 2,
                        self.x_cap,
                        self.x_cap,
                        0,
                    )
                    self.z_anchors = [
                        0,
                        0,
                        self.z_neck,
                        self.z_neck + uniform(0.6, 0.7) * neck_size,
                        self.z_neck + neck_size,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 1, 0, 1, 1, 0]
                case "champagne":
                    self.z_neck = uniform(0.4, 0.5)
                    self.z_cap = uniform(0.05, 0.08)
                    self.x_anchors = [
                        0,
                        1,
                        1,
                        1,
                        (1 + self.x_cap) / 2,
                        self.x_cap,
                        self.x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        self.z_neck,
                        self.z_neck + uniform(0.08, 0.1),
                        self.z_neck + uniform(0.15, 0.18),
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 1, 0, 0, 1, 1, 0]
                case "coke":
                    self.z_waist = uniform(0.4, 0.5)
                    self.z_neck = self.z_waist + uniform(0.2, 0.25)
                    self.z_cap = uniform(0.05, 0.08)
                    self.x_anchors = [
                        0,
                        uniform(0.85, 0.95),
                        1,
                        uniform(0.85, 0.95),
                        1,
                        1,
                        self.x_cap,
                        self.x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        uniform(0.08, 0.12),
                        uniform(0.18, 0.25),
                        self.z_waist,
                        self.z_neck,
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 0, 0, 1, 1, 1, 1, 0]
                case "vintage":
                    self.z_waist = uniform(0.1, 0.15)
                    self.z_neck = uniform(0.7, 0.75)
                    self.z_cap = uniform(0.0, 0.08)
                    x_lower = uniform(0.85, 0.95)
                    self.x_anchors = [
                        0,
                        x_lower,
                        (x_lower + 1) / 2,
                        1,
                        1,
                        (self.x_cap + 1) / 2,
                        self.x_cap,
                        self.x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        self.z_waist - uniform(0.1, 0.15),
                        self.z_waist,
                        self.z_neck,
                        self.z_neck + uniform(0.1, 0.2),
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 0, 1, 1, 0, 1, 1, 0]

            material_assignments = AssetList["BottleFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.wrap_surface = material_assignments["wrap_surface"].assign_material()
            if self.wrap_surface == text.Text:
                self.wrap_surface = text.Text(self.factory_seed, False)

            self.cap_surface = material_assignments["cap_surface"].assign_material()
            scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
            self.scratch, self.edge_wear = material_assignments["wear_tear"]
            self.scratch = None if uniform() > scratch_prob else self.scratch
            self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

            self.texture_shared = uniform() < 0.2
            self.cap_subsurf = uniform() < 0.5

    def create_asset(self, **params) -> bpy.types.Object:
        bottle = self.make_bottle()
        wrap = self.make_wrap(bottle)
        cap = self.make_cap()
        obj = join_objects([bottle, wrap, cap])

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)

    def make_bottle(self):
        x_anchors = np.array(self.x_anchors) * self.x_length
        z_anchors = np.array(self.z_anchors) * self.z_length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors, np.nonzero(self.is_vector)[0])
        subsurf(obj, 1)
        if self.bottle_width > 0:
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.bottle_width)
        self.surface.apply(obj, translucent=True)
        return obj

    def make_wrap(self, bottle):
        obj = new_cylinder(vertices=128)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
            bmesh.update_edit_mesh(obj.data)
        subdivide_edge_ring(obj, 16)
        z_max = self.z_neck - uniform(0.02, self.z_neck_offset) * (
            self.z_neck - self.z_waist
        )
        z_min = self.z_waist + uniform(0.02, self.z_waist_offset) * (
            self.z_neck - self.z_waist
        )
        radius = np.max(read_co(bottle)[:, 0]) + 2e-3
        obj.scale = radius, radius, (z_max - z_min) * self.z_length
        obj.location[-1] = z_min * self.z_length
        butil.apply_transform(obj, True)
        wrap_front_back(obj, self.wrap_surface, self.texture_shared)
        return obj

    def make_cap(self):
        obj = new_cylinder(vertices=128)
        obj.scale = [
            (self.x_cap + 0.1) * self.x_length,
            (self.x_cap + 0.1) * self.x_length,
            (self.z_cap + 0.01) * self.z_length,
        ]
        obj.location[-1] = (1 - self.z_cap) * self.z_length
        butil.apply_transform(obj, loc=True)
        subsurf(obj, 1, self.cap_subsurf)
        self.cap_surface.apply(obj)
        return obj
