# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import ClassVar

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring, subsurf
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects, new_cylinder
from infinigen.assets.utils.uv import wrap_front_back
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


class BottleParameters(LegacyBridgeParameters):
    pass


def _bottle_legacy_init(inst: "BottleFactory", seed: int, coarse: bool) -> None:
    inst.z_length = uniform(0.15, 0.25)
    inst.x_length = inst.z_length * uniform(0.15, 0.25)
    inst.x_cap = uniform(0.3, 0.35)
    inst.bottle_type = np.random.choice(
        ["beer", "bordeaux", "champagne", "coke", "vintage"]
    )
    inst.bottle_width = uniform(0.002, 0.005)
    inst.z_waist = 0
    match inst.bottle_type:
        case "beer":
            inst.z_neck = uniform(0.5, 0.6)
            inst.z_cap = uniform(0.05, 0.08)
            neck_size = uniform(0.06, 0.1)
            neck_ratio = uniform(0.4, 0.5)
            inst.x_anchors = [
                0,
                1,
                1,
                (neck_ratio + 1) / 2 + (1 - neck_ratio) / 2 * inst.x_cap,
                neck_ratio + (1 - neck_ratio) * inst.x_cap,
                inst.x_cap,
                inst.x_cap,
                0,
            ]
            inst.z_anchors = [
                0,
                0,
                inst.z_neck,
                inst.z_neck + uniform(0.6, 0.7) * neck_size,
                inst.z_neck + neck_size,
                1 - inst.z_cap,
                1,
                1,
            ]
            inst.is_vector = [0, 1, 1, 0, 1, 1, 1, 0]
        case "bordeaux":
            inst.z_neck = uniform(0.6, 0.7)
            inst.z_cap = uniform(0.1, 0.15)
            neck_size = uniform(0.1, 0.15)
            inst.x_anchors = (
                0,
                1,
                1,
                (1 + inst.x_cap) / 2,
                inst.x_cap,
                inst.x_cap,
                0,
            )
            inst.z_anchors = [
                0,
                0,
                inst.z_neck,
                inst.z_neck + uniform(0.6, 0.7) * neck_size,
                inst.z_neck + neck_size,
                1,
                1,
            ]
            inst.is_vector = [0, 1, 1, 0, 1, 1, 0]
        case "champagne":
            inst.z_neck = uniform(0.4, 0.5)
            inst.z_cap = uniform(0.05, 0.08)
            inst.x_anchors = [
                0,
                1,
                1,
                1,
                (1 + inst.x_cap) / 2,
                inst.x_cap,
                inst.x_cap,
                0,
            ]
            inst.z_anchors = [
                0,
                0,
                inst.z_neck,
                inst.z_neck + uniform(0.08, 0.1),
                inst.z_neck + uniform(0.15, 0.18),
                1 - inst.z_cap,
                1,
                1,
            ]
            inst.is_vector = [0, 1, 1, 0, 0, 1, 1, 0]
        case "coke":
            inst.z_waist = uniform(0.4, 0.5)
            inst.z_neck = inst.z_waist + uniform(0.2, 0.25)
            inst.z_cap = uniform(0.05, 0.08)
            inst.x_anchors = [
                0,
                uniform(0.85, 0.95),
                1,
                uniform(0.85, 0.95),
                1,
                1,
                inst.x_cap,
                inst.x_cap,
                0,
            ]
            inst.z_anchors = [
                0,
                0,
                uniform(0.08, 0.12),
                uniform(0.18, 0.25),
                inst.z_waist,
                inst.z_neck,
                1 - inst.z_cap,
                1,
                1,
            ]
            inst.is_vector = [0, 1, 0, 0, 1, 1, 1, 1, 0]
        case "vintage":
            inst.z_waist = uniform(0.1, 0.15)
            inst.z_neck = uniform(0.7, 0.75)
            inst.z_cap = uniform(0.0, 0.08)
            x_lower = uniform(0.85, 0.95)
            inst.x_anchors = [
                0,
                x_lower,
                (x_lower + 1) / 2,
                1,
                1,
                (inst.x_cap + 1) / 2,
                inst.x_cap,
                inst.x_cap,
                0,
            ]
            inst.z_anchors = [
                0,
                0,
                inst.z_waist - uniform(0.1, 0.15),
                inst.z_waist,
                inst.z_neck,
                inst.z_neck + uniform(0.1, 0.2),
                1 - inst.z_cap,
                1,
                1,
            ]
            inst.is_vector = [0, 1, 0, 1, 1, 0, 1, 1, 0]

    inst.surface = weighted_sample(material_assignments.plastics)()()
    inst.wrap_surface = text.Text()()
    if inst.wrap_surface == text.Text:
        inst.wrap_surface = text.Text(False)
    inst.cap_surface = weighted_sample(material_assignments.metals)()()
    inst.texture_shared = uniform() < 0.2
    inst.cap_subsurf = uniform() < 0.5
    inst.wrap_z_max = inst.z_neck - uniform(0.02, inst.z_neck_offset) * (
        inst.z_neck - inst.z_waist
    )
    inst.wrap_z_min = inst.z_waist + uniform(0.02, inst.z_waist_offset) * (
        inst.z_neck - inst.z_waist
    )


class BottleFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BottleParameters
    z_neck_offset = 0.05
    z_waist_offset = 0.15

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BottleParameters:
        return legacy_init_to_parameters(
            BottleParameters,
            BottleFactory,
            seed,
            self.coarse,
            init_fn=_bottle_legacy_init,
        )

    def apply_parameters(
        self, params: BottleParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def create_asset(self, **params) -> bpy.types.Object:
        bottle = self.make_bottle()
        wrap = self.make_wrap(bottle)
        cap = self.make_cap()
        obj = join_objects([bottle, wrap, cap])

        return obj

    def finalize_assets(self, assets):
        pass
        # if self.scratch:
        #     self.scratch.apply(assets)
        # if self.edge_wear:
        #     self.edge_wear.apply(assets)

    def make_bottle(self):
        x_anchors = np.array(self.x_anchors) * self.x_length
        z_anchors = np.array(self.z_anchors) * self.z_length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors, np.nonzero(self.is_vector)[0])
        subsurf(obj, 1)
        if self.bottle_width > 0:
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.bottle_width)

        surface.assign_material(obj, self.surface)

        return obj

    def make_wrap(self, bottle):
        obj = new_cylinder(vertices=128)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
            bmesh.update_edit_mesh(obj.data)
        subdivide_edge_ring(obj, 16)
        z_max = (
            self.wrap_z_max
            if self._use_fixed_spawn_draws
            else self.z_neck
            - uniform(0.02, self.z_neck_offset) * (self.z_neck - self.z_waist)
        )
        z_min = (
            self.wrap_z_min
            if self._use_fixed_spawn_draws
            else self.z_waist
            + uniform(0.02, self.z_waist_offset) * (self.z_neck - self.z_waist)
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
        surface.assign_material(obj, self.cap_surface)
        return obj
