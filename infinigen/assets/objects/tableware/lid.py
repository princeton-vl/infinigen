# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.decorate import read_center, subsurf, write_co
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects, new_cylinder, new_line
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


def _lid_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    inst.x_length = uniform(0.08, 0.15)
    inst.z_height = inst.x_length * uniform(0, 0.5)
    inst.thickness = uniform(0.003, 0.005)
    inst.is_glass = uniform() < 0.5
    inst.hardware_type = None
    inst.rim_height = uniform(1, 2) * inst.thickness
    inst.handle_type = np.random.choice(["handle", "knob"])
    if inst.handle_type == "knob":
        inst.handle_height = inst.x_length * uniform(0.1, 0.15)
    else:
        inst.handle_height = inst.x_length * uniform(0.2, 0.25)
    inst.handle_radius = inst.x_length * uniform(0.15, 0.25)
    inst.handle_width = inst.x_length * uniform(0.25, 0.3)
    inst.handle_subsurf_level = np.random.randint(0, 3)

    if inst.is_glass:
        surface_gen_class = weighted_sample(
            material_assignments.appliance_front_maybeglass
        )
    else:
        surface_gen_class = weighted_sample(material_assignments.decorative_hard)

    inst.surface_material_gen = surface_gen_class()

    rim_surface_gen_class = weighted_sample(material_assignments.metals)
    inst.rim_surface_material_gen = rim_surface_gen_class()

    handle_surface_gen_class = weighted_sample(material_assignments.decorative_hard)
    inst.handle_surface_material_gen = handle_surface_gen_class()

    scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
    scratch, edge_wear = material_assignments.wear_tear

    inst.scratch = None if uniform() > scratch_prob else scratch()
    inst.edge_wear = None if uniform() > edge_wear_prob else edge_wear()


class LidParameters(LegacyBridgeParameters):
    pass


class LidFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LidParameters

    def __init__(self, factory_seed, coarse=False):
        super(LidFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LidParameters:
        return legacy_init_to_parameters(
            LidParameters,
            LidFactory,
            seed,
            self.coarse,
            init_fn=_lid_legacy_init,
        )

    def apply_parameters(
        self, params: LidParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        self.rim_surface = self.rim_surface_material_gen()
        self.handle_surface = self.handle_surface_material_gen()

        x_anchors = 0, 0.01, self.x_length / 2, self.x_length
        z_anchors = self.z_height, self.z_height, self.z_height * uniform(0.7, 0.8), 0
        obj = spin((x_anchors, 0, z_anchors))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)

        surface.assign_material(obj, self.surface)
        parts = [obj]
        if self.is_glass:
            parts.append(self.add_rim())
        match self.handle_type:
            case "handle":
                parts.append(self.add_handle(obj))
            case _:
                parts.append(self.add_knob())
        obj = join_objects(parts)
        return obj

    def add_rim(self):
        butil.select_none()
        bpy.ops.mesh.primitive_torus_add(
            major_radius=self.x_length,
            minor_radius=self.thickness / 2,
            major_segments=128,
        )
        obj = bpy.context.active_object
        obj.scale[-1] = self.rim_height / self.thickness
        butil.apply_transform(obj)
        surface.assign_material(obj, self.rim_surface)
        return obj

    def add_handle(self, obj):
        center = read_center(obj)
        i = np.argmin(
            np.abs(center[:, :2] - np.array([self.handle_width, 0])[np.newaxis, :]).sum(
                -1
            )
        )
        z_offset = center[i, -1]
        obj = new_line(3)
        write_co(
            obj,
            np.array(
                [
                    [-self.handle_width, 0, 0],
                    [-self.handle_width, 0, self.handle_height],
                    [self.handle_width, 0, self.handle_height],
                    [self.handle_width, 0, 0],
                ]
            ),
        )
        subsurf(obj, self.handle_subsurf_level)
        butil.select_none()
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, self.thickness * 2, 0)}
            )
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)
        obj.location = 0, -self.thickness, z_offset
        butil.apply_transform(obj, True)
        surface.assign_material(obj, self.handle_surface)
        return obj

    def add_knob(self):
        obj = new_cylinder()
        obj.scale = *([self.thickness * uniform(1, 2)] * 2), self.handle_height
        obj.location[-1] = self.z_height
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)
        top = new_cylinder()
        top.scale = (
            self.handle_radius,
            self.handle_radius,
            self.thickness * uniform(1, 2),
        )
        top.location[-1] = self.z_height + self.handle_height
        butil.apply_transform(top, True)
        butil.modify_mesh(top, "BEVEL", width=self.thickness / 2, segments=4)
        obj = join_objects([obj, top])
        surface.assign_material(obj, self.handle_surface)
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
