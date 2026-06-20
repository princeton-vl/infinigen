# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import (
    read_co,
    remove_vertices,
    subsurf,
    write_attribute,
)
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform, weighted_sample

from .base import TablewareFactory


def _cup_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    AssetFactory.__init__(inst, seed, coarse)
    inst._init_tableware_base()
    inst.x_end = 0.25
    inst.is_short = uniform(0, 1) < 0.5
    if inst.is_short:
        inst.is_profile_straight = uniform(0, 1) < 0.2
        inst.x_lowest = log_uniform(0.6, 0.9)
        inst.depth = log_uniform(0.25, 0.5)
        inst.has_guard = uniform(0, 1) < 0.8
    else:
        inst.is_profile_straight = True
        inst.x_lowest = log_uniform(0.9, 1.0)
        inst.depth = log_uniform(0.5, 1.0)
        inst.has_guard = False
    if inst.is_profile_straight:
        inst.handle_location = uniform(0.45, 0.65)
    else:
        inst.handle_location = uniform(-0.1, 0.3)
    inst.handle_type = "shear" if uniform(0, 1) < 0.5 else "round"
    inst.handle_radius = inst.depth * uniform(0.2, 0.4)
    inst.handle_inner_radius = inst.handle_radius * log_uniform(0.2, 0.3)
    inst.handle_taper_x = uniform(0, 2)
    inst.handle_taper_y = uniform(0, 2)
    inst.x_lower_ratio = log_uniform(0.8, 1.0)
    inst.thickness = log_uniform(0.01, 0.04)
    inst.has_wrap = uniform() < 0.3
    inst.has_wrap = True
    inst.wrap_margin = uniform(0.1, 0.2)

    inst.wrap_surface = weighted_sample(material_assignments.graphicdesign)()()
    if inst.wrap_surface == text.Text:
        inst.wrap_surface = text.Text(inst.factory_seed, False)

    inst.has_inside = uniform(0, 1) < 0.5
    inst.scale = log_uniform(0.15, 0.3)


class CupParameters(LegacyBridgeParameters):
    pass


class CupFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CupParameters
    allow_transparent = True

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CupParameters:
        return legacy_init_to_parameters(
            CupParameters,
            CupFactory,
            seed,
            self.coarse,
            init_fn=_cup_legacy_init,
        )

    def apply_parameters(
        self, params: CupParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def create_asset(self, **params) -> bpy.types.Object:
        if self.is_profile_straight:
            x_anchors = 0, self.x_lowest * self.x_end, self.x_end
            z_anchors = 0, 0, self.depth
        else:
            x_anchors = (
                0,
                self.x_lowest * self.x_end,
                (self.x_lowest + self.x_lower_ratio * (1 - self.x_lowest)) * self.x_end,
                self.x_end,
            )
            z_anchors = 0, 0, self.depth * 0.5, self.depth
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [1])
        obj.scale = [1 / self.scale] * 3
        butil.apply_transform(obj, True)
        butil.modify_mesh(
            obj,
            "BEVEL",
            True,
            offset_type="PERCENT",
            width_pct=uniform(10, 50),
            segments=8,
        )
        if self.has_wrap:
            wrap = self.make_wrap(obj)
        else:
            wrap = None
        self.solidify_with_inside(obj, self.thickness)
        subsurf(obj, 2)
        handle_location = (
            x_anchors[-2] * (1 - self.handle_location)
            + x_anchors[-1] * self.handle_location,
            0,
            z_anchors[-2] * (1 - self.handle_location)
            + z_anchors[-1] * self.handle_location,
        )
        angle_low = np.arctan(
            (x_anchors[-1] - x_anchors[-2]) / (z_anchors[-1] - z_anchors[-2])
        )
        angle_height = np.arctan(
            (x_anchors[2] - x_anchors[1]) / (z_anchors[2] - z_anchors[1])
        )
        handle_angle = uniform(angle_low, angle_height + 1e-3)
        if self.has_guard:
            obj = self.add_handle(obj, handle_location, handle_angle)
        if self.has_wrap:
            butil.select_none()
            obj = join_objects([obj, wrap])
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def add_handle(self, obj, handle_location, handle_angle):
        bpy.ops.mesh.primitive_torus_add(
            location=handle_location,
            major_radius=self.handle_radius,
            minor_radius=self.handle_inner_radius,
        )
        handle = bpy.context.active_object
        handle.rotation_euler = np.pi / 2, handle_angle, 0
        butil.modify_mesh(
            handle,
            "SIMPLE_DEFORM",
            deform_method="TAPER",
            angle=self.handle_taper_x,
            deform_axis="X",
        )
        butil.modify_mesh(
            handle,
            "SIMPLE_DEFORM",
            deform_method="TAPER",
            angle=self.handle_taper_y,
            deform_axis="Y",
        )
        butil.modify_mesh(handle, "BOOLEAN", object=obj, operation="DIFFERENCE")
        butil.select_none()
        objs = butil.split_object(handle)
        i = np.argmax([np.max(read_co(o)[:, 0]) for o in objs])
        handle = objs[i]
        objs.remove(handle)
        butil.delete(objs)
        subsurf(handle, 1)
        write_attribute(handle, lambda nw: 1, "guard", "FACE")
        return join_objects([obj, handle])

    def make_wrap(self, obj):
        butil.select_none()
        obj = deep_clone_obj(obj)
        remove_vertices(
            obj,
            lambda x, y, z: (z / self.depth < self.wrap_margin)
            | (z / self.depth > 1 - self.wrap_margin + uniform(0.0, 0.1))
            | (np.abs(np.arctan2(y, x)) < np.pi * self.wrap_margin),
        )
        obj.scale = 1 + 1e-2, 1 + 1e-2, 1
        butil.apply_transform(obj)
        write_attribute(obj, lambda nw: 1, "text", "FACE")
        return obj

    def finalize_assets(self, assets):
        super().finalize_assets(assets)
        if self.has_wrap:
            for obj in assets if isinstance(assets, list) else [assets]:
                wrap_sides(obj, self.wrap_surface, "u", "v", "z", selection="text")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
