# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials import text
from infinigen.assets.objects.tableware.base import TablewareFactory
from infinigen.assets.utils.decorate import (
    read_co,
    remove_vertices,
    subsurf,
    write_attribute,
)
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class CupFactory(TablewareFactory):
    allow_transparent = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_end = 0.25
            self.is_short = uniform(0, 1) < 0.5
            if self.is_short:
                self.is_profile_straight = uniform(0, 1) < 0.2
                self.x_lowest = log_uniform(0.6, 0.9)
                self.depth = log_uniform(0.25, 0.5)
                self.has_guard = uniform(0, 1) < 0.8
            else:
                self.is_profile_straight = True
                self.x_lowest = log_uniform(0.9, 1.0)
                self.depth = log_uniform(0.5, 1.0)
                self.has_guard = False
            if self.is_profile_straight:
                self.handle_location = uniform(0.45, 0.65)
            else:
                self.handle_location = uniform(-0.1, 0.3)
            self.handle_type = "shear" if uniform(0, 1) < 0.5 else "round"
            self.handle_radius = self.depth * uniform(0.2, 0.4)
            self.handle_inner_radius = self.handle_radius * log_uniform(0.2, 0.3)
            self.handle_taper_x = uniform(0, 2)
            self.handle_taper_y = uniform(0, 2)
            self.x_lower_ratio = log_uniform(0.8, 1.0)
            self.thickness = log_uniform(0.01, 0.04)
            self.has_wrap = uniform() < 0.3
            self.has_wrap = True
            self.wrap_margin = uniform(0.1, 0.2)

            material_assignments = AssetList["CupFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.wrap_surface = material_assignments["wrap_surface"].assign_material()
            if self.wrap_surface == text.Text:
                self.wrap_surface = text.Text(self.factory_seed, False)
            self.scratch = self.edge_wear = None

            self.has_inside = uniform(0, 1) < 0.5
            self.scale = log_uniform(0.15, 0.3)

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
