# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import read_co, subsurf, write_attribute
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .pan import PanFactory


class PotFactory(PanFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.depth = log_uniform(0.6, 2.0)
            self.r_expand = 1
            self.r_mid = 1
            self.has_bar = uniform(0, 1) < 0.5
            self.has_handle = not self.has_handle
            self.has_guard = not self.has_bar
            self.bar_height = self.depth * uniform(0.75, 0.85)
            self.bar_radius = log_uniform(0.2, 0.3)
            self.bar_x = 1 + uniform(-self.bar_radius, self.bar_radius) * 0.05
            self.bar_inner_radius = log_uniform(0.2, 0.4) * self.bar_radius
            scale = log_uniform(0.6, 1.5)
            self.bar_scale = (
                log_uniform(0.6, 1.0) * scale,
                1 * scale,
                log_uniform(0.6, 1.2) * scale,
            )
            self.bar_taper = log_uniform(0.3, 0.8)
            self.bar_y_rotation = uniform(-np.pi / 6, 0)
            self.bar_x_offset = self.bar_radius * uniform(-0.1, 0.1)

            self.guard_type = "round"
            self.guard_depth = log_uniform(0.5, 1.0) * self.thickness
            self.scale = log_uniform(0.1, 0.15)

    def post_init(self):
        self.has_handle = not self.has_bar
        self.has_guard = not self.has_bar

        self.bar_x = 1 + uniform(-self.bar_radius, self.bar_radius) * 0.05
        self.bar_inner_radius = log_uniform(0.2, 0.4) * self.bar_radius
        self.bar_x_offset = self.bar_radius * uniform(-0.1, 0.1)

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        if self.has_bar:
            self.add_bar(obj)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        if self.has_bar:
            radius_ = (
                1
                + self.bar_x_offset
                + self.bar_radius
                + self.bar_inner_radius
                + self.thickness
            )
            obj = new_bbox(
                -radius_,
                radius_,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        elif self.has_handle:
            obj = new_bbox(
                -1 - self.thickness,
                1 + self.thickness + self.x_handle,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        else:
            obj = new_bbox(
                -1 - self.thickness,
                1 + self.thickness,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        obj.scale = (self.scale,) * 3
        butil.apply_transform(obj)
        return obj

    def add_bar(self, obj):
        bars = []
        for side in [-1, 1]:
            bpy.ops.mesh.primitive_torus_add(
                location=(side * (1 + self.bar_x_offset), 0, self.bar_height),
                major_radius=self.bar_radius,
                minor_radius=self.bar_inner_radius,
            )
            bar = bpy.context.active_object
            bar.scale = self.bar_scale
            butil.modify_mesh(
                bar,
                "SIMPLE_DEFORM",
                deform_method="TAPER",
                angle=self.bar_taper,
                deform_axis="X",
            )
            bar.rotation_euler = 0, self.bar_y_rotation, 0 if side == 1 else np.pi
            butil.apply_transform(bar)

            butil.modify_mesh(bar, "BOOLEAN", object=obj, operation="DIFFERENCE")
            butil.select_none()
            objs = butil.split_object(bar)
            i = np.argmax([np.max(read_co(o)[:, 0] * side) for o in objs])
            bar = objs[i]
            objs.remove(bar)
            butil.delete(objs)
            subsurf(bar, 1)
            write_attribute(bar, lambda nw: 1, "guard", "FACE")
            bars.append(bar)
        return join_objects([obj, *bars])
