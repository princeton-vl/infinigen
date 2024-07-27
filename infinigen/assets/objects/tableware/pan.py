# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix cutter bug

import bmesh
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import subsurf
from infinigen.assets.utils.object import (
    join_objects,
    new_base_circle,
    new_base_cylinder,
    origin2lowest,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .base import TablewareFactory


class PanFactory(TablewareFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.r_expand = 1 if uniform(0, 1) < 0.2 else log_uniform(1.0, 1.2)
            self.depth = log_uniform(0.3, 0.8)
            if self.r_expand == 1:
                self.r_mid = log_uniform(1.0, 1.3)
            else:
                self.r_mid = 1 + (self.r_expand - 1) * (
                    uniform(0.5, 0.85) if uniform(0, 1) < 0.5 else 0.5
                )
            self.has_handle = True
            self.has_handle_hole = uniform() < 0.6
            self.pre_level = 2
            self.x_handle = log_uniform(1.2, 2.0)
            self.z_handle = self.x_handle * uniform(0, 0.2)
            self.z_handle_mid = uniform(0.6, 0.8) * self.z_handle
            self.s_handle = log_uniform(0.8, 1.2)
            self.thickness = log_uniform(0.04, 0.06)
            self.has_guard = uniform(0, 1) < 0.8
            self.x_guard = self.r_expand + uniform(0, 0.2) * self.x_handle
            self.guard_type = "round"
            self.guard_depth = log_uniform(1.0, 2.0) * self.thickness
            material_assignments = AssetList["PanFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.inside_surface = material_assignments["inside"].assign_material()
            if self.surface == self.inside_surface:
                self.has_inside = uniform(0, 1) < 0.5
            else:
                self.has_inside = True
            self.metal_color = None
            self.scale = log_uniform(0.1, 0.15)
            self.scratch = self.edge_wear = None

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        origin2lowest(obj, vertical=True)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def make_base(self):
        n = 4 * int(log_uniform(4, 8))
        base = new_base_circle(vertices=n)
        middle = new_base_circle(
            vertices=n,
        )
        middle.location[-1] = self.depth / 2
        middle.scale = [self.r_mid] * 3
        upper = new_base_circle(vertices=n)
        upper.location[-1] = self.depth
        upper.scale = [self.r_expand] * 3
        butil.apply_transform(upper, loc=True)
        obj = join_objects([base, middle, upper])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.bridge_edge_loops()
            bm = bmesh.from_edit_mesh(obj.data)
            for v in bm.verts:
                v.select_set(np.abs(v.co[-1]) < 1e-3)
            bm.select_flush(False)
            bmesh.update_edit_mesh(obj.data)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.fill_grid(
                use_interp_simple=True, offset=np.random.randint(n // 4)
            )
            bpy.ops.mesh.quads_convert_to_tris(
                quad_method="BEAUTY", ngon_method="BEAUTY"
            )
        obj.rotation_euler[-1] = np.pi / n
        butil.apply_transform(obj)
        if self.has_handle:
            self.add_handle(obj)
        self.solidify_with_inside(obj, self.thickness)

        def selection(nw, x):
            return nw.compare("GREATER_THAN", x, self.x_guard)

        self.add_guard(obj, selection)
        subsurf(obj, 1, True)
        subsurf(obj, 3)
        if self.has_handle_hole:
            self.add_handle_hole(obj)
        return obj

    def add_handle(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bm = bmesh.from_edit_mesh(obj.data)
            bm.edges.ensure_lookup_table()
            m = []
            for e in bm.edges:
                u, v = e.verts
                m.append(u.co[0] + v.co[0] + u.co[2] + v.co[2])
            ri = np.argmax(m)
            for e in bm.edges:
                e.select_set(e.index == ri)
            bm.select_flush(False)
            bmesh.update_edit_mesh(obj.data)

            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (self.x_handle * 0.5, 0, self.z_handle_mid)
                }
            )
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (
                        self.x_handle * 0.5,
                        0,
                        (self.z_handle - self.z_handle_mid),
                    )
                }
            )
            bpy.ops.transform.resize(value=[self.s_handle] * 3)
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (1e-3, 0, 0)}
            )

    def add_handle_hole(self, obj):
        cutter = new_base_cylinder()
        cutter.scale = *([uniform(0.06, 0.1)] * 2), 1
        cutter.location[0] = self.r_expand + uniform(0.8, 0.9) * self.x_handle
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        butil.delete(cutter)
