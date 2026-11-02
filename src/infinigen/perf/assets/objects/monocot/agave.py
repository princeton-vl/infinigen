# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen agave (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/monocot/agave.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.objects.monocot.agave import (
    AgaveMonocotFactory as _DefaultAgaveMonocotFactory,
)
from infinigen.assets.utils.decorate import (
    displace_vertices,
    distance2boundary,
    read_co,
    read_loop_totals,
    read_loop_vertices,
)
from infinigen.assets.utils.draw import cut_plane, leaf
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.tagging import tag_object
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform
from infinigen.perf.assets.utils.draw import monocot_legacy

from .leaf_direct import mesh_from_polygons


class AgaveMonocotFactory(_DefaultAgaveMonocotFactory):
    use_distance = True

    @staticmethod
    def double_leaf(obj, distance, z_offset, z_ratio, weld=2e-4):
        """Two-sheet agave leaf built on arrays, sharing the lower-sheet vertices that
        move at most `weld` (the outline) in place of the default's join and weld."""
        co = read_co(obj)
        loop_verts = read_loop_vertices(obj)
        loop_totals = read_loop_totals(obj)
        lower = co.copy()
        lower[:, 2] += (1 - (1 - distance) ** z_ratio) * z_offset
        merged = np.linalg.norm(lower - co, axis=-1) <= weld
        keep = ~merged
        index = np.arange(len(co))
        index[keep] = len(co) + np.arange(int(keep.sum()))
        verts = np.concatenate([co, lower[keep]])
        mesh = mesh_from_polygons(
            verts,
            np.concatenate([loop_verts, index[loop_verts]]),
            np.concatenate([loop_totals, loop_totals]),
            obj.data.name,
        )
        old = obj.data
        obj.data = mesh
        bpy.data.meshes.remove(old)
        surface.write_attr_data(
            obj, "distance", np.concatenate([distance, distance[keep]])
        )
        return obj

    def build_leaf(self, face_size):
        x_anchors = 0, 0.2 * np.cos(self.bud_angle), uniform(1.0, 1.4), 1.5
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.1, 0.15), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        distance = distance2boundary(obj)

        z_offset = -log_uniform(0.08, 0.16)
        z_ratio = uniform(1.5, 2.5)
        if monocot_legacy():
            lower = deep_clone_obj(obj)
            displace_vertices(
                lower,
                lambda x, y, z: (0, 0, (1 - (1 - distance) ** z_ratio) * z_offset),
            )
            obj = join_objects([lower, obj])
            butil.modify_mesh(obj, "WELD", merge_threshold=2e-4)
        else:
            obj = AgaveMonocotFactory.double_leaf(obj, distance, z_offset, z_ratio)

        if uniform(0, 1) < self.cut_prob:
            angle = uniform(-np.pi / 3, np.pi / 3)
            cut_center = np.array([uniform(1.0, 1.4), 0, 0])
            cut_normal = np.array([np.cos(angle), np.sin(angle), 0])
            obj, cut = cut_plane(obj, cut_center, cut_normal)
            obj = join_objects([obj, cut])
            with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.region_to_loop()
                bpy.ops.mesh.remove_doubles(threshold=1e-2)

        self.decorate_leaf(obj)
        tag_object(obj, "agave")
        return obj
