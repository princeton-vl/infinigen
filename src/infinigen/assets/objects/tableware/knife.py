# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import subsurf, write_co
from infinigen.assets.utils.object import new_grid
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .base import TablewareFactory


class KnifeFactory(TablewareFactory):
    x_end = 0.5

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_length = log_uniform(0.4, 0.7)
            self.has_guard = uniform(0, 1) < 0.7
            if self.has_guard:
                self.y_length = log_uniform(0.1, 0.5)
                self.y_guard = self.y_length * log_uniform(0.2, 0.4)
            else:
                self.y_length = log_uniform(0.1, 0.2)
                self.y_guard = self.y_length * log_uniform(0.3, 0.5)
            self.x_guard = uniform(0, 0.2)
            self.has_tip = uniform(0, 1) < 0.7
            self.thickness = log_uniform(0.02, 0.03)
            y_off_rand = uniform(0, 1)
            self.y_offset = (
                0.2
                if y_off_rand < 1 / 8
                else 0.5
                if y_off_rand < 1 / 4
                else uniform(0.2, 0.6)
            )
            self.guard_type = "round" if uniform(0, 1) < 0.6 else "double"
            self.guard_depth = log_uniform(0.2, 1.0) * self.thickness
            self.scale = log_uniform(0.2, 0.3)

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = np.array(
            [
                self.x_end,
                uniform(0.5, 0.8) * self.x_end,
                uniform(0.3, 0.4) * self.x_end,
                1e-3,
                0,
                -1e-3,
                -2e-3,
                -self.x_end * self.x_length + 1e-3,
                -self.x_end * self.x_length,
            ]
        )
        y_anchors = np.array(
            [
                1e-3,
                self.y_length * log_uniform(0.75, 0.95),
                self.y_length,
                self.y_length,
                self.y_length,
                self.y_guard,
                self.y_guard,
                self.y_guard,
                self.y_guard,
            ]
        )
        if not self.has_guard:
            indices = [0, 1, 2, 4, 5, 7, 8]
            x_anchors = x_anchors[indices]
            y_anchors = y_anchors[indices]
        if self.has_tip:
            indices = [0] + list(range(len(x_anchors)))
            x_anchors = x_anchors[indices]
            x_anchors[0] += 1e-3
            y_anchors = y_anchors[indices]
            y_anchors[1] += 3e-3

        obj = new_grid(x_subdivisions=len(x_anchors) - 1, y_subdivisions=1)
        x = np.concatenate([x_anchors] * 2)
        y = np.concatenate([y_anchors, np.zeros_like(y_anchors)])
        y[0 :: len(y_anchors)] += self.y_offset * self.y_length
        if self.has_tip:
            y[1 :: len(y_anchors)] += self.y_offset * self.y_length
            y[2 :: len(y_anchors)] += self.y_offset * (self.y_length - y_anchors[2])
        else:
            y[1 :: len(y_anchors)] += self.y_offset * (self.y_length - y_anchors[1])
        z = np.concatenate([np.zeros_like(x_anchors)] * 2)
        write_co(obj, np.stack([x, y, z], -1))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        self.make_knife_tip(obj)
        subsurf(obj, 1)

        def selection(nw, x):
            return nw.compare(
                "LESS_THAN", x, -self.x_guard * self.x_length * self.x_end
            )

        if self.guard_type == "double":
            selection = self.make_double_sided(selection)
        self.add_guard(obj, selection)
        subsurf(obj, 1)
        subsurf(obj, 1, True)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def make_knife_tip(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            for e in bm.edges:
                u, v = e.verts
                x0, y0, z0 = u.co
                x1, y1, z1 = v.co
                if x0 >= 0 and x1 >= 0 and abs(x0 - x1) < 2e-4:
                    if (
                        y0 > self.y_offset * self.y_length
                        and y1 > self.y_offset * self.y_length
                    ):
                        bmesh.ops.pointmerge(
                            bm, verts=[u, v], merge_co=(u.co + v.co) / 2
                        )
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_loose(extend=False)
            bpy.ops.mesh.delete(type="EDGE")
