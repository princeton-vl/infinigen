# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.utils.decorate import read_co, write_attribute, write_co
from infinigen.assets.utils.object import new_cube, new_line
from infinigen.core.util.math import FixedSeed

from .straight import StraightStaircaseFactory


class LShapedStaircaseFactory(StraightStaircaseFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(LShapedStaircaseFactory, self).__init__(factory_seed, coarse, constants)
        with FixedSeed(self.factory_seed):
            self.m = int(self.n * uniform(0.4, 0.6))
            self.is_rail_circular = True

    def make_line(self, alpha):
        obj = new_line(self.n + 2)
        x = np.concatenate(
            [
                np.full(self.m + 2, alpha * self.step_width),
                -np.arange(self.n - self.m + 1) * self.step_length,
            ]
        )
        y = np.concatenate(
            [
                np.arange(self.m + 1) * self.step_length,
                [self.m * self.step_length + alpha * self.step_width],
                np.full(
                    self.n - self.m + 1,
                    self.m * self.step_length + alpha * self.step_width,
                ),
            ]
        )
        z = (
            np.concatenate(
                [np.arange(self.m + 1), [self.m], np.arange(self.m, self.n + 1)]
            )
            * self.step_height
        )
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def make_line_offset(self, alpha):
        obj = self.make_line(alpha)
        co = read_co(obj)
        co[self.m : self.m + 2] = co[self.m + 1 : self.m + 3]
        x, y, z = co.T
        x[self.m + 1] += min(self.step_length / 2, alpha * self.step_width)
        x[self.m + 2 :] -= self.step_length / 2
        y[: self.m] += self.step_length / 2
        z += self.step_height
        z[[self.m, self.m + 1, -1]] -= self.step_height
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def make_post_locs(self, alpha):
        temp = self.make_line_offset(alpha)
        cos = read_co(temp)
        butil.delete(temp)
        chunks = self.split(self.m - 1)
        chunks_ = self.split(self.m + 1, self.n + 2)
        indices = (
            list(c[0] for c in chunks)
            + [self.m - 1, self.m, self.m + 1]
            + list(c[0] for c in chunks_)
            + [self.n + 1, self.n + 2]
        )
        return cos[indices]

    def make_vertical_post_locs(self, alpha):
        temp = self.make_line_offset(alpha)
        cos = read_co(temp)
        butil.delete(temp)
        chunks = self.split(self.m - 1)
        chunks_ = self.split(self.m + 1, self.n + 2)
        indices = sum(list(c[1:].tolist() for c in chunks), [])
        indices_ = sum(list(c[1:].tolist() for c in chunks_), [])
        mid_cos = []
        mid = [self.m - 1, self.m]
        for m in mid:
            for r in np.linspace(
                0, 1, self.post_k + 1 if m >= self.m else self.post_k + 2
            )[1:-1]:
                mid_cos.append(r * cos[m] + (1 - r) * cos[m + 1])
        return np.concatenate([cos[indices], np.stack(mid_cos), cos[indices_]], 0)

    def make_steps(self):
        objs = super(LShapedStaircaseFactory, self).make_steps()
        for obj in objs[self.m :]:
            obj.rotation_euler[-1] = np.pi / 2
            obj.location = self.m * self.step_length, self.m * self.step_length, 0
            butil.apply_transform(obj, loc=True)
        lowest = np.min(read_co(objs[self.m]).T[-1])
        platform = new_cube(location=(1, 1, 1))
        butil.apply_transform(platform, loc=True)
        platform.location = 0, self.step_length * self.m, lowest
        platform.scale = (
            self.step_width / 2,
            self.step_width / 2,
            (self.step_height * self.m - lowest) / 2,
        )
        butil.apply_transform(platform, loc=True)
        write_attribute(platform, 1, "steps", "FACE")
        return objs + [platform]

    def make_treads(self):
        objs = super(LShapedStaircaseFactory, self).make_treads()
        for obj in objs[self.m :]:
            obj.rotation_euler[-1] = np.pi / 2
            obj.location = self.m * self.step_length, self.m * self.step_length, 0
            butil.apply_transform(obj, loc=True)
        platform = new_cube(location=(1, 1, 1))
        butil.apply_transform(platform, loc=True)
        platform.location = 0, self.step_length * self.m, self.step_height * self.m
        platform.scale = self.step_width / 2, self.step_width / 2, self.tread_height / 2
        butil.apply_transform(platform, loc=True)
        write_attribute(platform, 1, "treads", "FACE")
        return objs + [platform]

    def make_inner_sides(self):
        objs = super(LShapedStaircaseFactory, self).make_inner_sides()
        for obj in objs[self.m :]:
            obj.rotation_euler[-1] = np.pi / 2
            obj.location = self.m * self.step_length, self.m * self.step_length, 0
            butil.apply_transform(obj, loc=True)

        top_cutter = new_cube(location=(0, 0, 1))
        butil.apply_transform(top_cutter, loc=True)
        top_cutter.scale = [100] * 3
        top_cutter.location[-1] = self.m * self.step_height + self.tread_height
        for obj in objs[: self.m]:
            butil.modify_mesh(obj, "BOOLEAN", object=top_cutter, operation="DIFFERENCE")
        butil.delete(top_cutter)
        return objs

    def make_outer_sides(self):
        objs = self.make_inner_sides()
        for obj in objs[: self.m]:
            obj.location[0] += self.step_width
            butil.apply_transform(obj, loc=True)
        for obj in objs[self.m :]:
            obj.location[1] += self.step_width
            butil.apply_transform(obj, loc=True)
        platform = new_line(2)
        x = self.step_width, self.step_width, 0
        y = (
            self.m * self.step_length,
            self.m * self.step_length + self.step_width,
            self.m * self.step_length + self.step_width,
        )
        z = [self.m * self.step_height] * 3
        write_co(platform, np.stack([x, y, z], -1))
        butil.select_none()
        with butil.ViewportMode(platform, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, -self.side_height)}
            )
        butil.modify_mesh(platform, "SOLIDIFY", thickness=self.side_thickness)
        write_attribute(platform, 1, "sides", "FACE")
        return objs + [platform]

    @property
    def upper(self):
        return np.pi
