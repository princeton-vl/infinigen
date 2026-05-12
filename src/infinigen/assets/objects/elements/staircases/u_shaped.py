# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix constants

import bpy
import numpy as np

import infinigen.core.util.blender as butil
from infinigen.assets.utils.decorate import read_co, write_attribute, write_co
from infinigen.assets.utils.object import new_cube, new_line
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .straight import StraightStaircaseFactory


class UShapedStaircaseFactory(StraightStaircaseFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(UShapedStaircaseFactory, self).__init__(factory_seed, coarse, constants)
        with FixedSeed(self.factory_seed):
            self.m = self.n // 2
            self.is_rail_circular = True

    def build_size_config(self):
        self.n = int(np.random.randint(13, 21) / 2) * 2
        self.step_height = self.constants.wall_height / self.n
        self.step_width = log_uniform(0.9, 1.5)
        self.step_length = self.step_height * log_uniform(1, 1.2)

    def make_line(self, alpha):
        obj = new_line(self.n + 4)
        x = np.concatenate(
            [
                np.full(self.m + 2, alpha * self.step_width),
                [0],
                np.full(self.m + 2, -alpha * self.step_width),
            ]
        )
        y = np.concatenate(
            [
                np.arange(self.m + 1) * self.step_length,
                [self.m * self.step_length + alpha * self.step_width] * 3,
                np.arange(self.m, -1, -1) * self.step_length,
            ]
        )
        z = (
            np.concatenate(
                [np.arange(self.m + 1), [self.m] * 3, np.arange(self.m, self.n + 1)]
            )
            * self.step_height
        )
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def make_line_offset(self, alpha):
        obj = self.make_line(alpha)
        co = read_co(obj)
        co[self.m : self.m + 4] = co[self.m + 1 : self.m + 5]
        x, y, z = co.T
        y[: self.m] += self.step_length / 2
        y[self.m + 3] += min(self.step_length / 2, alpha * self.step_width)
        y[self.m + 4 :] -= self.step_length / 2
        z += self.step_height
        z[[self.m, self.m + 1, self.m + 2, self.m + 3, -1]] -= self.step_height
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def make_post_locs(self, alpha):
        temp = self.make_line_offset(alpha)
        cos = read_co(temp)
        butil.delete(temp)
        chunks = self.split(self.m - 1)
        chunks_ = self.split(self.m + 3, self.n + 4)
        mid = [self.m - 1, self.m, self.m + 1, self.m + 2, self.m + 3]
        indices = (
            list(c[0] for c in chunks)
            + mid
            + list(c[0] for c in chunks_)
            + [self.n + 3, self.n + 4]
        )
        return cos[indices]

    def make_vertical_post_locs(self, alpha):
        temp = self.make_line_offset(alpha)
        cos = read_co(temp)
        butil.delete(temp)
        chunks = self.split(self.m - 1)
        chunks_ = np.array_split(
            np.arange(self.m + 3, self.n + 4), np.ceil((self.n - self.m) / self.post_k)
        )
        indices = sum(list(c[1:].tolist() for c in chunks + chunks_), [])
        indices_ = sum(list(c[1:].tolist() for c in chunks_), [])
        mid_cos = []
        mid = [self.m - 1, self.m, self.m + 1, self.m + 2]
        for m in mid:
            for r in np.linspace(
                0, 1, self.post_k + 1 if m >= self.m else self.post_k + 2
            )[1:-1]:
                mid_cos.append(r * cos[m] + (1 - r) * cos[m + 1])
        return np.concatenate([cos[indices], np.stack(mid_cos), cos[indices_]], 0)

    def make_steps(self):
        objs = super(UShapedStaircaseFactory, self).make_steps()
        for obj in objs[self.m :]:
            obj.rotation_euler[-1] = np.pi
            obj.location = 0, 2 * self.m * self.step_length, 0
            butil.apply_transform(obj, loc=True)
        lowest = np.min(read_co(objs[self.m]).T[-1])
        platform = new_cube(location=(0, 1, 1))
        butil.apply_transform(platform, loc=True)
        platform.location = 0, self.step_length * self.m, lowest
        platform.scale = (
            self.step_width,
            self.step_width / 2,
            (self.step_height * self.m - lowest) / 2,
        )
        butil.apply_transform(platform, loc=True)
        write_attribute(platform, 1, "steps", "FACE")
        return objs + [platform]

    def make_treads(self):
        objs = super(UShapedStaircaseFactory, self).make_treads()
        for obj in objs[self.m :]:
            obj.rotation_euler[-1] = np.pi
            obj.location = 0, 2 * self.m * self.step_length, 0
            butil.apply_transform(obj, loc=True)
        platform = new_cube(location=(0, 1, 1))
        butil.apply_transform(platform, loc=True)
        platform.location = 0, self.step_length * self.m, self.step_height * self.m
        platform.scale = self.step_width, self.step_width / 2, self.tread_height / 2
        butil.apply_transform(platform, loc=True)
        write_attribute(platform, 1, "treads", "FACE")
        return objs + [platform]

    def make_inner_sides(self):
        objs = super(UShapedStaircaseFactory, self).make_inner_sides()
        for obj in objs[self.m :]:
            obj.rotation_euler[-1] = np.pi
            obj.location = 0, 2 * self.m * self.step_length, 0
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
            obj.location[0] -= self.step_width
            butil.apply_transform(obj, loc=True)
        platform = new_line(4)
        x = self.step_width, self.step_width, 0, -self.step_width, -self.step_width
        mid = self.m * self.step_length + self.step_width
        y = self.m * self.step_length, mid, mid, mid, self.m * self.step_length
        z = [self.m * self.step_height] * 5
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
        return -np.pi / 2
