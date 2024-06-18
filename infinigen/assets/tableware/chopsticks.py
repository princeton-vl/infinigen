# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.tableware.base import TablewareFactory
from infinigen.assets.utils.decorate import subsurf, write_co
from infinigen.core.util.random import log_uniform
from infinigen.assets.utils.object import join_objects, new_grid
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed

from infinigen.core.util import blender as butil


class ChopsticksFactory(TablewareFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.y_length = uniform(.01, .02)
            self.y_shrink = log_uniform(.2, .8)
            self.is_square = uniform(0, 1) < .5
            self.has_guard = uniform(0, 1) < .4
            self.x_guard = uniform(.4, .9)
            self.guard_depth = 0.
            self.pre_level = 2
            self.scale = log_uniform(.2, .4)

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_single()
        if uniform(0, 1) < .6:
            obj = self.make_parallel(obj)
        else:
            obj = self.make_crossed(obj)
        return obj

    def make_parallel(self, obj):
        distance = log_uniform(self.y_length, .04)
        if uniform(0, 1) < .5:
            other = deep_clone_obj(obj)
            obj.location[1] = distance
            obj.rotation_euler[-1] = uniform(0, np.pi / 8)
            other.location[1] = -distance
            other.rotation_euler[-1] = -uniform(0, np.pi / 8)
        else:
            obj.location[0] = -1
            butil.apply_transform(obj, loc=True)
            other = deep_clone_obj(obj)
            obj.location[1] = distance
            obj.rotation_euler[-1] = -uniform(0, np.pi / 8)
            other.location[1] = -distance
            other.rotation_euler[-1] = uniform(0, np.pi / 8)
        return join_objects([obj, other])

    def make_crossed(self, obj):
        other = deep_clone_obj(obj)
        other.location = uniform(-.1, .2), uniform(-.2, .2), self.y_length
        sign = np.sign(other.location[1])
        other.rotation_euler[-1] = -sign * log_uniform(np.pi / 8, np.pi / 4)
        return join_objects([obj, other])

    def make_single(self):
        n = int(1 / self.y_length)
        obj = new_grid(x_subdivisions=n - 1, y_subdivisions=1)
        butil.modify_mesh(obj, 'SOLIDIFY', thickness=self.y_length * 2)
        l = np.linspace(self.y_shrink, 1, n) * self.y_length
        x = np.concatenate([np.linspace(0, 1, n)] * 4)
        y = np.concatenate([-l, l, -l, l])
        z = np.concatenate([l, l, -l, -l])
        write_co(obj, np.stack([x, y, z], -1))
        subsurf(obj, 2, self.is_square)
        self.add_guard(obj, lambda nw, x: nw.compare('GREATER_THAN', x, self.x_guard))
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj
