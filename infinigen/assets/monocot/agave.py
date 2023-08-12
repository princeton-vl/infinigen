# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import add_distance_to_boundary, join_objects, displace_vertices
from infinigen.assets.utils.draw import cut_plane, leaf
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class AgaveMonocotFactory(MonocotGrowthFactory):
    use_distance = True

    def __init__(self, factory_seed, coarse=False):
        super(AgaveMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(.0, .5)
            self.angle = uniform(np.pi / 9, np.pi / 6)
            self.z_drag = uniform(.05, .1)
            self.min_y_angle = uniform(np.pi * .1, np.pi * .15)
            self.max_y_angle = uniform(np.pi * .4, np.pi * .52)
            self.count = int(log_uniform(32, 64))
            self.scale_curve = [(0, uniform(.8, 1.)), (.5, 1), (1, uniform(.6, 1.))]

            self.bud_angle = uniform(np.pi / 8, np.pi / 4)
            self.cut_prob = 0 if uniform(0, 1) < .5 else uniform(.2, .4)

    @staticmethod
    def build_base_hue():
        return uniform(.12, .32)

    def build_leaf(self, face_size):
        x_anchors = 0, .2 * np.cos(self.bud_angle), uniform(1., 1.4), 1.5
        y_anchors = 0, .2 * np.sin(self.bud_angle), uniform(.1, .15), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        distance = add_distance_to_boundary(obj)

        lower = deep_clone_obj(obj)
        z_offset = -log_uniform(.08, .16)
        z_ratio = uniform(1.5, 2.5)
        displace_vertices(lower, lambda x, y, z: (0, 0, (1 - (1 - distance) ** z_ratio) * z_offset))
        obj = join_objects([lower, obj])
        butil.modify_mesh(obj, "WELD", merge_threshold=2e-4)

        if uniform(0, 1) < self.cut_prob:
            angle = uniform(-np.pi / 3, np.pi / 3)
            cut_center = np.array([uniform(1., 1.4), 0, 0])
            cut_normal = np.array([np.cos(angle), np.sin(angle), 0])
            obj, cut = cut_plane(obj, cut_center, cut_normal)
            obj = join_objects([obj, cut])
            with butil.ViewportMode(obj, 'EDIT'), butil.Suppress():
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.region_to_loop()
                bpy.ops.mesh.remove_doubles(threshold=1e-2)

        self.decorate_leaf(obj)
        tag_object(obj, 'agave')
        return obj
