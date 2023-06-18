# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: Jun 16, 2023

import bpy
import numpy as np
from numpy.random import uniform

import util.blender as butil
from assets.creatures.animation.driver_repeated import repeated_driver
from assets.monocot.growth import MonocotGrowthFactory
from assets.utils.draw import bezier_curve, leaf
from assets.utils.decorate import assign_material, join_objects
from assets.utils.misc import log_uniform
from assets.utils.object import origin2leftmost
from nodes.node_wrangler import NodeWrangler
from placement.detail import remesh_with_attrs
from util.math import FixedSeed


class KelpMonocotFactory(MonocotGrowthFactory):
    max_leaf_length = 1.2
    align_angle = uniform(np.pi / 24, np.pi / 12)

    def __init__(self, factory_seed, coarse=False):
        super(KelpMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = 10.
            self.angle = uniform(np.pi / 6, np.pi / 4)
            self.z_drag = uniform(.0, .2)
            self.min_y_angle = uniform(0, np.pi * .1)
            self.max_y_angle = self.min_y_angle
            self.bend_angle = uniform(0, np.pi / 6)
            self.twist_angle = uniform(0, np.pi / 6)
            self.count = 512
            self.leaf_prob = uniform(.6, .7)
            self.align_angle = uniform(np.pi / 30, np.pi / 15)
            self.radius = .02
            self.align_factor = self.make_align_factor()
            self.align_direction = self.make_align_direction()
            flow_angle = uniform(0, np.pi * 2)
            self.align_direction = np.cos(flow_angle), np.sin(flow_angle), uniform(-.2, .2)
            self.anim_freq = 1 / log_uniform(100, 200)
            self.anim_offset = uniform(0, 1)
            self.anim_seed = np.random.randint(1e5)

    def make_align_factor(self):
        def align_factor(nw: NodeWrangler):
            rand = nw.uniform(.7, .95)
            driver = rand.inputs[2].driver_add('default_value').driver
            driver.expression = repeated_driver(.7, .85, self.anim_freq, self.anim_offset, self.anim_seed)
            return nw.scalar_multiply(nw.bernoulli(.9), rand)

        return align_factor

    def make_align_direction(self):
        def align_direction(nw: NodeWrangler):
            direction = nw.combine(1, 0, 0)
            driver = direction.inputs[2].driver_add('default_value').driver
            driver.expression = repeated_driver(-.5, -.1, self.anim_freq, self.anim_offset, self.anim_seed)
            return direction

        return align_direction

    @staticmethod
    def build_base_hue():
        return uniform(.05, .25)

    def build_instance(self, i, face_size):
        x_anchors = np.array([0, -.02, -.04])
        y_anchors = np.array([0, uniform(.01, .02), 0])
        curves = []
        for angle in np.linspace(0, np.pi * 2, 6):
            anchors = [x_anchors, np.cos(angle) * y_anchors, np.sin(angle) * y_anchors]
            curves.append(bezier_curve(anchors))
        bud = butil.join_objects(curves)
        bud.location[0] += .02
        with butil.ViewportMode(bud, 'EDIT'):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.convex_hull()
        remesh_with_attrs(bud, face_size)

        x_anchors = 0, uniform(.35, .65), uniform(.8, 1.2)
        y_anchors = 0, uniform(.06, .08), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        obj = join_objects([obj, bud])
        self.decorate_leaf(obj, uniform(-2, 2), uniform(-np.pi / 4, np.pi / 4), uniform(-np.pi / 4, np.pi / 4))
        origin2leftmost(obj)
        return obj

    def create_asset(self, **params):
        obj = self.create_raw(apply=False)
        obj, mod = butil.modify_mesh(obj, 'SIMPLE_DEFORM', False, deform_method='TWIST', deform_axis='Z',
                                     return_mod=True)
        twist_driver = mod.driver_add('angle').driver
        extra_twist_angle = uniform(0, np.pi / 60)
        twist_driver.expression = repeated_driver(self.twist_angle - extra_twist_angle,
                                                  self.twist_angle + extra_twist_angle, self.anim_freq,
                                                  self.anim_offset, self.anim_seed)
        obj, mod = butil.modify_mesh(obj, 'SIMPLE_DEFORM', False, deform_method='BEND', deform_axis='Y',
                                     return_mod=True)
        bend_driver = mod.driver_add('angle').driver
        extra_bend_angle = uniform(0, np.pi / 60)
        bend_driver.expression = repeated_driver(self.bend_angle + extra_bend_angle,
                                                 self.bend_angle - extra_bend_angle, self.anim_freq,
                                                 self.anim_offset, self.anim_seed)
        obj.scale = uniform(.8, 1.2), uniform(.8, 1.2), self.z_scale
        assign_material(obj, self.material)
        return obj
