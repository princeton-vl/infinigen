# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

import numpy as np
from numpy.random import uniform

from assets.deformed_trees import FallenTreeFactory
from assets.utils.decorate import read_co
from nodes.node_info import Nodes
from nodes.node_wrangler import NodeWrangler
from surfaces import surface
from util import blender as butil
from assets.utils.tag import tag_object, tag_nodegroup

class TruncatedTreeFactory(FallenTreeFactory):

    @staticmethod
    def geo_cutter(nw: NodeWrangler, strength, scale, radius, metric_fn):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        offset = nw.scalar_multiply(nw.new_node(Nodes.Clamp, [nw.new_node(Nodes.NoiseTexture, input_kwargs={
            'Vector': nw.new_node(Nodes.InputPosition),
            'Scale': scale}), .3, .7]), strength)
        anchors = (-1, 0), (-.5, 0), (0, 1), (.5, 0), (1, 0)
        offset = nw.scalar_multiply(offset, nw.build_float_curve(surface.eval_argument(nw, metric_fn), anchors))
        geometry = nw.new_node(Nodes.SetPosition, [geometry, None, None, nw.combine(0, 0, offset)])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def create_asset(self, face_size, **params):
        obj = self.build_tree(face_size, **params)
        x, y, z = read_co(obj).T
        radius = np.amax(np.sqrt(x ** 2 + y ** 2)[z < .1])
        self.trunk_surface.apply(obj)
        butil.apply_modifiers(obj)
        cut_center = np.array([0, 0, uniform(.8, 1.5)])
        cut_normal = np.array([uniform(-.4, .4), 0, 1])
        noise_strength = uniform(.6, 1.)
        noise_scale = uniform(10, 15)
        self.build_half(obj, cut_center, cut_normal, noise_strength, noise_scale, radius, False)
        tag_object(obj, 'truncated_tree')
        return obj
