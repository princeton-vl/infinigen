# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.object import new_cube
from infinigen.assets.utils.decorate import geo_extension, join_objects
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil
from infinigen.assets.cactus.base import BaseCactusFactory
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class PrickyPearBaseCactusFactory(BaseCactusFactory):
    spike_distance = .08

    @staticmethod
    def geo_leaf(nw: NodeWrangler):
        resolution = 64
        profile_curve = nw.new_node(Nodes.CurveCircle)
        curve = nw.new_node(Nodes.ResampleCurve, [nw.new_node(Nodes.CurveLine), None, resolution])
        anchors = [(0, uniform(.15, .2)), (uniform(.4, .6), log_uniform(.4, .5)), (1., .05)]
        radius = nw.scalar_multiply(nw.build_float_curve(nw.new_node(Nodes.SplineParameter), anchors, 'AUTO'),
                                    log_uniform(.5, 1.5))
        curve = nw.new_node(Nodes.SetCurveRadius, [curve, None, radius])
        geometry = nw.curve2mesh(curve, profile_curve)
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def build_leaf(self):
        obj = new_cube()
        surface.add_geomod(obj, self.geo_leaf, apply=True)
        surface.add_geomod(obj, geo_extension, apply=True, input_kwargs={'musgrave_dimensions': '2D'})
        obj.scale = uniform(.8, 1.2), uniform(.2, .25), uniform(.8, 1.2)
        butil.apply_transform(obj)
        return obj

    def build_leaves(self, level=0):
        if level == 0:
            return self.build_leaf()
        n = np.random.randint(1, 3)
        leaves = [self.build_leaves(level - 1) for _ in range(n)]
        base = self.build_leaf()
        angles = np.random.permutation(
            [-uniform(np.pi / 3, np.pi / 2), uniform(-np.pi / 16, np.pi / 16), uniform(np.pi / 3, np.pi / 2)])[
        :n]
        vectors = [[np.sin(a), 0, np.cos(a) + .5] for a in angles]
        locations = np.array([v.co for v in base.data.vertices])
        for a, v, leaf in zip(angles, vectors, leaves):
            index = np.argmax(locations @ v)
            leaf.location[-1] -= .15
            butil.apply_transform(leaf, loc=True)
            leaf.scale = [uniform(.5, .75)] * 3
            leaf.location = locations[index]
            leaf.rotation_euler = 0, a, uniform(-np.pi / 3, np.pi / 3)
        obj = join_objects([base, *leaves])
        return obj

    def create_asset(self, face_size=.01, **params) -> bpy.types.Object:
        obj = self.build_leaves(2)
        write_attr_data(obj, 'selection', np.ones(len(obj.data.vertices)))
        tag_object(obj, 'prickypear_cactus')
        return obj
