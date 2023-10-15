# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


import bmesh
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.deformed_trees.base import BaseDeformedTreeFactory
from infinigen.assets.utils.decorate import assign_material, join_objects, remove_vertices, separate_loose
from infinigen.assets.utils.draw import cut_plane
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class FallenTreeFactory(BaseDeformedTreeFactory):

    @staticmethod
    def geo_cutter(nw: NodeWrangler, strength, scale, radius, metric_fn):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
        selection = nw.compare('LESS_THAN', nw.scalar_add(nw.power(x, 2), nw.power(y, 2)), 1)
        offset = nw.scalar_multiply(nw.new_node(Nodes.Clamp, [nw.new_node(Nodes.NoiseTexture, input_kwargs={
            'Vector': nw.new_node(Nodes.InputPosition),
            'Scale': scale
        }), .3, .7]), strength)
        offset = nw.scalar_multiply(offset, nw.build_float_curve(x, [(-radius, 1), (radius, 0)]))
        anchors = (-1, 0), (-.5, 0), (0, -1), (.5, 0), (1, 0)
        offset = nw.scalar_multiply(offset, nw.build_float_curve(surface.eval_argument(nw, metric_fn), anchors))
        geometry = nw.new_node(Nodes.SetPosition, [geometry, selection, None, nw.combine(0, 0, offset)])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def build_half(self, obj, cut_center, cut_normal, noise_strength, noise_scale, radius, is_up=True):
        obj, cut = cut_plane(obj, cut_center, cut_normal, not is_up)
        assign_material(cut, self.material)
        obj = join_objects([obj, cut])
        with butil.ViewportMode(obj, 'EDIT'), butil.Suppress():
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.remove_doubles(threshold=1e-2)
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.fill_holes()
        metric_fn = lambda nw: nw.dot(nw.sub(nw.new_node(Nodes.InputPosition), cut_center), cut_normal)
        surface.add_geomod(obj, self.geo_cutter, apply=True,
                           input_args=[noise_strength, noise_scale, radius, metric_fn])
        obj = separate_loose(obj)
        surface.add_geomod(obj, self.geo_xyz, apply=True)
        return obj

    def create_asset(self, i, distance=0, **params):
        upper = self.build_tree(i, distance, **params)
        radius = max([np.sqrt(v.co[0] ** 2 + v.co[1] ** 2) for v in upper.data.vertices if v.co[-1] < .1])
        self.trunk_surface.apply(upper)
        butil.apply_modifiers(upper)
        lower = deep_clone_obj(upper, keep_materials=True)
        cut_center = np.array([0, 0, uniform(.6, 1.2)])
        cut_normal = np.array([uniform(.1, .2), 0, 1])
        noise_strength = uniform(.3, .5)
        noise_scale = uniform(10, 15)
        upper = self.build_half(upper, cut_center, cut_normal, noise_strength, noise_scale, radius, True)
        lower = self.build_half(lower, cut_center, cut_normal, noise_strength, noise_scale, radius, False)

        ortho = np.array([-cut_normal[0], 0, 1])
        locations = np.array([v.co for v in lower.data.vertices])
        highest = locations[np.argmax(locations @ ortho)] + np.array(
            [-uniform(.05, .15), 0, -uniform(.05, .15)])
        upper.location = - highest
        butil.apply_transform(upper, loc=True)

        x, _, z = np.mean(np.stack([v.co for v in upper.data.vertices]), 0)
        r = np.sqrt(x * x + z * z)
        if r > 0:
            upper.rotation_euler[1] = np.pi / 2 + np.arcsin((highest[-1] - uniform(0, .2)) / r) - np.arctan(
                x / z)
        upper.location = highest
        butil.apply_transform(upper, loc=True)
        remove_vertices(upper, lambda x, y, z: z < -.5)
        upper = separate_loose(upper)
        obj = join_objects([upper, lower])
        tag_object(obj, 'fallen_tree')
        return obj
