# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

import bpy
import numpy as np
from numpy.random import uniform

from assets.deformed_trees.base import BaseDeformedTreeFactory
from assets.utils.decorate import assign_material, join_objects, read_material_index, write_material_index
from assets.utils.nodegroup import geo_selection
from nodes.node_info import Nodes
from nodes.node_wrangler import NodeWrangler
from surfaces import surface
from util.blender import deep_clone_obj
from util import blender as butil
from assets.utils.tag import tag_object, tag_nodegroup

class HollowTreeFactory(BaseDeformedTreeFactory):

    @staticmethod
    def geo_texture(nw: NodeWrangler, material_index):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        selection = nw.compare('EQUAL', nw.new_node(Nodes.MaterialIndex), material_index)
        offset = nw.scale(nw.scalar_multiply(nw.musgrave(uniform(10, 20)), -uniform(.03, .06)),
                          nw.new_node(Nodes.InputNormal))
        geometry = nw.new_node(Nodes.SetPosition, [geometry, selection, None, offset])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def create_asset(self, face_size, **params):
        obj = self.build_tree(face_size, **params)
        scale = uniform(.8, 1.)
        threshold = uniform(.38, .42)

        def selection(nw: NodeWrangler):
            x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
            radius = nw.power(nw.scalar_add(nw.power(x, 2), nw.power(y, 2)), .5)
            vector = nw.combine(nw.scalar_divide(x, radius), nw.scalar_divide(y, radius), z)
            noise = nw.compare('GREATER_THAN',
                               nw.new_node(Nodes.NoiseTexture, [vector], input_kwargs={'Scale': scale}),
                               threshold)
            r_outside = nw.compare('GREATER_THAN', nw.scalar_add(nw.power(x, 2), nw.power(y, 2)), 1)
            z_lower = nw.scalar_add(.1,
                                    nw.scale(nw.new_node(Nodes.NoiseTexture, attrs={'noise_dimensions': '2D'}),
                                             .4))
            z_upper = nw.scalar_sub(3.5,
                                    nw.scale(nw.new_node(Nodes.NoiseTexture, attrs={'noise_dimensions': '2D'}),
                                             .4))
            z_outside = nw.boolean_math('OR', nw.compare('LESS_THAN', z, z_lower),
                                        nw.compare('GREATER_THAN', z, z_upper))
            return nw.boolean_math('OR', nw.boolean_math('OR', z_outside, noise), r_outside)

        surface.add_geomod(obj, geo_selection, apply=True, input_args=[selection])
        hollow = deep_clone_obj(obj)

        self.trunk_surface.apply(obj)
        butil.apply_modifiers(obj)
        assign_material(hollow, self.material)
        obj = join_objects([obj, hollow])

        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.bridge_edge_loops(type='PAIRS', number_cuts=10, interpolation='LINEAR')

        ring_material_index = list(obj.data.materials).index(obj.data.materials['shader_rings'])
        surface.add_geomod(obj, self.geo_texture, apply=True, input_args=[ring_material_index])

        material_indices = read_material_index(obj)
        null_indices = np.array([i for i, m in enumerate(obj.data.materials) if not hasattr(m, 'name')])
        material_indices[
            np.any(material_indices[:, np.newaxis] == null_indices[np.newaxis, :], -1)] = ring_material_index
        write_material_index(obj, material_indices)
        tag_object(obj, 'hollow_tree')
        return obj
