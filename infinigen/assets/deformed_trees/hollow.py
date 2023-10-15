# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.deformed_trees.base import BaseDeformedTreeFactory
from infinigen.assets.utils.decorate import assign_material, join_objects, read_co, read_material_index, separate_loose, \
    write_material_index
from infinigen.assets.utils.nodegroup import geo_selection
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util.blender import deep_clone_obj, select_none
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class HollowTreeFactory(BaseDeformedTreeFactory):

    @staticmethod
    def geo_texture(nw: NodeWrangler, material_index):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        selection = nw.compare('EQUAL', nw.new_node(Nodes.MaterialIndex), material_index)
        offset = nw.scale(nw.scalar_multiply(nw.musgrave(uniform(10, 20)), -uniform(.03, .06)),
                          nw.new_node(Nodes.InputNormal))
        geometry = nw.new_node(Nodes.SetPosition, [geometry, selection, None, offset])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def filter_lower(obj):
        select_none()
        objs = butil.split_object(obj)
        filtered = [o for o in objs if np.min(read_co(o)[:, -1]) < .5]
        obj = filtered[np.argmax([len(o.data.vertices) for o in filtered])]
        objs.remove(obj)
        butil.delete(objs)
        return obj

    def create_asset(self, i, distance=0, **params):
        obj = self.build_tree(i, distance, **params)
        scale = uniform(.8, 1.)
        threshold = uniform(.36, .4)

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
        obj = join_objects([self.filter_lower(obj), self.filter_lower(hollow)])

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
