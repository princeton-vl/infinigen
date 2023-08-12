# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bmesh
import bpy
import numpy as np
from mathutils import Vector
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.corals.base import BaseCoralFactory
from infinigen.assets.utils.decorate import displace_vertices, geo_extension, join_objects
from infinigen.assets.utils.object import new_empty, new_icosphere
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util.blender import deep_clone_obj
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class StarBaseCoralFactory(BaseCoralFactory):
    tentacle_prob = 1.
    noise_strength = .002
    density = 3000

    @staticmethod
    def points_fn(nw: NodeWrangler, points):
        points = nw.new_node(Nodes.SeparateGeometry, [points, nw.new_node(Nodes.NamedAttribute, ['outermost'])])
        return points

    def __init__(self, factory_seed, coarse=False):
        super(StarBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.points_fn = StarBaseCoralFactory.points_fn

    @staticmethod
    def geo_dual_mesh(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        perturb = .05
        geometry = nw.new_node(Nodes.SetPosition,
                               [geometry, None, None, nw.uniform([-perturb] * 3, [perturb] * 3)])

        geometry = nw.new_node(Nodes.DualMesh, input_kwargs={'Mesh': geometry})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def geo_separate_faces(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        selection = nw.compare('GREATER_THAN', nw.separate(nw.new_node(Nodes.InputPosition))[-1], 0)
        geometry = nw.new_node(Nodes.SeparateGeometry, [geometry, selection])
        geometry = nw.new_node(Nodes.SplitEdges, [geometry])
        scale = nw.uniform(.9, 1.2)
        geometry = nw.new_node(Nodes.ScaleElements, [geometry, None, scale])
        geometry = nw.new_node(Nodes.StoreNamedAttribute,
            input_kwargs={'Geometry': geometry, 'Name': 'custom_normal', 'Value': nw.new_node(Nodes.InputNormal)},
            attrs={'data_type': 'FLOAT_VECTOR'})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def geo_flower(nw: NodeWrangler, size, resolution, anchor):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        t = nw.scalar_divide(nw.math('FLOOR', nw.scalar_divide(nw.new_node(Nodes.Index), size)), resolution)
        offset = nw.build_float_curve(t, [(0, 0), anchor, (1, 0)], 'AUTO')
        normal = nw.new_node(Nodes.NamedAttribute, ['custom_normal'], attrs={'data_type': 'FLOAT_VECTOR'})
        geometry = nw.new_node(Nodes.SetPosition, [geometry, None, None, nw.scale(offset, normal)])
        outer = nw.boolean_math('AND', nw.compare('GREATER_THAN', t, .4), nw.compare('LESS_THAN', t, .6))
        geometry = nw.new_node(Nodes.StoreNamedAttribute, 
            input_kwargs={'Geometry': geometry, 'Name': 'outermost', 'Value': outer})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def create_asset(self, face_size=0.01, **params):
        obj = new_icosphere(subdivisions=3)
        obj.location[-1] = uniform(.25, .5)
        butil.apply_transform(obj, loc=True)
        surface.add_geomod(obj, self.geo_dual_mesh, apply=True)
        displace_vertices(obj, lambda x, y, z: (0, 0, -.9 * np.clip(z, None, 0)))

        rings = deep_clone_obj(obj)
        levels = 3
        butil.modify_mesh(obj, 'SUBSURF', levels=levels, render_levels=levels)
        butil.modify_mesh(rings, 'SHRINKWRAP', target=obj)

        surface.add_geomod(rings, self.geo_separate_faces, apply=True)
        levels = 3
        butil.modify_mesh(rings, 'SUBSURF', levels=levels, render_levels=levels)

        butil.select_none()
        with butil.ViewportMode(rings, 'EDIT'):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.select_all(action='INVERT')
            bpy.ops.mesh.delete(type='VERT')

        flowers = []
        resolution = 16

        for ring in butil.split_object(rings):
            size = len(ring.data.vertices)
            center = np.mean([v.co for v in ring.data.vertices], 0)
            empty = new_empty(scale=[uniform(.3, .5) ** (1 / resolution)] * 3)
            butil.modify_mesh(ring, 'ARRAY', apply=True, use_relative_offset=False, use_object_offset=True,
                              count=resolution + 1, offset_object=empty)
            butil.delete(empty)

            with butil.ViewportMode(ring, 'EDIT'):
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.bridge_edge_loops()

                bm = bmesh.from_edit_mesh(ring.data)
                bm.verts.ensure_lookup_table()
                for i in range(1, resolution + 1):
                    c = np.mean([v.co for v in bm.verts[i * size:(i + 1) * size]], 0)
                    for j in range(i * size, (i + 1) * size):
                        bm.verts[j].co += Vector(center - c)
                bmesh.update_edit_mesh(ring.data)

                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.region_to_loop()
                bpy.ops.mesh.bridge_edge_loops()

            anchor = uniform(.4, .6), uniform(.08, .15)
            surface.add_geomod(ring, self.geo_flower, apply=True, input_args=[size, resolution, anchor])
            flowers.append(ring)

        obj = join_objects([obj, *flowers])
        surface.add_geomod(obj, geo_extension, apply=True)
        tag_object(obj, 'star_coral')
        return obj
