# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bmesh
import bpy
import numpy as np
from mathutils import kdtree
from numpy.random import uniform

from infinigen.assets.corals.base import BaseCoralFactory
from infinigen.assets.corals.tentacles import make_radius_points_fn
from infinigen.assets.utils.decorate import displace_vertices, geo_extension, read_co, remove_vertices, \
    separate_loose, write_co
from infinigen.assets.utils.draw import make_circular_interp
from infinigen.assets.utils.misc import log_uniform
from infinigen.assets.utils.object import new_circle, origin2lowest
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class ElkhornBaseCoralFactory(BaseCoralFactory):
    tentacle_prob = 0.
    noise_strength = .005

    def __init__(self, factory_seed, coarse=False):
        super(ElkhornBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.points_fn = make_radius_points_fn(.05, .6)

    @staticmethod
    def geo_elkhorn(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        start_index = nw.boolean_math('AND', nw.compare('GREATER_THAN', nw.vector_math('LENGTH', nw.new_node(
            Nodes.InputPosition)), .7), nw.bernoulli(.005))
        end_index = nw.compare('LESS_THAN', nw.vector_math('LENGTH', nw.new_node(Nodes.InputPosition)), .02)
        distance = nw.vector_math('DISTANCE', *nw.new_node(Nodes.InputEdgeVertices).outputs[2:])
        weight = nw.scale(distance, nw.musgrave(10))

        curve = nw.new_node(Nodes.EdgePathToCurve, [geometry, start_index,
            nw.new_node(Nodes.ShortestEdgePath, [end_index, weight]).outputs[0]])
        curve = nw.new_node(Nodes.SplineType, [curve], attrs={'spline_type': 'NURBS'})

        geometry = nw.new_node(Nodes.MergeByDistance, [nw.curve2mesh(curve), None, .005])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def create_asset(self, face_size=0.01, **params):
        obj = new_circle(location=(0, 0, 0), vertices=1024)
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.fill_grid()
        displace_vertices(obj, lambda x, y, z: (*uniform(-.005, .005, (2, len(x))), 0))
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        temp = deep_clone_obj(obj)
        surface.add_geomod(temp, self.geo_elkhorn, apply=True)

        locations = np.array([temp.matrix_world @ v.co for v in temp.data.vertices])
        butil.delete(temp)
        self.tree2mesh(obj, locations)
        obj = separate_loose(obj)
        angles = self.build_angles(obj)
        self.cluster_displace(obj, angles)
        obj = separate_loose(obj)
        obj.rotation_euler[-1] = uniform(0, np.pi * 2)
        butil.apply_transform(obj)

        butil.modify_mesh(obj, 'SOLIDIFY', thickness=.02)
        surface.add_geomod(obj, geo_extension, apply=True, input_kwargs={'musgrave_dimensions': '2D'})
        texture = bpy.data.textures.new(name='elkhorn_coral', type='STUCCI')
        texture.noise_scale = log_uniform(.1, .5)
        butil.modify_mesh(obj, 'DISPLACE', True, strength=uniform(.1, .2), texture=texture, mid_level=0,
                          direction='Z')
        origin2lowest(obj)
        tag_object(obj, 'elkhorn_coral')
        return obj

    @staticmethod
    def tree2mesh(obj, locations):
        kd = kdtree.KDTree(len(locations))
        for i, loc in enumerate(locations):
            kd.insert(loc, i)
        kd.balance()

        large_radius = uniform(.08, .12)
        remove_vertices(obj, lambda x, y, z: np.array(
            [kd.find(v)[-1] for v in np.stack([x, y, z], -1)]) > .015 + large_radius * (
                                                     1 - np.sqrt(x * x + y * y)))

    @staticmethod
    def build_angles(obj):
        angle_radius = .2
        with butil.ViewportMode(obj, 'EDIT'):
            bm = bmesh.from_edit_mesh(obj.data)
            angles = np.full(len(bm.verts), -100.)
            queue = set()
            for v in bm.verts:
                x, y, z = v.co
                if np.sqrt(x * x + y * y) <= angle_radius:
                    angles[v.index] = np.arctan2(y, x)
                    for e in v.link_edges:
                        o = e.other_vert(v)
                        queue.add(o)
            while queue:
                new_queue = set()
                for v in queue:
                    pairs = []
                    if angles[v.index] <= -100.:
                        for e in v.link_edges:
                            o = e.other_vert(v)
                            if angles[o.index] > -100.:
                                pairs.append((e.calc_length(), angles[o.index]))
                        angles[v.index] = min(pairs)[1]
                    for e in v.link_edges:
                        o = e.other_vert(v)
                        if angles[o.index] <= -100.:
                            new_queue.add(o)
                queue = new_queue
        return angles

    @staticmethod
    def cluster_displace(obj, angles):
        f_scale = make_circular_interp(.3, 1., 5)
        f_rotation = make_circular_interp(0, np.pi / 3, 10)
        f_power = make_circular_interp(1., 1.6, 5)

        x, y, z = read_co(obj).T
        a = np.array([angles[_] for _ in range(len(x))]) + np.pi
        z += f_scale(a) * (x * x + y * y) ** f_power(a)
        rotation = f_rotation(a)
        c, s = np.cos(rotation), np.sin(rotation)
        co = np.stack([c * x - s * z, c * y - s * z, c * z + s * np.sqrt(x * x + y * y)], -1)
        write_co(obj, co)
        with butil.ViewportMode(obj, 'EDIT'):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = []
            for e in bm.edges:
                if e.calc_length() > .04:
                    geom.append(e)
            bmesh.ops.delete(bm, geom=geom, context='EDGES')
            bmesh.update_edit_mesh(obj.data)
        return obj
