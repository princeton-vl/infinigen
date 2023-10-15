# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bmesh
import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.corals.base import BaseCoralFactory
from infinigen.assets.utils.decorate import displace_vertices, geo_extension, read_co, subsurface2face_size, treeify
from infinigen.assets.utils.draw import shape_by_angles
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import new_circle, origin2lowest
from infinigen.assets.utils.shortest_path import geo_shortest_path
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class FanBaseCoralFactory(BaseCoralFactory):
    tentacle_prob = 0.
    noise_strength = 0.

    @staticmethod
    def weight(nw: NodeWrangler):
        u, v = nw.new_node(Nodes.InputEdgeVertices).outputs[2:]
        length = nw.vector_math('DISTANCE', u, v)
        return nw.uniform(nw.scalar_multiply(length, .4), length)

    def create_asset(self, face_size=0.01, **params):
        obj = new_circle(vertices=512)
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.fill_grid()
        displace_vertices(obj, lambda x, y, z: uniform(-.005, .005, (3, len(x))))
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        shape_by_angles(obj, np.array([-np.pi / 2, 0, np.pi / 2]),
                        np.array([uniform(.2, .8), 1, uniform(.2, .8)]))
        obj.rotation_euler = np.pi / 2, -np.pi / 2, 0
        butil.apply_transform(obj)

        end_indices = np.nonzero(read_co(obj)[:, -1] < 1e-2)[0]
        end_index = lambda nw: nw.build_index_case(np.random.choice(end_indices, 5))
        texture = bpy.data.textures.new(name='fan', type='STUCCI')
        texture.noise_scale = uniform(.5, 1)
        butil.modify_mesh(obj, 'DISPLACE', texture=texture, strength=uniform(.5, 1.), direction='Y')
        surface.add_geomod(obj, geo_extension, apply=True)
        obj.scale = uniform(.6, 1.2), 1, 1
        butil.apply_transform(obj)
        surface.add_geomod(obj, geo_shortest_path, input_args=[end_index, self.weight, .05], apply=True)
        obj = self.add_radius(obj)
        surface.add_geomod(obj, geo_radius, apply=True, input_args=['radius', 32])
        butil.modify_mesh(obj, 'WELD', merge_threshold=.001)
        subsurface2face_size(obj, face_size)
        origin2lowest(obj)
        tag_object(obj, 'fan_coral')
        return obj

    @staticmethod
    def add_radius(obj):
        obj = treeify(obj)
        counts = np.zeros(len(obj.data.vertices))
        with butil.ViewportMode(obj, 'EDIT'):
            bm = bmesh.from_edit_mesh(obj.data)
            queue = list(sorted(bm.verts, key=lambda v: v.co[-1]))[:1]
            visited = np.zeros(len(bm.verts))
            visited[queue[0].index] = 1
            order = queue.copy()
            while queue:
                v = queue.pop()
                for e in v.link_edges:
                    o = e.other_vert(v)
                    if not visited[o.index]:
                        visited[o.index] = 1
                        queue.append(o)
                        order.append(o)
            for v in reversed(order):
                count = 1
                for e in v.link_edges:
                    count += counts[e.other_vert(v).index]
                counts[v.index] = count
        vg = obj.vertex_groups.new(name='radius')
        thresh = uniform(100, 200)
        ratio = uniform(.5, 1.5)
        for i, c in enumerate(counts):
            r = 1 if c < thresh else 1 + ratio * np.log(c / thresh)
            vg.add([i], .008 * r, 'REPLACE')
        return obj
