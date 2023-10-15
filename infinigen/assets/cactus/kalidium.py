# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.trees.tree import TreeVertices, build_radius_tree, recursive_path
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import data2mesh, mesh2obj, new_cube, origin2lowest
from infinigen.assets.utils.decorate import displace_vertices, geo_extension, read_co, remove_vertices, separate_loose, \
    subsurface2face_size
from infinigen.assets.utils.shortest_path import geo_shortest_path
from infinigen.core.nodes.node_info import Nodes

from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.assets.cactus.base import BaseCactusFactory
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class KalidiumBaseCactusFactory(BaseCactusFactory):
    cap_percentage = .0
    noise_strength = .0
    density = .0

    @staticmethod
    def build_twig(i):
        branch_config = {
            'n': 1,
            'path_kargs': lambda idx: {'n_pts': 5, 'std': .5, 'momentum': .85, 'sz': .01},
            'spawn_kargs': lambda idx: {'init_vec': (0, 0, 1)}
        }
        obj = build_radius_tree(None, branch_config, .005)
        surface.add_geomod(obj, geo_radius, apply=True, input_args=['radius'])
        return obj

    def create_asset(self, face_size=.01, **params) -> bpy.types.Object:
        resolution = 20
        obj = new_cube(location=(1, 1, 1))
        butil.modify_mesh(obj, 'ARRAY', count=resolution, relative_offset_displace=(1, 0, 0),
                          use_merge_vertices=True)
        butil.modify_mesh(obj, 'ARRAY', count=resolution, relative_offset_displace=(0, 1, 0),
                          use_merge_vertices=True)
        butil.modify_mesh(obj, 'ARRAY', count=resolution, relative_offset_displace=(0, 0, 1),
                          use_merge_vertices=True)
        obj.scale = [1 / resolution] * 3
        obj.location = -1, -1, -.1
        butil.apply_transform(obj, loc=True)
        remove_vertices(obj,
                        lambda x, y, z: (x ** 2 + y ** 2 + (z - 1) ** 2 > 1.1) | (uniform(0, 1, len(x)) < .05))
        end_indices = np.nonzero(read_co(obj)[:, -1] < 5 / resolution)[0]
        end_index = lambda nw: nw.build_index_case(np.random.choice(end_indices, 5))
        displace_vertices(obj, lambda x, y, z: uniform(-.8 / resolution, .8 / resolution, (3, len(x))))
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        surface.add_geomod(obj, geo_extension, apply=True)

        weight = lambda nw: nw.scalar_multiply(
            nw.vector_math('DISTANCE', *nw.new_node(Nodes.InputEdgeVertices).outputs[2:]), nw.uniform(.8, 1))
        surface.add_geomod(obj, geo_shortest_path, apply=True, input_args=[end_index, weight, .05])
        surface.add_geomod(obj, geo_radius, apply=True, input_args=[.006])

        twigs = make_asset_collection(self.build_twig, 5, verbose=False)
        surface.add_geomod(obj, self.geo_twigs, apply=True, input_args=[twigs])
        butil.delete_collection(twigs)
        obj = separate_loose(obj)

        obj.scale = uniform(.8, 1.2, 3)
        butil.apply_transform(obj)
        subsurface2face_size(obj, face_size)
        origin2lowest(obj)
        tag_object(obj, 'kalidium_cactus')
        return obj

    @staticmethod
    def geo_twigs(nw: NodeWrangler, instances):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])

        points, _, rotation = nw.new_node(Nodes.DistributePointsOnFaces, [geometry],
                                          input_kwargs={'Density': 2e3}).outputs[:3]
        points = nw.new_node(Nodes.MergeByDistance, [points, None, .005])
        perturb = .4
        rotation = nw.new_node(Nodes.AlignEulerToVector,
                               [nw.add(rotation, nw.uniform([-perturb] * 3, [perturb] * 3)),
                                   nw.uniform(.2, .5)], attrs={'axis': 'Z'})
        instances = nw.new_node(Nodes.CollectionInfo, [instances, True, True])

        twigs = nw.new_node(Nodes.RealizeInstances, [nw.new_node(Nodes.InstanceOnPoints,
                                                                 [points, None, instances, True, None, rotation,
                                                                     nw.combine(1, 1, nw.uniform(1., 1.5))])])
        geometry = nw.new_node(Nodes.JoinGeometry, [[geometry, twigs]])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})
