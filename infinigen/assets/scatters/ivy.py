# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this
# source tree.

# Authors: Lingjie Mei


from collections.abc import Iterable

import numpy as np
from numpy.random import uniform

from infinigen.assets.leaves.leaf_maple import LeafFactoryMaple
from infinigen.assets.trees.generate import random_season
from infinigen.assets.utils.decorate import assign_material, fix_tree
from infinigen.assets.utils.nodegroup import geo_base_selection, geo_radius
from infinigen.assets.utils.shortest_path import geo_shortest_path
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.surface import shaderfunc_to_material
from infinigen.assets.materials.simple_brownish import shader_simple_brown
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


def geo_leaf(nw: NodeWrangler, leaves):
    leaf_up_prob = uniform(.0, .2)
    geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    normal = nw.new_node(Nodes.NamedAttribute, ['custom_normal'], attrs={'data_type': 'FLOAT_VECTOR'})
    tangent = nw.new_node(Nodes.NamedAttribute, ['tangent'], attrs={'data_type': 'FLOAT_VECTOR'})
    cotangent = nw.vector_math('CROSS_PRODUCT', tangent, normal)
    switch = nw.compare('LESS_THAN', nw.separate(cotangent)[-1], 0)
    cotangent = nw.scale(nw.switch(nw.bernoulli(leaf_up_prob), -1, 1),
                         nw.scale(nw.switch(switch, 1, -1), cotangent))

    perturb = np.pi / 6
    points, _, rotation = nw.new_node(Nodes.DistributePointsOnFaces,
                                      input_kwargs={'Mesh': geometry, 'Density': uniform(500, 1000)}).outputs[
    :3]
    rotation = nw.new_node(Nodes.AlignEulerToVector, [rotation, 1., normal], attrs={'axis': 'Z'})
    # Leaves have primary axes Y
    rotation = nw.new_node(Nodes.AlignEulerToVector, [rotation, 1., cotangent],
                           attrs={'axis': 'Y', 'pivot_axis': 'Z'})
    rotation = nw.add(rotation, nw.uniform([-perturb] * 3, [perturb] * 3))

    leaves = nw.new_node(Nodes.CollectionInfo, [leaves, True, True])
    instances = nw.new_node(Nodes.InstanceOnPoints, input_kwargs={
        'Points': points,
        'Instance': leaves,
        'Pick Instance': True,
        'Rotation': rotation,
        'Scale': nw.uniform(.6, 1.)
    })
    instances = nw.new_node(Nodes.RealizeInstances, [instances])
    geometry = nw.new_node(Nodes.JoinGeometry, [[geometry, instances]])
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})


class LeafFactoryIvy(LeafFactoryMaple):

    def __init__(self, factory_seed, season='spring', coarse=False):
        super().__init__(factory_seed, season, coarse)

    def create_asset(self, face_size, **params):
        obj = super().create_asset(face_size=face_size, **params)
        obj.scale = [.2] * 3
        butil.apply_transform(obj)
        butil.modify_mesh(obj, 'WELD', merge_threshold=face_size / 2, mode='CONNECTED')
        tag_object(obj, 'leaf_ivy')
        return obj


class Ivy:

    def __init__(self):
        self.factory = LeafFactoryIvy(np.random.randint(0, 1e5), random_season())
        self.col = make_asset_collection(self.factory, 5)

    def apply(self, obj, selection=None):

        scatter_obj = butil.spawn_vert('scatter:' + 'ivy')
        surface.add_geomod(scatter_obj, geo_base_selection, apply=True, input_args=[obj, selection, .05])

        end_index = lambda nw: nw.compare('EQUAL', nw.new_node(Nodes.Index),
                                          np.random.randint(len(scatter_obj.data.vertices)))
        weight = lambda nw: nw.scalar_multiply(nw.uniform(.8, 1), nw.scalar_sub(2, nw.math('ABSOLUTE', nw.dot(
            nw.vector_math('NORMALIZE', nw.sub(*nw.new_node(Nodes.InputEdgeVertices).outputs[2:])),
            (0, 0, 1)))))
        surface.add_geomod(scatter_obj, geo_shortest_path, apply=True,
                           input_args=[end_index, weight, uniform(.1, .15), uniform(.1, .15)])
        fix_tree(scatter_obj)
        surface.add_geomod(scatter_obj, geo_radius, apply=True, input_args=[.005, 12])
        assign_material(scatter_obj, shaderfunc_to_material(shader_simple_brown))
        surface.add_geomod(scatter_obj, geo_leaf, apply=True, input_args=[self.col])

        return scatter_obj

def apply(obj, selection=None):
    factory = LeafFactoryIvy(np.random.randint(0, 1e5), random_season())
    col = make_asset_collection(factory, 5)

    scatter_obj = butil.spawn_vert('scatter:' + 'ivy')
    surface.add_geomod(scatter_obj, geo_base_selection, apply=True, input_args=[obj, selection, .05])

    end_index = lambda nw: nw.compare('EQUAL', nw.new_node(Nodes.Index),
                                        np.random.randint(len(scatter_obj.data.vertices)))
    weight = lambda nw: nw.scalar_multiply(nw.uniform(.8, 1), nw.scalar_sub(2, nw.math('ABSOLUTE', nw.dot(
        nw.vector_math('NORMALIZE', nw.sub(*nw.new_node(Nodes.InputEdgeVertices).outputs[2:])),
        (0, 0, 1)))))
    surface.add_geomod(scatter_obj, geo_shortest_path, apply=True,
                        input_args=[end_index, weight, uniform(.1, .15), uniform(.1, .15)])
    fix_tree(scatter_obj)
    surface.add_geomod(scatter_obj, geo_radius, apply=True, input_args=[.005, 12])
    assign_material(scatter_obj, shaderfunc_to_material(shader_simple_brown))
    surface.add_geomod(scatter_obj, geo_leaf, apply=True, input_args=[col])

    return scatter_obj
