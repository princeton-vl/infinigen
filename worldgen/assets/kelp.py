# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: Jun 14 2023

import math

import bpy
import numpy as np
from numpy.random import normal, uniform

import util.blender as butil
from assets.seaweed import SeaweedFactory
from assets.utils.draw import leaf
from assets.utils.misc import log_uniform
from assets.utils.object import mesh2obj, data2mesh, new_cube
from assets.utils.decorate import assign_material, displace_vertices, join_objects, read_co
from assets.utils.nodegroup import build_curve
from util.math import FixedSeed
from nodes.node_wrangler import NodeWrangler, Nodes
from placement.detail import remesh_with_attrs
from placement.factory import AssetFactory
from surfaces import surface
from assets.utils.tag import tag_object, tag_nodegroup


class KelpFactory(AssetFactory):
    
    thickness = .02

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.angle = uniform(0, np.pi * 2)
            base_hue = uniform(.05, .4)
            self.material = surface.shaderfunc_to_material(SeaweedFactory.shader_seaweed, base_hue)

    def create_asset(self, placeholder, face_size=0.01, **params):
        if placeholder is not None:
            placeholder.rotation_euler = 0, 0, 0
        axis = self.build_axis()
        stem = self.build_stem(axis, face_size)
        leaves = self.build_leaves(axis)
        obj = join_objects([stem, leaves])
        butil.delete(axis)
        assign_material(obj, self.material)
        tag_object(obj, 'kelp')
        return obj

    def build_axis(self):
        bpy.ops.curve.primitive_bezier_curve_add()
        axis = bpy.context.active_object
        axis.name = 'kelp_axis'
        shift = np.array([np.cos(self.angle), np.sin(self.angle), 0]) * uniform(.1, .5)
        surface.add_geomod(axis, self.geo_kelp_axis, input_args=[shift])
        return axis

    @staticmethod
    def geo_kelp_axis(nw: NodeWrangler, shift):
        segments = 5
        length = 10
        stddev = 1
        resolution = 128
        keypoints = np.zeros((3, segments + 1))
        keypoints[-1, :] = np.linspace(0, length, segments + 1)
        keypoints[:-1, 1:] = normal(0, stddev, (2, segments))
        keypoints[:, 1:] += shift[:, np.newaxis] * keypoints[-1:, 1:]
        coefficients = np.array([math.comb(segments, i) for i in range(segments + 1)])[..., np.newaxis] * \
                       np.linspace(1, 0, resolution + 1)[np.newaxis, ...] ** np.arange(segments + 1)[::-1,
                       np.newaxis] * (
                               np.linspace(0, 1, resolution + 1)[np.newaxis, ...] ** np.arange(segments + 1)[
                           ..., np.newaxis])
        positions = (keypoints[..., np.newaxis] * coefficients[np.newaxis, ...]).sum(1)
        curve = build_curve(nw, positions)
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': curve})
        return curve

    def build_stem(self, axis, face_size):
        obj = new_cube()
        surface.add_geomod(obj, self.geo_kelp_stem, apply=True, input_args=[axis])
        with butil.SelectObjects(obj):
            bpy.ops.object.shade_flat()
        remesh_with_attrs(obj, face_size)
        tag_object(obj, 'stem')
        return obj

    @staticmethod
    def geo_kelp_stem(nw: NodeWrangler, axis):
        axis = nw.new_node(Nodes.ObjectInfo, [axis]).outputs['Geometry']
        profile_curve = nw.new_node(Nodes.CurveCircle,
                                    input_kwargs={'Radius': KelpFactory.thickness, 'Resolution': 6})
        geometry = nw.curve2mesh(axis, profile_curve)
        geometry = nw.new_node(Nodes.MergeByDistance, input_kwargs={'Geometry': geometry, 'Distance': .02})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def build_leaf():
        x_anchors = 0, uniform(.15, .35), uniform(.4, .6)
        y_anchors = 0, uniform(.08, .12), 0
        obj = leaf(x_anchors, y_anchors)
        displace_vertices(obj, lambda x, y, z: (0, 0, np.sin(np.pi * x / x_anchors[-1])))
        for direction in 'YZ':
            texture = bpy.data.textures.new(name='cap', type='STUCCI')
            texture.noise_scale = log_uniform(.1, .25)
            butil.modify_mesh(obj, 'DISPLACE', texture=texture, direction=direction, strength=uniform(.01, .02))

        locations = read_co(obj)
        obj.location = - locations[np.argmin(locations[:, 0])]
        obj.location[-1] -= .02
        butil.apply_transform(obj, loc=True)
        tag_object(obj, 'leaf')
        return obj

    @staticmethod
    def geo_kelp_leaves(nw: NodeWrangler, axis, leaves, direction):
        resolution = 128
        prob = .6
        perturb = .1
        factor = uniform(.4, .6)
        axis = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': axis}).outputs['Geometry']
        points, _, _, rotation = nw.new_node(Nodes.CurveToPoints, [axis, resolution]).outputs
        instance = nw.new_node(Nodes.CollectionInfo, [leaves, True, True])
        z_rotations = nw.new_node(Nodes.AccumulateField, [nw.uniform(np.pi / 4, np.pi / 2)])
        rotation = nw.new_node(Nodes.RotateEuler,
                               input_kwargs={'Rotate By': rotation, 'Rotation': nw.combine(0, 0, z_rotations)})
        rotation = nw.add(rotation, nw.uniform([-perturb] * 3, [perturb] * 3, data_type='FLOAT_VECTOR'))
        rotation = nw.new_node(Nodes.AlignEulerToVector,
                               input_kwargs={'Rotation': rotation, 'Factor': factor, 'Vector': direction},
                               attrs={'pivot_axis': 'Z'})
        instances = nw.new_node(Nodes.InstanceOnPoints, input_kwargs={
            'Points': points,
            'Instance': instance,
            'Rotation': rotation,
            'Pick Instance': True,
            'Selection': nw.bernoulli(prob),
            'Scale': nw.uniform([.8] * 3, [2.] * 3, data_type='FLOAT_VECTOR')})
        realized = nw.new_node(Nodes.RealizeInstances, [instances])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': realized})
        return realized

    def build_leaves(self, axis):
        leaves = new_cube()
        n_leaves = 5
        col = butil.group_in_collection([self.build_leaf() for _ in range(n_leaves)], "leaves")
        leaf_direction = np.array([np.cos(self.angle), np.sin(self.angle), uniform(-.1, 0)])
        surface.add_geomod(leaves, self.geo_kelp_leaves, apply=True, input_args=[axis, col, leaf_direction])
        butil.delete(list(col.objects))
        bpy.data.collections.remove(col)
        return leaves
