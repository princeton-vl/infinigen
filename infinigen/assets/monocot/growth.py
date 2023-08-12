# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

from functools import reduce

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import assign_material, displace_vertices, geo_extension, join_objects
from infinigen.assets.utils.misc import build_color_ramp, log_uniform
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.core.placement.detail import adapt_mesh_resolution
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util import blender as butil
from infinigen.assets.utils.object import data2mesh, mesh2obj, new_cube, origin2leftmost
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core import surface
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class MonocotGrowthFactory(AssetFactory):
    use_distance = False

    def __init__(self, factory_seed, coarse=False):
        super(MonocotGrowthFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.count = 128
            self.perturb = .05
            self.angle = np.pi / 6
            self.min_y_angle = 0.
            self.max_y_angle = np.pi / 2
            self.leaf_prob = uniform(.8, .9)
            self.leaf_range = 0, 1
            self.stem_offset = .2
            self.scale_curve = [(0, 1), (1, 1)]
            self.radius = .01
            self.bend_angle = np.pi / 4
            self.twist_angle = np.pi / 6
            self.z_drag = 0.
            self.z_scale = uniform(1., 1.2)
            self.align_factor = 0
            self.align_direction = 1, 0, 0
            self.base_hue = self.build_base_hue()
            self.bright_color = *colorsys.hsv_to_rgb(self.base_hue, uniform(.6, .8), log_uniform(.05, .1)), 1
            self.dark_color = *colorsys.hsv_to_rgb((self.base_hue + uniform(-.03, .03)) % 1, uniform(.8, 1.),
                                                   log_uniform(.05, .2)), 1
            self.material = shaderfunc_to_material(self.shader_monocot, self.dark_color, self.bright_color,
                                                   self.use_distance)

    @staticmethod
    def build_base_hue():
        return uniform(.15, .35)

    @property
    def is_grass(self):
        return False

    def build_leaf(self, face_size):
        raise NotImplemented

    @staticmethod
    def decorate_leaf(obj, y_ratio=4, y_bend_angle=np.pi / 6, z_bend_angle=np.pi / 6, noise_scale=.1,
                      strength=.02, leftmost=True):
        obj.rotation_euler[1] = -np.pi / 2
        butil.apply_transform(obj)
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=uniform(.5, 1) * y_bend_angle,
                          deform_axis='Y')
        obj.rotation_euler[1] = np.pi / 2
        butil.apply_transform(obj)
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=uniform(-1, 1) * z_bend_angle,
                          deform_axis='Z')

        displace_vertices(obj, lambda x, y, z: (0, 0, y_ratio * uniform(0, 1) * y * y))
        surface.add_geomod(obj, geo_extension, apply=True)

        texture = bpy.data.textures.new(name='grasses', type='STUCCI')
        texture.noise_scale = noise_scale
        butil.modify_mesh(obj, 'DISPLACE', strength=strength, texture=texture)

        for direction, width in zip('XY', obj.dimensions[:2]):
            texture = bpy.data.textures.new(name='grasses', type='STUCCI')
            texture.noise_scale = noise_scale
            butil.modify_mesh(obj, 'DISPLACE', strength=uniform(.01, .02) * width, texture=texture,
                              direction=direction)
        if leftmost:
            origin2leftmost(obj)
        return obj

    def make_geo_flower(self):
        def geo_flower(nw: NodeWrangler, leaves):
            stem = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
            line = nw.new_node(Nodes.CurveLine, input_kwargs={'End': (0, 0, self.stem_offset)})
            points = nw.new_node(Nodes.ResampleCurve, [line, None, self.count])
            parameter = nw.new_node(Nodes.SplineParameter)
            y_rotation = nw.build_float_curve(parameter, [(0, -self.min_y_angle), (1, -self.max_y_angle)])
            z_rotation = nw.new_node(Nodes.AccumulateField,
                                     [None, nw.uniform(self.angle * .95, self.angle * 1.05)])
            rotation = nw.combine(0, y_rotation, z_rotation)
            scale = nw.build_float_curve(parameter, self.scale_curve, 'AUTO')
            if self.perturb:
                rotation = nw.add(rotation, nw.uniform([-self.perturb] * 3, [self.perturb] * 3))
                scale = nw.add(scale, nw.uniform([-self.perturb] * 3, [self.perturb] * 3))
            if self.align_factor:
                rotation = nw.new_node(Nodes.AlignEulerToVector, input_kwargs={
                    'Rotation': rotation,
                    'Factor': surface.eval_argument(nw, self.align_factor),
                    'Vector': self.align_direction
                }, attrs={'pivot_axis': 'Z'})
            points, _, z_rotation = nw.new_node(Nodes.CaptureAttribute, [points, None, z_rotation]).outputs[:3]
            leaves = nw.new_node(Nodes.CollectionInfo, [leaves, True, True])
            is_leaf = reduce(lambda *xs: nw.boolean_math('AND', *xs), [nw.bernoulli(self.leaf_prob),
                nw.compare('GREATER_EQUAL', parameter, self.leaf_range[0]),
                nw.compare('LESS_EQUAL', parameter, self.leaf_range[1])])
            instances = nw.new_node(Nodes.InstanceOnPoints, input_kwargs={
                'Points': points,
                'Selection': is_leaf,
                'Instance': leaves,
                'Pick Instance': True,
                'Rotation': rotation,
                'Scale': scale
            })
            geometry = nw.new_node(Nodes.RealizeInstances, [instances])
            geometry = nw.new_node(Nodes.StoreNamedAttribute, 
                input_kwargs={'Geometry': geometry, 'Name':'z_rotation', 'Value': z_rotation})
            geometry = nw.new_node(Nodes.JoinGeometry, [[stem, geometry]])
            nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

        return geo_flower

    def build_instance(self, i, face_size):
        obj = self.build_leaf(face_size)
        origin2leftmost(obj)
        obj.location[0] -= .01
        butil.apply_transform(obj, loc=True)
        return obj

    def make_collection(self, face_size):
        return make_asset_collection(self.build_instance, 10, 'leaves', verbose=False, face_size=face_size)

    def build_stem(self, face_size):
        obj = mesh2obj(data2mesh([[0, 0, 0], [0, 0, self.stem_offset]], [[0, 1]]))
        butil.modify_mesh(obj, 'SUBSURF', True, levels=9, render_levels=9)
        surface.add_geomod(obj, geo_radius, apply=True, input_args=[self.radius, 16])
        adapt_mesh_resolution(obj, face_size, 'subdivide')

        texture = bpy.data.textures.new(name='grasses', type='STUCCI')
        texture.noise_scale = .1
        butil.modify_mesh(obj, 'DISPLACE', strength=.01, texture=texture)
        tag_object(obj, 'stem')
        return obj

    def create_asset(self, **params):
        obj = self.create_raw(**params)
        self.decorate_monocot(obj)
        tag_object(obj, 'monocot_growth')
        return obj

    def create_raw(self, face_size=.01, apply=True, **params):
        if self.angle != 0:
            frequency = 2 * np.pi / self.angle
            if .01 < frequency - int(frequency) < .05:
                frequency += .05
            elif -.05 < frequency - int(frequency) < -.01:
                frequency -= .05
            self.angle = 2 * np.pi / frequency
        leaves = self.make_collection(face_size)
        obj = self.build_stem(face_size)
        surface.add_geomod(obj, self.make_geo_flower(), apply=apply, input_args=[leaves])
        if apply:
            butil.delete_collection(leaves)
        tag_object(obj, 'flower')
        return obj

    def decorate_monocot(self, obj):
        displace_vertices(obj, lambda x, y, z: (0, 0, -self.z_drag * (x * x + y * y)))
        surface.add_geomod(obj, geo_extension, apply=True, input_args=[.4])
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='TWIST',
                          angle=uniform(-self.twist_angle, self.twist_angle), deform_axis='Z')
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=uniform(0, self.bend_angle))
        obj.scale = uniform(.8, 1.2), uniform(.8, 1.2), self.z_scale
        obj.rotation_euler[-1] = uniform(0, np.pi * 2)
        butil.apply_transform(obj)
        assign_material(obj, self.material)

    @staticmethod
    def shader_monocot(nw: NodeWrangler, dark_color, bright_color, use_distance):
        specular = uniform(.0, .2)
        clearcoat = 0 if uniform(0, 1) < .8 else uniform(.2, .5)
        if use_distance:
            distance = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'distance'}).outputs['Fac']
            exponent = uniform(1.8, 3.5)
            ratio = nw.scalar_sub(1, nw.math('POWER', nw.scalar_sub(1, distance), exponent))
            color = nw.new_node(Nodes.MixRGB, [ratio, bright_color, dark_color])
        else:
            color = build_color_ramp(nw, nw.musgrave(10), [.0, .3, .7, 1.],
                                     [bright_color, bright_color, dark_color, dark_color], )
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 50})
        roughness = nw.build_float_curve(noise_texture, [(0, .5), (1, .8)])
        bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Specular': specular,
            'Clearcoat': clearcoat,
            'Subsurface': .01,
            'Subsurface Radius': (.01, .01, .01),
        })
        return bsdf
