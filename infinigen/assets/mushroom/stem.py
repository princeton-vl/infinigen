# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import assign_material, geo_extension, join_objects, subsurface2face_size
from infinigen.assets.utils.draw import spin

from infinigen.assets.utils.misc import log_uniform
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class MushroomStemFactory(AssetFactory):

    def __init__(self, factory_seed, inner_radius, material_func, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.web_builders = [self.build_hollow_web, self.build_solid_web, None]
            web_weights = np.array([1, 1, 2])
            self.web_builder = np.random.choice(self.web_builders, p=web_weights / web_weights.sum())
            self.has_band = uniform(0, 1) < .75

            self.material = material_func()
            self.material_web = material_func()
            self.inner_radius = inner_radius

    def build_solid_web(self, inner_radius):
        outer_radius = inner_radius * uniform(1.5, 3.5)
        z = uniform(.0, .05)
        length = uniform(.15, .2)
        x_anchors = inner_radius, (outer_radius + inner_radius) / 2, outer_radius
        z_anchors = - z, -z - uniform(.3, .4) * length, -z - length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors)
        surface.add_geomod(obj, self.geo_inverse_band, apply=True, input_args=[-uniform(.008, .01)])
        tag_object(obj, 'web')
        return obj

    @staticmethod
    def geo_voronoi(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        selection = nw.compare('LESS_THAN',
                               nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Scale': uniform(15, 20)},
                                           attrs={'feature': 'DISTANCE_TO_EDGE'}), .06)
        geometry = nw.new_node(Nodes.SeparateGeometry, [geometry, selection])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def build_hollow_web(self, inner_radius):
        outer_radius = inner_radius * uniform(2, 3.5)
        z = uniform(.0, .05)
        length = log_uniform(.2, .4)
        x_anchors = inner_radius, (outer_radius + inner_radius) / 2, outer_radius
        z_anchors = - z, -z - uniform(.3, .4) * length, -z - length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors)
        levels = 3
        butil.modify_mesh(obj, 'SUBSURF', True, render_levels=levels, levels=levels)
        surface.add_geomod(obj, self.geo_voronoi, apply=True)
        butil.modify_mesh(obj, 'SMOOTH', iterations=2)
        tag_object(obj, 'web')
        return obj

    def create_asset(self, face_size, **params) -> bpy.types.Object:
        length = log_uniform(.4, .8)
        x_anchors = 0, self.inner_radius, log_uniform(1, 2) * self.inner_radius, self.inner_radius * uniform(1,
                                                                                                             1.2), 0
        z_anchors = 0, 0, -length * uniform(.3, .7), -length, -length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors, [1, 4])
        remesh_with_attrs(obj, face_size)
        if self.has_band:
            surface.add_geomod(obj, self.geo_band, apply=True, input_args=[length, uniform(.008, .01)])
        assign_material(obj, self.material)

        if self.web_builder is not None:
            web = self.web_builder(self.inner_radius)
            surface.add_geomod(web, geo_extension, apply=True)
            subsurface2face_size(web, face_size / 2)
            assign_material(obj, self.material_web)
            obj = join_objects([web, obj])

        texture = bpy.data.textures.new(name='cap', type='STUCCI')
        texture.noise_scale = uniform(.005, .01)
        butil.modify_mesh(obj, 'DISPLACE', strength=.008, texture=texture, mid_level=0)

        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=-uniform(0, np.pi / 2),
                          deform_axis='Y')
        tag_object(obj, 'stem')
        return obj

    @staticmethod
    def geo_band(nw: NodeWrangler, length, scale):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        wave = nw.new_node(Nodes.WaveTexture, input_kwargs={
            'Scale': log_uniform(5, 10),
            'Distortion': uniform(5, 10),
            'Detail Scale': 2, }, attrs={'bands_direction': 'Z', 'wave_profile': 'SAW'}).outputs['Fac']
        selection = nw.compare('LESS_THAN', nw.separate(nw.new_node(Nodes.InputPosition))[-1],
                               -uniform(.3, .7) * length)
        normal = nw.vector_math('NORMALIZE', nw.add(nw.new_node(Nodes.InputNormal), (0, 0, 2)))
        geometry = nw.new_node(Nodes.SetPosition,
                               [geometry, selection, None, nw.scale(nw.scale(wave, scale), normal)])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def geo_inverse_band(nw: NodeWrangler, scale):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
        vector = nw.combine(x, y, nw.scalar_multiply(-1, z))
        wave = nw.new_node(Nodes.WaveTexture, input_kwargs={
            'Vector': vector,
            'Scale': log_uniform(5, 10),
            'Distortion': uniform(5, 10),
            'Detail Scale': 2, }, attrs={'bands_direction': 'Z', 'wave_profile': 'SAW'}).outputs['Fac']
        normal = nw.vector_math('NORMALIZE', nw.add(nw.new_node(Nodes.InputNormal), (0, 0, 2)))
        geometry = nw.new_node(Nodes.SetPosition,
                               [geometry, None, None, nw.scale(nw.scale(wave, scale), normal)])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})
