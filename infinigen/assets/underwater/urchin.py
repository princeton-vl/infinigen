# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys
import bpy
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.creatures.util.animation.driver_repeated import repeated_driver
from infinigen.assets.utils.object import new_icosphere
from infinigen.assets.utils.decorate import assign_material, geo_extension, separate_loose
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import adapt_mesh_resolution
from infinigen.core.placement.factory import AssetFactory
from infinigen.core import surface
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class UrchinFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.base_hue = uniform(-.25, .15) % 1
            self.materials = [surface.shaderfunc_to_material(shader, self.base_hue) for shader in
                [self.shader_spikes, self.shader_girdle, self.shader_base]]
            self.freq = 1 / log_uniform(100, 200)

    def create_asset(self, placeholder, face_size=0.01, **params):
        obj = new_icosphere(subdivisions=4)
        surface.add_geomod(obj, geo_extension, apply=True)
        obj.scale[-1] = uniform(.8, 1.)
        butil.apply_transform(obj)
        butil.modify_mesh(obj, 'BEVEL', offset_type='PERCENT', width_pct=25, angle_limit=0)
        surface.add_geomod(obj, self.geo_extrude, apply=True, attributes=['spike', 'girdle'],
                           domains=['FACE'] * 2)
        levels = 1
        butil.modify_mesh(obj, 'SUBSURF', apply=True, levels=levels, render_levels=levels)
        obj.scale = [2 / max(obj.dimensions)] * 3
        obj.scale[-1] *= log_uniform(.6, 1.2)
        butil.apply_transform(obj)
        adapt_mesh_resolution(obj, face_size, method='subdiv_by_area')
        obj = separate_loose(obj)
        butil.modify_mesh(obj, 'DISPLACE', texture=bpy.data.textures.new(name='urchin', type='STUCCI'),
                          strength=.005, mid_level=0)
        surface.add_geomod(obj, self.geo_material_index, apply=True, input_attributes=[None, 'spike', 'girdle'])
        assign_material(obj, self.materials)
        self.animate_stretch(obj)
        tag_object(obj, 'urchin')
        return obj

    def animate_stretch(self, obj):
        obj, mod = butil.modify_mesh(obj, 'SIMPLE_DEFORM', False, return_mod=True, deform_method='STRETCH',
                                     deform_axis='Z')
        driver = mod.driver_add('factor').driver
        driver.expression = repeated_driver(-.1, .1, self.freq)

    @staticmethod
    def geo_extrude(nw: NodeWrangler):
        face_prob = .98
        girdle_height = .1
        extrude_height = log_uniform(1., 5.)
        perturb = .1
        girdle_size = uniform(.6, 1)
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        face_vertices = nw.new_node(Nodes.FaceNeighbors)
        selection = nw.boolean_math('AND', nw.compare('GREATER_EQUAL', face_vertices, 5),
                                    nw.bernoulli(face_prob))
        geometry, top, _ = nw.new_node(Nodes.ExtrudeMesh, [geometry, selection, None, girdle_height]).outputs
        geometry, top, girdle = nw.new_node(Nodes.ExtrudeMesh, [geometry, top, None, 1e-3]).outputs
        geometry = nw.new_node(Nodes.ScaleElements, [geometry, top, girdle_size])
        geometry, top, _ = nw.new_node(Nodes.ExtrudeMesh, [geometry, top, None, -girdle_height]).outputs
        direction = nw.scale(nw.add(nw.new_node(Nodes.InputNormal), nw.uniform([-perturb] * 3, [perturb] * 3)),
                             nw.uniform(.5 * extrude_height, extrude_height))
        geometry, top, side = nw.new_node(Nodes.ExtrudeMesh, [geometry, top, direction]).outputs
        geometry = nw.new_node(Nodes.ScaleElements, [geometry, top, .2])
        spike = nw.boolean_math('OR', top, side)
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry, 'Spike': spike, 'Girdle': girdle})

    @staticmethod
    def shader_spikes(nw: NodeWrangler, base_hue):
        transmission = uniform(.95, .99)
        subsurface = uniform(.1, .2)
        roughness = uniform(.5, .8)
        color = *colorsys.hsv_to_rgb(base_hue, uniform(.5, 1.), log_uniform(.05, 1.)), 1
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Subsurface': subsurface,
            'Subsurface Color': color,
            'Transmission': transmission
        })
        return principled_bsdf

    @staticmethod
    def shader_girdle(nw: NodeWrangler, base_hue):
        roughness = uniform(.5, .8)
        color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .5), log_uniform(.02, .1)), 1
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                      input_kwargs={'Base Color': color, 'Roughness': roughness})
        return principled_bsdf

    @staticmethod
    def shader_base(nw: NodeWrangler, base_hue):
        roughness = uniform(.5, .8)
        color = *colorsys.hsv_to_rgb(base_hue, uniform(.8, 1.), log_uniform(.01, .02)), 1
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                      input_kwargs={'Base Color': color, 'Roughness': roughness})
        return principled_bsdf

    @staticmethod
    def geo_material_index(nw: NodeWrangler):
        geometry, spike, girdle = nw.new_node(Nodes.GroupInput,
                                              expose_input=[('NodeSocketGeometry', 'Geometry', None),
                                                  ('NodeSocketFloat', 'Spike', None),
                                                  ('NodeSocketFloat', 'Girdle', None)]).outputs[:-1]
        geometry = nw.new_node(Nodes.SetMaterialIndex, [geometry, None, 2])
        geometry = nw.new_node(Nodes.SetMaterialIndex, [geometry, spike, 0])
        geometry = nw.new_node(Nodes.SetMaterialIndex, [geometry, girdle, 1])
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})
