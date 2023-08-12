# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import assign_material, displace_vertices, geo_extension, join_objects, \
    subsurface2face_size
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.misc import build_color_ramp, log_uniform
from infinigen.assets.utils.object import data2mesh, mesh2obj
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class MushroomCapFactory(AssetFactory):

    def __init__(self, factory_seed, base_hue, material_func, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_scale, self.z_scale = uniform(.7, 1.4, 2)
            self.cap_configs = [self.campanulate, self.conical, self.convex, self.depressed, self.flat,
                self.infundiuliform, self.ovate, self.umbillicate, self.umbonate]
            config_weights = np.array([2, 2, 2, 1, 2, 1, 2, 1, 1])
            cap_config = np.random.choice(self.cap_configs, p=config_weights / config_weights.sum())
            self.cap_config = {**cap_config,
                'x_anchors': [_ * self.x_scale for _ in cap_config['x_anchors']],
                'z_anchors': [_ * self.z_scale for _ in cap_config['z_anchors']]
            }

            self.radius = max(self.cap_config['x_anchors'])
            self.inner_radius = log_uniform(.2, .35) * self.radius

            self.gill_configs = [self.adnexed_gill, self.decurrent_gill, None]
            gill_configs = np.array([1, 1, 1])
            self.gill_config = np.random.choice(self.gill_configs, p=gill_configs / gill_configs.sum())
            if not self.cap_config['has_gill']:
                self.gill_config = None

            self.shader_funcs = [self.shader_cap, self.shader_noise, self.shader_voronoi, self.shader_speckle]
            shader_weights = np.array([2, 1, 1, 1])
            self.shader_func = np.random.choice(self.shader_funcs, p=shader_weights / shader_weights.sum())

            self.is_morel = uniform(0, 1) < .5 and self.shader_func == self.shader_cap

            self.base_hue = base_hue
            self.material_cap = surface.shaderfunc_to_material(self.shader_func, self.base_hue)
            self.material = material_func()

    @property
    def campanulate(self):
        x = uniform(.12, .15)
        return {
            'x_anchors': [0, x, x, .08, .04, 0],
            'z_anchors': [0, 0, uniform(.03, .05), uniform(.1, .12), uniform(.16, .2), .2],
            'vector_locations': [],
            'has_gill': True
        }

    @property
    def conical(self):
        z = uniform(.2, .3)
        return {
            'x_anchors': [0, uniform(.12, .15), .01, 0],
            'z_anchors': [0, 0, z, z],
            'vector_locations': [1],
            'has_gill': True
        }

    @property
    def convex(self):
        z = uniform(.14, .16)
        return {
            'x_anchors': [0, .15, .12, .01, 0],
            'z_anchors': [0, 0, uniform(.04, .06), z, z],
            'vector_locations': [1],
            'has_gill': True
        }

    @property
    def depressed(self):
        z = uniform(.03, .05)
        return {
            'x_anchors': [0, .15, .12, 0],
            'z_anchors': [0, 0, uniform(.06, .08), z],
            'vector_locations': [1],
            'has_gill': True
        }

    @property
    def flat(self):
        z = uniform(.05, .07)
        return {
            'x_anchors': [0, .15, .12, 0],
            'z_anchors': [0, 0, z, z],
            'vector_locations': [1],
            'has_gill': True
        }

    @property
    def infundiuliform(self):
        z = uniform(.08, .12)
        x = uniform(.12, .15)
        return {
            'x_anchors': [0, .03, x, x - .01, 0],
            'z_anchors': [0, 0, z, z + uniform(.005, .01), .02],
            'vector_locations': [],
            'has_gill': False
        }

    @property
    def ovate(self):
        z = uniform(.2, .3)
        return {
            'x_anchors': [0, uniform(.12, .15), .08, .01, 0],
            'z_anchors': [0, 0, .8 * z, z, z],
            'vector_locations': [1],
            'has_gill': True
        }

    @property
    def umbillicate(self):
        z = uniform(.03, .05)
        return {
            'x_anchors': [0, .15, .12, .02, 0],
            'z_anchors': [0, .04, uniform(.06, .08), z + .02, z],
            'vector_locations': [],
            'has_gill': False
        }

    @property
    def umbonate(self):
        z = uniform(.05, .07)
        z_ = z + uniform(.02, .04)
        return {
            'x_anchors': [0, .15, .12, .06, .02, 0],
            'z_anchors': [0, 0, z - .01, z, z_, z_],
            'vector_locations': [1],
            'has_gill': True
        }

    @property
    def adnexed_gill(self):
        return {
            'x_anchors': [self.radius, (self.radius + self.inner_radius) / 2, self.inner_radius,
                self.inner_radius, self.radius],
            'z_anchors': [0, -uniform(.05, .08), -uniform(0, .02), 0, 0],
            'vector_locations': [3]
        }

    @property
    def decurrent_gill(self):
        return {
            'x_anchors': [self.radius, (self.radius + self.inner_radius) / 2, self.inner_radius, 0,
                self.radius],
            'z_anchors': [0, -uniform(.05, .08), -uniform(.08, .1), 0, 0],
            'vector_locations': [2]
        }

    @staticmethod
    def geo_xyz(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        for name, component in zip('xyz', nw.separate(nw.new_node(Nodes.InputPosition))):
            component = nw.math('ABSOLUTE', component)
            m = nw.new_node(Nodes.AttributeStatistic, [geometry, None, component]).outputs['Max']
            geometry = nw.new_node(Nodes.StoreNamedAttribute,
                input_kwargs={
                    'Geometry': geometry, 
                    'Name': name, 
                    'Value': nw.scalar_divide(component, m)
                })
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    @staticmethod
    def geo_morel(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        selection = nw.compare('LESS_THAN', nw.new_node(Nodes.VoronoiTexture, input_kwargs={
            'Scale': uniform(15, 20),
            'Randomness': uniform(.5, 1)
        }, attrs={'feature': 'DISTANCE_TO_EDGE'}), .05)
        geometry = nw.new_node(Nodes.StoreNamedAttribute, 
            input_kwargs={'Geometry':geometry, 'Name':'morel', 'Value': selection})
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

    def apply_cut(self, obj):
        if max(self.cap_config['x_anchors']) > .1:
            return
        n_cuts = np.random.randint(0, 5)
        angles = polygon_angles(n_cuts, np.pi / 4, np.pi * 2)
        for a in angles:
            width = uniform(.15, .2) * .4
            vertices = [[0, 0, .4], [.4, -width, .4], [.4, width, .4], [0, 0, -1], [.4, -width, -.01],
                [.4, width, -.01]]
            faces = [[0, 1, 2], [1, 0, 3, 4], [2, 1, 4, 5], [0, 2, 5, 3], [5, 4, 3]]
            cutter = mesh2obj(data2mesh(vertices, [], faces))
            displace_vertices(cutter, lambda x, y, z: (0, 2 * y * y, 0))
            butil.modify_mesh(cutter, 'SUBSURF', render_levels=5, levels=5, subdivision_type='SIMPLE')
            depth = self.radius * uniform(.4, .7)
            cutter.location = np.cos(a) * depth, np.sin(a) * depth, 0
            cutter.rotation_euler = 0, 0, a + uniform(-np.pi / 4, np.pi / 4)
            butil.modify_mesh(obj, 'WELD', merge_threshold=.002)
            butil.modify_mesh(obj, 'BOOLEAN', object=cutter, operation='DIFFERENCE', apply=True)
            butil.delete(cutter)

    def create_asset(self, face_size, **params) -> bpy.types.Object:
        cap_config = self.cap_config
        anchors = cap_config['x_anchors'], 0, cap_config['z_anchors']
        obj = spin(anchors, cap_config['vector_locations'])
        self.apply_cut(obj)
        remesh_with_attrs(obj, face_size)
        surface.add_geomod(obj, self.geo_xyz, apply=True)
        surface.add_geomod(obj, self.geo_morel, apply=True)
        assign_material(obj, self.material_cap)

        if self.is_morel:
            obj.data.attributes.active = obj.data.attributes['morel']
            bpy.ops.geometry.attribute_convert(mode='VERTEX_GROUP')
            butil.modify_mesh(obj, 'DISPLACE', vertex_group='morel', strength=.04, mid_level=.7)

        if self.gill_config is not None:
            gill_config = self.gill_config
            anchors = gill_config['x_anchors'], 0, gill_config['z_anchors']
            gill = spin(anchors, gill_config['vector_locations'], dupli=True, loop=True,
                        resolution=np.random.randint(8, 20))
            subsurface2face_size(gill, face_size)
            assign_material(gill, self.material)
            obj = join_objects([obj, gill])

        texture = bpy.data.textures.new(name='cap', type=np.random.choice(['STUCCI', 'MARBLE']))
        texture.noise_scale = log_uniform(.01, .05)
        butil.modify_mesh(obj, 'DISPLACE', strength=.008, texture=texture, mid_level=0)

        surface.add_geomod(obj, geo_extension, apply=True, input_args=[.1])
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='TWIST', angle=uniform(-np.pi / 4, np.pi / 4),
                          deform_axis='X')
        r1, r2, r3, r4 = uniform(-.25, .25, 4)
        displace_vertices(obj, lambda x, y, z: (np.where(x > 0, r1, r2) * x, np.where(y > 0, r3, r4) * y, 0))
        tag_object(obj, 'cap')
        return obj

    @staticmethod
    def shader_voronoi(nw: NodeWrangler, base_hue):
        bright_color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .8), log_uniform(.05, .2)), 1
        dark_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.4, .8),
                                          log_uniform(.01, .05)), 1
        subsurface_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.4, .8),
                                                log_uniform(.05, .2)), 1
        light_color = *colorsys.hsv_to_rgb(base_hue, uniform(0, .1), uniform(.2, .8)), 1
        anchors = [.0, .3, .6, 1.] if uniform(0, 1) < .5 else [.0, .4, .7, 1.]
        color = build_color_ramp(nw, nw.musgrave(500), anchors,
                                 [dark_color, dark_color, bright_color, bright_color])

        x = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'x'}).outputs['Fac']
        y = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'y'}).outputs['Fac']
        r = nw.power(nw.add(nw.power(x, 2), nw.power(y, 2)), .5)
        coord = nw.scale(nw.combine(x, y, 0), nw.build_float_curve(r, [(0, 1), (uniform(.5, .7), 2), (1, 8)]))

        perturbed_position = nw.add(coord,
                                    nw.scale(nw.new_node(Nodes.NoiseTexture, attrs={'noise_dimensions': '2D'}),
                                             .2))
        voronoi = nw.new_node(Nodes.VoronoiTexture,
                              input_kwargs={'Scale': uniform(2, 2.5), 'Vector': perturbed_position},
                              attrs={'voronoi_dimensions': '2D', 'feature': 'DISTANCE_TO_EDGE'})

        ratio = nw.divide(voronoi, nw.scalar_add(1, nw.scalar_multiply(5, nw.power(r, 2))))
        ratio = nw.build_float_curve(ratio, [(0, .4), (.04, 0)])
        ratio = nw.scalar_multiply(ratio, nw.new_node(Nodes.MapRange, [
            nw.new_node(Nodes.MusgraveTexture, input_kwargs={'Scale': 20}), -.2, .1, 0, 1]))
        color = nw.new_node(Nodes.MixRGB, [ratio, color, light_color])

        roughness = uniform(.2, .5) if uniform(0, 1) < .5 else uniform(.8, 1.)
        specular = uniform(.2, .8)
        clearcoat = uniform(.2, .5) if uniform(0, 1) < .25 else 0
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Specular': specular,
            'Clearcoat': clearcoat,
            'Subsurface Color': subsurface_color,
            'Subsurface': .01,
            'Subsurface Radius': (.05, .05, .05)
        })
        return principled_bsdf

    @staticmethod
    def shader_speckle(nw: NodeWrangler, base_hue):
        bright_color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .8), log_uniform(.05, .2)), 1
        dark_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.4, .8),
                                          log_uniform(.01, .05)), 1
        subsurface_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.4, .8),
                                                log_uniform(.05, .2)), 1
        light_color = *colorsys.hsv_to_rgb(base_hue, uniform(0, .1), uniform(.2, .8)), 1
        anchors = [.0, .3, .6, 1.] if uniform(0, 1) < .5 else [.0, .4, .7, 1.]
        color = build_color_ramp(nw, nw.musgrave(500), anchors,
                                 [dark_color, dark_color, bright_color, bright_color])

        musgrave = nw.build_float_curve(nw.musgrave(50), [(.7, 0), (.72, 1.)])
        color = nw.new_node(Nodes.MixRGB, [musgrave, color, light_color])

        roughness = uniform(.2, .5) if uniform(0, 1) < .5 else uniform(.8, 1.)
        specular = uniform(.2, .8)
        clearcoat = uniform(.2, .5) if uniform(0, 1) < .25 else 0
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Specular': specular,
            'Clearcoat': clearcoat,
            'Subsurface Color': subsurface_color,
            'Subsurface': .01,
            'Subsurface Radius': (.05, .05, .05)
        })
        return principled_bsdf

    @staticmethod
    def shader_noise(nw: NodeWrangler, base_hue):
        bright_color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .8), log_uniform(.05, .2)), 1
        dark_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.4, .8),
                                          log_uniform(.01, .05)), 1
        subsurface_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.4, .8),
                                                log_uniform(.05, .2)), 1
        light_color = *colorsys.hsv_to_rgb(base_hue, uniform(0, .1), uniform(.2, .8)), 1
        anchors = [.0, .3, .6, 1.] if uniform(0, 1) < .5 else [.0, .4, .7, 1.]
        color = build_color_ramp(nw, nw.musgrave(500), anchors,
                                 [dark_color, dark_color, bright_color, bright_color])

        ratio = nw.build_float_curve(nw.musgrave(10), [(.52, 0), (.56, .2), (.6, 0.)])
        ratio = nw.scalar_multiply(ratio, nw.new_node(Nodes.MapRange, [
            nw.new_node(Nodes.MusgraveTexture, input_kwargs={'Scale': 20}), -.2, .1, 0, 1]))
        color = nw.new_node(Nodes.MixRGB, [ratio, color, light_color])

        roughness = uniform(.2, .5) if uniform(0, 1) < .5 else uniform(.8, 1.)
        specular = uniform(.2, .8)
        clearcoat = uniform(.2, .5) if uniform(0, 1) < .25 else 0
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Specular': specular,
            'Clearcoat': clearcoat,
            'Subsurface Color': subsurface_color,
            'Subsurface': .01,
            'Subsurface Radius': (.05, .05, .05)
        })
        return principled_bsdf

    @staticmethod
    def shader_cap(nw: NodeWrangler, base_hue):
        bright_color = *colorsys.hsv_to_rgb(base_hue, uniform(.6, .8), log_uniform(.05, .2)), 1
        dark_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.4, .8),
                                          log_uniform(.01, .05)), 1
        light_color = *colorsys.hsv_to_rgb(base_hue, uniform(0, .1), uniform(.6, .8)), 1
        subsurface_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.05, .05)) % 1, uniform(.6, .8),
                                                log_uniform(.05, .2)), 1

        anchors = [.0, .3, .6, 1.] if uniform(0, 1) < .5 else [.0, .4, .7, 1.]
        color = build_color_ramp(nw, nw.musgrave(500), anchors,
                                 [dark_color, dark_color, bright_color, bright_color])

        z = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'z'})
        musgrave = nw.build_float_curve(z, [(uniform(0, .2), uniform(.95, .98)),
            (uniform(.2, .4), uniform(.98, 1)), (.8, 1)])
        color = nw.new_node(Nodes.MixRGB, [musgrave, light_color, color])

        roughness = uniform(.2, .5) if uniform(0, 1) < .5 else uniform(.8, 1.)
        specular = uniform(.2, .8)
        clearcoat = uniform(.2, .5) if uniform(0, 1) < .25 else 0
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Specular': specular,
            'Clearcoat': clearcoat,
            'Subsurface Color': subsurface_color,
            'Subsurface': .01,
            'Subsurface Radius': (.05, .05, .05)
        })
        return principled_bsdf
