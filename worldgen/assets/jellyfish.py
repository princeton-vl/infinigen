# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 


import colorsys

import bmesh
import numpy as np
import bpy
from numpy.random import uniform
from scipy.interpolate import interp1d

import util.blender as butil
from assets.creatures.animation.driver_repeated import repeated_driver
from assets.utils.nodegroup import geo_base_selection
from assets.utils.object import data2mesh, mesh2obj, new_circle, new_empty, new_icosphere, origin2highest
from assets.utils.decorate import assign_material, geo_extension, join_objects, read_co, remove_vertices, \
    subsurface2face_size, write_attribute
from assets.utils.misc import log_uniform
from nodes.node_info import Nodes
from nodes.node_wrangler import NodeWrangler
from placement.factory import AssetFactory
from surfaces import surface
from surfaces.surface import read_attr_data, shaderfunc_to_material, write_attr_data

from util.blender import deep_clone_obj
from util.math import FixedSeed
from assets.utils.tag import tag_object, tag_nodegroup

class JellyfishFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.base_hue = np.random.normal(0.57, 0.15)
            self.outside_material = self.make_transparent() if uniform(0, 1) < .8 else self.make_dotted()
            self.inside_material = self.make_transparent() if uniform(0, 1) < .8 else self.make_opaque()
            self.short_material = self.make_transparent()
            self.long_mat_transparent = self.make_transparent()
            self.long_mat_opaque = self.make_opaque()
            self.long_mat_solid = self.make_solid()
            long_mat_prob = uniform(0, 1, 3)
            self.long_mat_prob = long_mat_prob / long_mat_prob.sum()

            self.opaque_arm_prob = uniform(.2, .5)
            self.anim_freq = 1 / log_uniform(25, 100)
            self.move_freq = 1 / log_uniform(500, 1000)

    def create_asset(self, face_size, **params):
        obj, radius = self.build_cap(face_size)
        assign_material(obj, [self.outside_material, self.inside_material])
        length_scale = uniform(.75, 1.5)
        placement_radius = radius * uniform(.4, .6)

        def selection(nw: NodeWrangler):
            x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
            center = nw.compare('LESS_THAN', nw.add(nw.math('POWER', x, 2), nw.math('POWER', y, 2)),
                                placement_radius ** 2)
            down = nw.compare('LESS_THAN', nw.separate(nw.new_node(Nodes.InputNormal))[-1], 0)
            return nw.boolean_math('AND', center, down)

        long_arms = self.place_tentacles(obj, selection, uniform(.04, .06), uniform(.03, .06),
                                         log_uniform(2, 5) * length_scale, uniform(0, np.pi / 60))
        long_material_index = np.random.choice([0, 1, 2], len(long_arms), p=self.long_mat_prob)
        for i, mat in enumerate([self.long_mat_opaque, self.long_mat_transparent, self.long_mat_solid]):
            assign_material([a for a, o in zip(long_arms, long_material_index) if o == i], mat)

        short_arms = self.place_tentacles(obj, 'boundary', uniform(.04, .06), uniform(.005, .01),
                                          log_uniform(1.5, 2.5) * length_scale, uniform(0, np.pi / 12))
        assign_material(short_arms, self.short_material)

        head_z = np.amax(read_co(obj)[:, -1])
        obj = join_objects([obj] + long_arms + short_arms)
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=uniform(-np.pi / 12, np.pi / 12),
                          deform_axis='Y')
        tail_z = -np.amin(read_co(obj)[:, -1])
        self.animate_expansion(obj, head_z, tail_z)
        self.animate_movement(obj)

        tag_object(obj, 'jellyfish')

        return obj

    def animate_movement(self, obj):
        offset = uniform(0, 1)
        seed = np.random.randint(1e5)
        driver_x, driver_y, driver_z = [_.driver for _ in obj.driver_add('location')]
        driver_x.expression = repeated_driver(uniform(-.2, .2), uniform(-.2, .2), self.move_freq, offset, seed)
        driver_y.expression = repeated_driver(uniform(-.2, .2), uniform(-.2, .2), self.move_freq, offset, seed)
        driver_z.expression = repeated_driver(- uniform(0, -1), uniform(0, 1), self.move_freq, offset, seed)
        driver_rot = obj.driver_add('rotation_euler')[-1].driver
        twist_range = uniform(0, np.pi / 60)
        driver_rot.expression = repeated_driver(-twist_range, twist_range, self.move_freq, offset, seed)

        obj, mod = butil.modify_mesh(obj, 'SIMPLE_DEFORM', False, deform_method='TWIST', deform_axis='Z',
                                     return_mod=True)
        twist_driver = mod.driver_add('angle').driver
        twist_driver.expression = repeated_driver(-np.pi / 30, np.pi / 30, self.move_freq, offset, seed)

    def animate_expansion(self, obj, head_z, tail_z):
        obj.shape_key_add(name='Base')
        offset = uniform(0, 1)
        seed = np.random.randint(1e5)
        self.animate_radius(obj, offset, seed, head_z, tail_z)
        self.animate_height(obj, offset, seed, head_z, tail_z)
        self.animate_arms(obj, tail_z)

    def animate_height(self, obj, offset, seed, head_z, tail_z):
        x, y, z = read_co(obj).T
        obj.active_shape_key_index = 0
        key_block_z = obj.shape_key_add(name='Height')
        z_anchors = -tail_z, 0, head_z
        z_disp = 1, 1, uniform(.6, .8)
        z_curve = interp1d(z_anchors, z_disp, fill_value='extrapolate')
        co = np.stack([x, y, z_curve(z) * z], -1)
        key_block_z.data.foreach_set('co', co.reshape(-1))
        dr = key_block_z.driver_add('value').driver
        dr.expression = repeated_driver(0, 1, self.anim_freq, offset + uniform(.05, .15), seed)

    def animate_radius(self, obj, offset, seed, head_z, tail_z):
        obj.active_shape_key_index = 0
        x, y, z = read_co(obj).T
        key_block_r = obj.shape_key_add(name='Radius')
        z_anchors = -tail_z, -head_z * 2, -head_z, 0, head_z
        r_scale = uniform(.7, .9), uniform(.85, .95), 1, uniform(1.2, 1.4), 1
        r_curve = interp1d(z_anchors, r_scale, 'quadratic', fill_value='extrapolate')
        co = np.stack([r_curve(z) * x, r_curve(z) * y, z], -1)
        key_block_r.data.foreach_set('co', co.reshape(-1))
        dr = key_block_r.driver_add('value').driver
        dr.expression = repeated_driver(0, 1, self.anim_freq, offset, seed)

    def animate_arms(self, obj, tail_z):
        def geo_musgrave_texture(nw: NodeWrangler, axis):
            geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
            z = nw.separate(nw.new_node(Nodes.InputPosition))[-1]
            musgrave = nw.new_node(Nodes.MusgraveTexture, input_kwargs={'Scale': uniform(1, 2)},
                                   attrs={'musgrave_dimensions': '2D'})
            offset = nw.scalar_multiply(log_uniform(.1, .4), nw.new_node(Nodes.CombineXYZ, input_kwargs={
                axis: nw.scalar_divide(nw.scalar_multiply(musgrave, z), -tail_z)
            }))
            geometry = nw.new_node(Nodes.SetPosition, [geometry,
                nw.boolean_math('NOT', nw.new_node(Nodes.NamedAttribute, ['pin'])), None, offset])
            nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})

        for i, axis in enumerate('XY'):
            obj.active_shape_key_index = 0
            key_block_r = obj.shape_key_add(name=f'Arm_{i}')
            temp = deep_clone_obj(obj)
            temp.shape_key_clear()
            surface.add_geomod(temp, geo_musgrave_texture, apply=True, input_args=[axis])
            key_block_r.data.foreach_set('co', read_co(temp).reshape(-1))
            butil.delete(temp)
            dr = key_block_r.driver_add('value').driver
            dr.expression = repeated_driver(0, 1, self.anim_freq)

    def place_tentacles(self, obj, selection, min_distance, size, length, bend_angle):
        temp = butil.spawn_vert('temp')
        surface.add_geomod(temp, geo_base_selection, apply=True, input_args=[obj, selection, min_distance])
        locations = read_co(temp)
        locations[:, -1] -= uniform(0, .05, len(locations))
        butil.delete(temp)
        n = min(10, len(locations))
        arms = [self.build_arm(size, length, bend_angle) for _ in range(n)]
        arms += [deep_clone_obj(np.random.choice(arms)) for _ in range(len(locations) - n)]
        for arm, loc in zip(arms, locations):
            arm.rotation_euler[-1] = np.arctan2(loc[1], loc[0]) + uniform(-np.pi / 10, np.pi / 10) + np.pi
            arm.location = loc
        return arms

    @staticmethod
    def build_cap(face_size):
        obj = new_icosphere(subdivisions=6)
        write_attribute(obj, lambda nw, position: 0, 'material_index', 'FACE')

        thickness = uniform(.05, .6)
        radius = uniform(.6, .9)
        d = np.sqrt(1 - radius * radius) + 1 - thickness
        r = (d * d + radius * radius) / (2 * d)

        cutter = new_icosphere(subdivisions=6, radius=r)
        write_attribute(cutter, lambda nw, position: 1, 'material_index', 'FACE')
        cutter.location[-1] = 1 - thickness - r
        butil.modify_mesh(obj, 'BOOLEAN', object=cutter, operation='DIFFERENCE')
        co = read_co(obj)
        outside = np.abs(np.linalg.norm(co, axis=-1) - 1) < 1e-6
        co[:, -1] -= cutter.location[-1]
        inside = np.abs(np.linalg.norm(co, axis=-1) - r) < 1e-6
        write_attr_data(obj, 'boundary', ((~inside) & (~outside)).astype(float))
        butil.delete(cutter)

        surface.add_geomod(obj, geo_extension, apply=True,
                           input_args=[uniform(.1, .25), uniform(1., 1.5), '2D'])
        obj.scale = *uniform(.4, .5, 2), log_uniform(.4, .6)
        radius *= min(obj.scale[:2])
        butil.apply_transform(obj)
        subsurface2face_size(obj, face_size)
        obj.vertex_groups.new(name='pin')
        tag_object(obj, 'cap')
        return obj, radius

    @staticmethod
    def build_arm(radius, length, bend_angle):
        obj = new_circle()
        obj.scale = radius, radius * uniform(0, 1), 1
        butil.apply_transform(obj)
        remove_vertices(obj, lambda x, y, z: y * (-1) ** np.random.randint(2) > 0)
        steps = 512

        empty = new_empty(location=(0, 0, 1), rotation=(0, -uniform(0, np.pi / 24), 0))
        butil.modify_mesh(obj, 'SCREW', angle=log_uniform(.5, 3) * np.pi * (-1) ** int(uniform(0, 1)),
                          screw_offset=-length * uniform(.5, 1.), object=empty, steps=steps, render_steps=steps)
        butil.delete(empty)
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='TAPER', factor=uniform(.5, 1.), deform_axis='Z')
        texture = bpy.data.textures.new(name='arm', type='MARBLE')
        texture.noise_scale = log_uniform(.1, .2)
        butil.modify_mesh(obj, 'DISPLACE', texture=texture, strength=uniform(.01, .02), direction='Y')
        texture = bpy.data.textures.new(name='arm', type='MARBLE')
        texture.noise_scale = log_uniform(.1, 2.)
        butil.modify_mesh(obj, 'DISPLACE', texture=texture, strength=log_uniform(.1, .2), direction='X')
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=bend_angle * uniform(.6, 1.5),
                          deform_axis='Y')
        co = read_co(obj)
        i = np.argmax(co[:, 1])
        obj.location[1] = -min(co[i, 1], 0)
        butil.apply_transform(obj, loc=True)
        tag_object(obj, 'arm')
        return obj

    @staticmethod
    def shader_jellyfish(nw: NodeWrangler, base_hue, saturation, transparency):
        fresnel = nw.build_float_curve(nw.new_node(Nodes.Fresnel),
                                       [(0, 0), (.4, 0), (uniform(.6, .9), 1), (1, 1)])
        emission_color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .6), 1), 1
        transparent_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.1, .1)) % 1, saturation, 1), 1
        emission = nw.new_node(Nodes.Emission, [emission_color])
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Transmission': 1.})
        mix_shader = nw.new_node(Nodes.MixShader, [fresnel, emission, principled_bsdf])
        transparent = nw.new_node(Nodes.TransparentBSDF, [transparent_color])
        transparency = surface.eval_argument(nw, transparency)
        mix_shader = nw.new_node(Nodes.MixShader, [transparency, mix_shader, transparent])
        return mix_shader

    def make_transparent(self):
        hue = (self.base_hue + uniform(-.1, .1)) % 1
        return shaderfunc_to_material(self.shader_jellyfish, hue, uniform(.1, .3), uniform(.88, .92))

    def make_opaque(self):
        hue = (self.base_hue + uniform(-.1, .1)) % 1
        return shaderfunc_to_material(self.shader_jellyfish, hue, uniform(.3, .6), uniform(.75, .8))

    def make_solid(self):
        hue = (self.base_hue + uniform(-.1, .1)) % 1
        return shaderfunc_to_material(self.shader_jellyfish, hue, uniform(.5, .8), uniform(.4, .5))

    def make_dotted(self):
        def transparency(nw: NodeWrangler):
            return nw.build_float_curve(nw.musgrave(uniform(20, 50)),
                                        [(0, uniform(.92, .96)), (.62, uniform(.92, .96)),
                                            (.65, uniform(.5, .6)), (1, uniform(.5, .6))])

        hue = (self.base_hue + uniform(-.1, .1)) % 1
        return shaderfunc_to_material(self.shader_jellyfish, hue, uniform(.5, .8), transparency)
