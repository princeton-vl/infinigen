# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei



import colorsys

import bmesh
import numpy as np
import bpy
from mathutils import Vector
from numpy.random import uniform
from scipy.interpolate import interp1d

from infinigen.assets.creatures.util.animation.driver_repeated import repeated_driver
from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.nodegroup import geo_base_selection
from infinigen.assets.utils.object import data2mesh, mesh2obj, new_circle, new_empty, new_icosphere, origin2highest
from infinigen.assets.utils.decorate import assign_material, geo_extension, join_objects, read_co, remove_vertices, \
    subsurface2face_size, write_attribute, write_co
from infinigen.assets.utils.misc import log_uniform, make_circular, make_circular_angle
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core import surface
from infinigen.core.surface import read_attr_data, shaderfunc_to_material, write_attr_data
import infinigen.core.util.blender as butil

from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class JellyfishFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.base_hue = np.random.normal(0.57, 0.15)
            self.outside_material = self.make_transparent() if uniform(0, 1) < .8 else self.make_dotted()
            self.inside_material = self.make_transparent() if uniform(0, 1) < .8 else self.make_opaque()
            self.tentacle_material = self.make_transparent()
            self.arm_mat_transparent = self.make_transparent()
            self.arm_mat_opaque = self.make_opaque()
            self.arm_mat_solid = self.make_solid()

            self.has_arm = uniform(0, 1) < .5
            arm_radius = uniform(0, .3)
            self.arm_radius_range = arm_radius, arm_radius + uniform(.1, .4)
            self.arm_height_range = -uniform(.4, .5), -uniform(0, .2)
            self.arm_min_distance = uniform(.06, .08)
            self.arm_size = uniform(.03, .06)
            self.arm_length = log_uniform(2, 5)
            self.arm_bend_angle = uniform(0, np.pi / 60)
            self.arm_displace_range = uniform(0, .4), uniform(.4, .8)

            self.tentacle_min_distance = uniform(.04, .06)
            self.tentacle_size = uniform(.005, .01)
            self.tentacle_length = log_uniform(1.5, 2.5)
            self.tentacle_bend_angle = uniform(0, np.pi / 12)

            self.cap_thickness = uniform(.05, .6)
            self.cap_inner_radius = uniform(.6, .8)
            self.cap_z_scale = log_uniform(.4, 1.5)
            self.cap_dent = uniform(.15, .3) if uniform(0, 1) < .5 else 0

            self.length_scale = log_uniform(.25, 2.)
            self.anim_freq = 1 / log_uniform(25, 100)
            self.move_freq = 1 / log_uniform(500, 1000)

    def create_asset(self, face_size, **params):
        obj, radius = self.build_cap(face_size)

        assign_material(obj, [self.outside_material, self.inside_material])
        for axis in 'XY':
            butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='TWIST', angle=uniform(-np.pi / 3, np.pi / 3),
                              deform_axis=axis)
        for axis in 'XY':
            butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=uniform(-np.pi / 3, np.pi / 3),
                              deform_axis=axis)

        def selection(nw: NodeWrangler):
            x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
            r = nw.math('POWER', nw.add(nw.math('POWER', x, 2), nw.math('POWER', y, 2)), .5)
            center = nw.boolean_math('AND', nw.compare('GREATER_THAN', r, self.arm_radius_range[0] * radius),
                                     nw.compare('LESS_THAN', r, self.arm_radius_range[1] * radius))
            down = nw.compare('LESS_THAN', nw.separate(nw.new_node(Nodes.InputNormal))[-1], 0)
            inside = nw.new_node(Nodes.NamedAttribute, ["inside"])
            return nw.boolean_math('AND', nw.boolean_math('AND', center, down), inside)

        if self.has_arm:
            long_arms = self.place_tentacles(obj, selection, self.arm_min_distance, self.arm_size,
                                             self.arm_length, self.arm_bend_angle, displace=True)
            for a in long_arms:
                assign_material(a, np.random.choice(
                    [self.arm_mat_opaque, self.arm_mat_transparent, self.arm_mat_solid]))
        else:
            long_arms = []

        tentacles = self.place_tentacles(obj, 'boundary', self.tentacle_min_distance, self.tentacle_size,
                                         self.tentacle_length, self.tentacle_bend_angle)
        assign_material(tentacles, self.tentacle_material)

        obj = join_objects([obj] + long_arms + tentacles)
        head_z = np.amax(read_co(obj)[:, -1])
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
        driver_z.expression = repeated_driver(uniform(-1.5, -.5), uniform(.5, 1.5), self.move_freq, offset,
                                              seed)
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

    def place_tentacles(self, obj, selection, min_distance, size, length, bend_angle, displace=False):
        temp = butil.spawn_vert('temp')
        surface.add_geomod(temp, geo_base_selection, apply=True, input_args=[obj, selection, min_distance])
        locations = read_co(temp)
        if displace:
            locations[:, -1] -= uniform(*self.arm_displace_range, len(locations))
        butil.delete(temp)
        n = min(10, len(locations))
        arms = [self.build_arm(size, length, bend_angle) for _ in range(n)]
        arms += [deep_clone_obj(np.random.choice(arms)) for _ in range(len(locations) - n)]
        for arm, loc in zip(arms, locations):
            arm.rotation_euler[-1] = np.arctan2(loc[1], loc[0]) + uniform(-np.pi / 6, np.pi / 6) + np.pi
            arm.location = loc
        return arms

    def build_cap(self, face_size):
        obj = new_icosphere(subdivisions=6)
        write_attribute(obj, lambda nw, position: 0, 'material_index', 'FACE')

        d = np.sqrt(1 - self.cap_inner_radius ** 2) + 1 - self.cap_thickness
        r = (d * d + self.cap_inner_radius ** 2) / (2 * d)

        cutter = new_icosphere(subdivisions=6, radius=r)
        write_attribute(cutter, lambda nw, position: 1, 'material_index', 'FACE')
        cutter.location[-1] = 1 - self.cap_thickness - r
        butil.modify_mesh(obj, 'BOOLEAN', object=cutter, operation='DIFFERENCE')
        co = read_co(obj)
        outside = np.abs(np.linalg.norm(co, axis=-1) - 1) < 1e-6
        co[:, -1] -= cutter.location[-1]
        inside = np.abs(np.linalg.norm(co, axis=-1) - r) < 1e-6
        write_attr_data(obj, 'inside', inside.astype(float))
        write_attr_data(obj, 'boundary', ((~inside) & (~outside)).astype(float))
        butil.delete(cutter)

        if self.cap_dent > 0:
            self.apply_cap_dent(obj)

        surface.add_geomod(obj, geo_extension, apply=True,
                           input_args=[log_uniform(.2, .4), log_uniform(.5, 1.), '2D'])
        obj.scale *= Vector(uniform(.4, .6, 3))
        obj.scale[-1] *= self.cap_z_scale
        radius = self.cap_inner_radius * min(obj.scale[:2])
        butil.apply_transform(obj)
        subsurface2face_size(obj, face_size)

        obj.vertex_groups.new(name='pin')
        tag_object(obj, 'cap')
        return obj, radius

    def apply_cap_dent(self, obj):
        n_dent = np.random.randint(6, 12)
        angles = polygon_angles(n_dent)
        angles = np.concatenate([angles, angles + 2 * np.pi])
        dent = uniform(1 - self.cap_dent, 1, n_dent)
        margin = uniform(np.pi * .02, np.pi * .05, n_dent)
        x, y, z = read_co(obj).T
        a = np.arctan2(y, x) + np.pi * 1.5
        difference = np.abs(a[:, np.newaxis] - angles[np.newaxis, :])
        index = np.argmin(difference, 1) % n_dent
        dent_ = np.take(dent, index)
        margin_ = np.take(margin, index)
        s = np.exp(np.log(dent_) / margin_ * np.clip(margin_ - np.min(difference, 1), 0, None))
        co = np.stack([s * x, s * y, z]).T
        write_co(obj, co)

    def build_arm(self, radius, length, bend_angle):
        obj = new_circle(vertices=16)
        obj.scale = radius, radius * uniform(0, 1), 1
        butil.apply_transform(obj)
        remove_vertices(obj, lambda x, y, z: y * (-1) ** np.random.randint(2) > 0)
        steps = 256

        empty = new_empty(location=(0, 0, 1), rotation=(0, -uniform(0, np.pi / 24), 0))
        butil.modify_mesh(obj, 'SCREW', angle=log_uniform(.5, 3) * np.pi * (-1) ** int(uniform(0, 1)),
                          screw_offset=-length * self.length_scale * uniform(.5, 1.), object=empty, steps=steps,
                          render_steps=steps)
        butil.delete(empty)
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='TAPER', factor=uniform(.5, 1.), deform_axis='Z')
        texture = bpy.data.textures.new(name='arm', type='MARBLE')
        texture.noise_scale = log_uniform(.1, .2)
        butil.modify_mesh(obj, 'DISPLACE', texture=texture, strength=uniform(.01, .02), direction='Y')
        texture = bpy.data.textures.new(name='arm', type='MARBLE')
        texture.noise_scale = log_uniform(.1, 2.)
        butil.modify_mesh(obj, 'DISPLACE', texture=texture, strength=log_uniform(.1, .2), direction='X')
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=bend_angle * log_uniform(.5, 1.5),
                          deform_axis='Y')
        co = read_co(obj)
        x, y, z = co.T
        center = np.mean(co[z > -.01], 0)
        obj.location[0] -= center[0]
        obj.location[1] -= center[1]
        butil.apply_transform(obj, loc=True)
        tag_object(obj, 'arm')
        return obj

    @staticmethod
    def shader_jellyfish(nw: NodeWrangler, base_hue, saturation, transparency):
        layerweight = nw.build_float_curve(nw.new_node(Nodes.LayerWeight, input_kwargs={'Blend': 0.3}),
                                       [(0, 0), (.4, 0), (uniform(.6, .9), 1), (1, 1)])
        emission_color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .6), 1), 1
        transparent_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.1, .1)) % 1, saturation, 1), 1
        emission = nw.new_node(Nodes.Emission, [emission_color])
        glossy = nw.new_node(Nodes.GlossyBSDF, 
            input_kwargs={'Color': transparent_color, 'Roughness': uniform(0.8, 1)})
        transparent = nw.new_node(Nodes.TransparentBSDF, [transparent_color])
        mix_shader = nw.new_node(Nodes.MixShader, [0.5, glossy, transparent])
        mix_shader = nw.new_node(Nodes.MixShader, [layerweight, emission, mix_shader])
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
            return nw.build_float_curve(
                nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': uniform(20, 50)}),
                [(0, uniform(.92, .96)), (.62, uniform(.92, .96)), (.65, uniform(.5, .6)),
                    (1, uniform(.5, .6))])

        hue = (self.base_hue + uniform(-.1, .1)) % 1
        return shaderfunc_to_material(self.shader_jellyfish, hue, uniform(.5, .8), transparency)
