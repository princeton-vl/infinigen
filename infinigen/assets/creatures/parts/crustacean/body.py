# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform
from scipy.interpolate import interp1d

from infinigen.assets.creatures.util.creature import Part, PartFactory
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures.parts.utils.draw import geo_symmetric_texture
from infinigen.assets.utils.decorate import add_distance_to_boundary, displace_vertices, join_objects, read_co, write_co
from infinigen.assets.utils.draw import leaf, spin
from infinigen.assets.utils.misc import log_uniform
from infinigen.assets.utils.object import new_line
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.placement import placeholder_locs
from infinigen.core import surface
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil


class CrabBodyFactory(PartFactory):
    tags = ['body']
    min_spike_distance = .1
    min_spike_radius = .02

    def make_part(self, params) -> Part:
        x_length, x_tip, bend_height = map(params.get, ['x_length', 'x_tip', 'bend_height'])
        upper = self.make_surface(params)
        lower = butil.deep_clone_obj(upper)
        self.make_surface_side(upper, params, 'upper')
        self.make_surface_side(lower, params, 'lower')
        self.add_spikes(upper, params)
        self.add_mouth(lower, params)
        obj = join_objects([upper, lower])

        x, y, z = read_co(obj).T
        write_attr_data(obj, 'ratio', np.where(z > np.min(z) * params['color_cutoff'], 1, 0))
        butil.modify_mesh(obj, 'WELD', merge_threshold=.001)

        height_scale = interp1d([0, -x_tip + .01, -x_tip - .01, -1], [0, bend_height, bend_height, 0],
                                'quadratic', fill_value="extrapolate")
        displace_vertices(obj, lambda x, y, z: (0, 0, height_scale(x / x_length)))
        self.add_head(obj, params)

        line = new_line(x_length)
        line.location[0] -= x_length
        butil.apply_transform(line, loc=True)

        line.rotation_euler[1] = np.pi / 2
        butil.apply_transform(line)
        butil.modify_mesh(line, 'SIMPLE_DEFORM', deform_method='BEND', angle=-params['bend_angle'],
                          deform_axis='Y')
        line.rotation_euler[1] = -np.pi / 2
        butil.apply_transform(line)
        skeleton = read_co(line)
        butil.delete(line)

        obj.rotation_euler[1] = np.pi / 2
        butil.apply_transform(obj)
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=-params['bend_angle'],
                          deform_axis='Y')
        obj.rotation_euler[1] = -np.pi / 2
        butil.apply_transform(obj)
        joints = {i: Joint((0, 0, 0), bounds=np.array([[0, 0, 0], [0, 0, 0]])) for i in
            np.linspace(0, 1, 5, endpoint=True)}
        return Part(skeleton, obj, joints=joints)

    def add_head(self, obj, params):
        def offset(nw: NodeWrangler, vector):
            head = nw.scalar_add(1, nw.scalar_divide(nw.separate(nw.new_node(Nodes.InputPosition))[0],
                                                     params['x_length']))
            texture = nw.new_node(Nodes.MusgraveTexture, [vector],
                                  input_kwargs={'Scale': params['noise_scale']})
            return nw.combine(nw.scalar_multiply(head, nw.scalar_multiply(texture, params['noise_strength'])),
                              0, 0)

        surface.add_geomod(obj, geo_symmetric_texture, input_args=[offset], apply=True)

    @staticmethod
    def make_surface(params):
        x_length, y_length, x_tip, y_tail = map(params.get, ['x_length', 'y_length', 'x_tip', 'y_tail'])
        x_anchors = np.array([0, 0, -x_tip / 2, -x_tip, -x_tip, -x_tip, -(x_tip + 1) / 2, -1, -1]) * x_length
        y_anchors = np.array(
            [0, .1, params['front_midpoint'], 1, 1, 1, params['back_midpoint'], y_tail, 0]) * y_length
        tip_size = params['tip_size']
        if params['has_sharp_tip']:
            front_angle = params['front_angle']
            back_angle = params['back_angle']
            x_anchors[3] += tip_size * np.sin(front_angle) * x_length
            x_anchors[5] -= tip_size * np.sin(back_angle) * x_length
            y_anchors[3] += tip_size * (1 - np.cos(front_angle)) * x_length
            y_anchors[4] += tip_size * x_length
            y_anchors[5] += tip_size * (1 - np.cos(back_angle)) * x_length
            vector_locations = [4]
        else:
            x_anchors[3] += .05 * x_tip * x_length
            x_anchors[5] -= .05 * (1 - x_tip) * x_length
            vector_locations = []
        obj = leaf(x_anchors, y_anchors, vector_locations)
        butil.modify_mesh(obj, 'SUBSURF', levels=1, render_levels=1)
        add_distance_to_boundary(obj)
        return obj

    def make_surface_side(self, obj, params, prefix="upper"):
        vg = obj.vertex_groups['distance']
        distance = np.array([vg.weight(i) for i in range(len(obj.data.vertices))])
        height_scale = interp1d([0, .5, 1], [0, params[f'{prefix}_alpha'], 1], 'quadratic')
        displace_vertices(obj, lambda x, y, z: (
            0, 0, (1 if prefix == 'upper' else -1) * height_scale(distance) * params[f'{prefix}_z']))
        displace_vertices(obj, lambda x, y, z: (params[f'{prefix}_shift'] * z, 0, 0))
        offset = lambda nw, vector, distance: nw.combine(0, 0, nw.scalar_multiply(distance, nw.scalar_multiply(
            nw.new_node(Nodes.MusgraveTexture, [vector], input_kwargs={'Scale': params['noise_scale']
            }), params[f'noise_strength'])))
        surface.add_geomod(obj, geo_symmetric_texture, input_args=[offset], apply=True)
        return obj

    def add_spikes(self, obj, params):
        def selection(nw: NodeWrangler):
            x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
            return nw.boolean_math('AND', nw.compare('GREATER_THAN', y, 0), nw.compare('GREATER_THAN', z, .02))

        locations = placeholder_locs(obj, params['spike_density'], selection, self.min_spike_distance, 0)
        locations_ = locations.copy()
        locations_[:, 1] = -locations_[:, 1]
        locations = np.concatenate([locations, locations_], 0)
        if len(locations) == 0: return
        x, y, z = read_co(obj).T
        dist = np.amin(np.linalg.norm(read_co(obj)[np.newaxis] - locations[:, np.newaxis], axis=-1), 0)
        extrude = params['spike_height'] * np.clip(1 - dist / self.min_spike_radius, 0, None)
        d = np.stack([x + params['spike_center'] * params['x_length'], y, z + params['spike_depth']], -1)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        displace_vertices(obj, lambda x, y, z: (d * extrude[:, np.newaxis]).T)

    def add_mouth(self, obj, params):
        def selection(nw: NodeWrangler):
            x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
            z_length = params['lower_z'] if 'lower_z' in params else params['z_length']
            z_range = nw.boolean_math('AND', nw.compare('GREATER_THAN', z, -params['mouth_z'] * z_length),
                                      nw.compare('LESS_THAN', z, 0))
            x_range = nw.compare('GREATER_THAN', x, -params['mouth_x'] * params['x_length'])
            return nw.boolean_math('AND', z_range, x_range)

        def offset(nw: NodeWrangler, vector, distance):
            wave_texture = nw.new_node(Nodes.WaveTexture, [vector], input_kwargs={
                'Scale': params['mouth_noise_scale'],
                'Distortion': 20,
                'Detail': 0
            })
            ratio = nw.scalar_multiply(distance,
                                       nw.build_float_curve(distance, [(0, 0), (.001, 0), (.005, 1), (1, 1)]))
            return nw.scale(
                nw.scalar_multiply(ratio, nw.scalar_multiply(wave_texture, params['mouth_noise_strength'])),
                nw.new_node(Nodes.InputNormal))

        surface.add_geomod(obj, geo_symmetric_texture, input_args=[offset, selection], apply=True)

    def sample_params(self):
        x_length = uniform(.8, 1.2)
        y_length = x_length * uniform(.5, .7)
        x_tip = uniform(.3, .6)
        y_tail = uniform(.1, .3)
        has_sharp_tip = uniform(0, 1) < .4
        front_midpoint = uniform(.7, .9)
        back_midpoint = uniform(.7, .9)
        front_angle = uniform(np.pi / 12, np.pi / 8)
        back_angle = uniform(np.pi / 6, np.pi / 4)
        tip_size = uniform(.05, .15)
        upper_z = x_length * uniform(.15, .3)
        upper_alpha = uniform(.8, .9)
        upper_shift = uniform(-.6, -.4)
        noise_strength = uniform(.02, .03)
        noise_scale = uniform(8, 15)
        lower_alpha = uniform(.96, .98)
        lower_z = x_length * uniform(.3, .4)
        lower_shift = uniform(.1, .2)
        spike_height = uniform(.05, .2) if uniform(0, 1) < .5 else 0
        spike_depth = log_uniform(.4, 2)
        spike_center = uniform(.3, .7)
        spike_density = log_uniform(100, 500)
        mouth_z = uniform(.5, .8)
        mouth_x = uniform(.1, .15)
        mouth_noise_scale = uniform(10, 15)
        mouth_noise_strength = uniform(.1, .2)
        bend_angle = uniform(0, np.pi / 3)
        bend_height = uniform(.08, .12)
        color_cutoff = uniform(0, .5)
        return {
            'x_length': x_length,
            'y_length': y_length,
            'x_tip': x_tip,
            'y_tail': y_tail,
            'has_sharp_tip': has_sharp_tip,
            'front_midpoint': front_midpoint,
            'back_midpoint': back_midpoint,
            'front_angle': front_angle,
            'back_angle': back_angle,
            'tip_size': tip_size,
            'upper_z': upper_z,
            'upper_alpha': upper_alpha,
            'upper_shift': upper_shift,
            'noise_strength': noise_strength,
            'noise_scale': noise_scale,
            'lower_z': lower_z,
            'lower_alpha': lower_alpha,
            'lower_shift': lower_shift,
            'spike_height': spike_height,
            'spike_depth': spike_depth,
            'spike_density': spike_density,
            'spike_center': spike_center,
            'mouth_z': mouth_z,
            'mouth_x': mouth_x,
            'mouth_noise_scale': mouth_noise_scale,
            'mouth_noise_strength': mouth_noise_strength,
            'bend_angle': bend_angle,
            'bend_height': bend_height,
            'color_cutoff': color_cutoff,
        }


class LobsterBodyFactory(CrabBodyFactory):
    tags = ['body']
    min_spike_distance = .08
    min_spike_radius = .01

    def make_part(self, params) -> Part:
        x_length, y_length, z_length = map(params.get, ['x_length', 'y_length', 'z_length'])
        x_anchors = np.array([0, 0, 1 / 3, 2 / 3, 1, 1]) * x_length
        y_anchors = np.array([0, 1, params['midpoint_second'], params['midpoint_first'], .01, 0]) * y_length
        obj = spin([x_anchors, y_anchors, 0], [1, 4], axis=(1, 0, 0))
        self.add_mouth(obj, params)

        height_fn = interp1d([0, 1 / 2, 1], [0, params['z_shift_midpoint'] / 2, params['z_shift']],
                             fill_value='extrapolate')
        displace_vertices(obj, lambda x, y, z: (0, 0, height_fn(x / x_length) * y_length))

        z = read_co(obj).T[-1]
        write_attr_data(obj, 'ratio', 1 + np.where(z > 0, 0, uniform(1, 1.5) * z / y_length))
        displace_vertices(obj, lambda x, y, z: (
            0, 0, -np.clip(z + y_length * params['bottom_cutoff'], None, 0) * (1 - params['bottom_shift'])))

        obj.scale[-1] = z_length / y_length
        butil.apply_transform(obj)

        offset = lambda nw, vector: nw.scale(nw.scalar_multiply(
            nw.new_node(Nodes.MusgraveTexture, [vector], input_kwargs={'Scale': params['noise_scale']
            }), params[f'noise_strength']), nw.new_node(Nodes.InputNormal))
        surface.add_geomod(obj, geo_symmetric_texture, input_args=[offset], apply=True)

        n_segments = 4
        co = read_co(obj)
        skeleton = np.zeros((n_segments, 3))
        skeleton[:, 0] = np.linspace(0, x_length, n_segments)
        head_z = co[np.argmax(co[:, 0])][-1]
        skeleton[:, -1] = np.linspace(0, head_z, n_segments)
        return Part(skeleton, obj)

    def sample_params(self):
        x_length = uniform(.6, .8)
        y_length = uniform(.15, .2)
        z_length = y_length * uniform(1, 1.2)
        midpoint_first = uniform(.65, .75)
        midpoint_second = uniform(.95, 1.05)
        z_shift = uniform(.4, .6)
        z_shift_midpoint = uniform(.2, .3)
        noise_strength = uniform(.02, .04)
        noise_scale = uniform(5, 8)
        bottom_shift = uniform(.3, .5)
        bottom_cutoff = uniform(.2, .3)
        mouth_z = uniform(.5, .8)
        mouth_x = uniform(.1, .15) - 1
        mouth_noise_scale = uniform(10, 15)
        mouth_noise_strength = uniform(.2, .3)
        return {
            'x_length': x_length,
            'y_length': y_length,
            'z_length': z_length,
            'midpoint_first': midpoint_first,
            'midpoint_second': midpoint_second,
            'z_shift': z_shift,
            'z_shift_midpoint': z_shift_midpoint,
            'noise_strength': noise_strength,
            'noise_scale': noise_scale,
            'bottom_shift': bottom_shift,
            'bottom_cutoff': bottom_cutoff,
            'mouth_z': mouth_z,
            'mouth_x': mouth_x,
            'mouth_noise_scale': mouth_noise_scale,
            'mouth_noise_strength': mouth_noise_strength,
        }
