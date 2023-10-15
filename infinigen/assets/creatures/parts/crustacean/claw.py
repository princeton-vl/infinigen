# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform
from scipy.interpolate import interp1d

from infinigen.assets.creatures.util.animation.driver_repeated import bend_bones_lerp
from infinigen.assets.creatures.util.creature import Part
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures.parts.crustacean.leg import CrabLegFactory
from infinigen.assets.creatures.parts.utils.draw import decorate_segment
from infinigen.assets.utils.decorate import displace_vertices, join_objects, read_co, remove_vertices
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.misc import log_uniform
from infinigen.assets.utils.nodegroup import geo_base_selection
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil


class CrabClawFactory(CrabLegFactory):
    tags = ['claw']
    min_spike_radius = .01

    def make_part(self, params) -> Part:
        x_length = params['x_length']
        segments, x_cuts = self.make_segments(params)
        butil.delete(segments[-1])
        claw, lower = self.make_claw(params)
        segments[-1] = claw
        obj = join_objects(segments)
        lower.parent = obj

        skeleton = np.zeros((2, 3))
        skeleton[1, 0] = x_length
        joints = {x: Joint(rest=(0, 0, 0)) for x in x_cuts[1:]}
        return Part(skeleton, obj, joints=joints, settings={'rig_extras': True})

    def make_claw(self, params):
        x_length, y_length, z_length, x_mid, y_mid = map(params.get,
                                                         ['x_length', 'y_length', 'z_length', 'x_mid_second',
                                                             'y_mid_second'])
        xs = x_mid, (x_mid + 1) / 2, (x_mid + 3) / 4, 1
        ys = y_mid, y_mid * params['claw_y_first'], y_mid * params['claw_y_second'], .01
        obj = spin([np.array([xs[0], *xs, xs[-1]]) * x_length, np.array([0, *ys, 0]) * y_length, .0],
                   [1, len(xs)], axis=(1, 0, 0))

        bottom_cutoff = params['bottom_cutoff']
        claw_x_depth = params['claw_x_depth']
        displace_vertices(obj, lambda x, y, z: (0, 0, -np.clip(
            z + y_length * bottom_cutoff + y_length * (y_mid - bottom_cutoff) * (
                    x / x_length - x_mid) / claw_x_depth, None, 0) * (1 - params['bottom_shift'])))
        width_scale = interp1d([x_mid, x_mid + claw_x_depth,
                                   x_mid + claw_x_depth + params['claw_x_turn'] * (1 - x_mid - claw_x_depth),
                                   1], [0, 0, params['claw_z_width'], 0], 'cubic', fill_value='extrapolate')
        displace_vertices(obj, lambda x, y, z: (0, 0,
        np.where(x > (x_mid + claw_x_depth) * x_length, width_scale(x / x_length) * y_mid * y_length, 0)))
        displace_vertices(obj, lambda x, y, z: (0, 0,
        np.where(z > 0, np.clip(params['top_cutoff'] * y_length - np.abs(y), 0, None) * params['top_shift'],
                 0)))
        z = read_co(obj).T[-1]
        write_attr_data(obj, 'ratio', 1 + np.where(z > 0, 0, uniform(.5, 1.) * z / params['y_length']))

        def selection(nw: NodeWrangler):
            x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
            lower = nw.compare('LESS_THAN', nw.separate(nw.new_node(Nodes.InputNormal))[-1], 0)
            x_range = nw.boolean_math('AND',
                                      nw.compare('GREATER_THAN', x, (x_mid + claw_x_depth * 1.5) * x_length),
                                      nw.compare('LESS_THAN', x, x_length * .98))
            center = nw.compare('LESS_THAN', nw.math('ABSOLUTE', y), params['y_length'] * .5)
            return nw.boolean_math('AND', nw.boolean_math('AND', lower, x_range), center)

        temp = butil.spawn_vert('temp')
        surface.add_geomod(temp, geo_base_selection, apply=True,
                           input_args=[obj, selection, params['claw_spike_distance']])
        locations = read_co(temp)
        np.random.shuffle(locations)
        locations = locations[:100]
        butil.delete(temp)
        if len(locations) > 0:
            dist = np.amin(np.linalg.norm(read_co(obj)[np.newaxis] - locations[:, np.newaxis], axis=-1), 0)
            extrude = params['claw_spike_strength'] * np.clip(1 - dist / self.min_spike_radius, 0, None)
            displace_vertices(obj, lambda x, y, z: (0, 0, -extrude))

        decorate_segment(obj, params, x_mid, 1)
        obj.scale[-1] = z_length / y_length
        butil.apply_transform(obj)

        lower_scale = params['lower_scale']
        lower = butil.deep_clone_obj(obj)
        remove_vertices(lower, lambda x, y, z: x < (x_mid + claw_x_depth) * x_length)
        lower.location[0] = -(x_mid + claw_x_depth) * x_length
        butil.apply_transform(lower, loc=True)
        lower.scale = lower_scale, lower_scale, -lower_scale * params['lower_z_scale']
        lower.rotation_euler[1] = uniform(np.pi / 12, np.pi / 4)
        butil.apply_transform(lower)
        lower.location[0] = (x_mid + claw_x_depth) * x_length
        lower.location[-1] = params['lower_z_offset'] * z_length
        butil.apply_transform(lower, loc=True)
        butil.modify_mesh(lower, 'WELD', merge_threshold=.001)
        return obj, lower

    @staticmethod
    def animate_bones(arma, bones, params):
        main_bones = [b for b in bones if 'extra' not in b.name]
        bend_bones_lerp(arma, main_bones, params['claw_curl'], params['freq'], symmetric=False)
        extra_bones = [b for b in bones if 'extra' in b.name]
        bend_bones_lerp(arma, extra_bones, params['claw_lower_curl'], params['freq'], symmetric=False)

    def sample_params(self):
        params = super().sample_params()
        z_length = params['y_length'] * uniform(1, 1.2)
        x_mid_first = uniform(.2, .25)
        x_mid_second = uniform(.4, .6)
        y_mid_first = uniform(1.5, 2.)
        y_mid_second = y_mid_first * log_uniform(1., 1.5)
        y_expand = uniform(1.4, 1.5)
        noise_strength = uniform(.01, .02)
        top_shift = uniform(.6, .8)
        claw_y_first = uniform(.6, 1.5)
        claw_y_second = claw_y_first * uniform(.4, .6)
        claw_x_depth = (1 - x_mid_second) * uniform(.3, .5)
        claw_x_turn = uniform(.2, .4)
        claw_z_width = uniform(.2, .3)
        claw_spike_strength = uniform(.02, .03)
        claw_spike_distance = uniform(.03, .06)
        lower_z_scale = uniform(.4, .6)
        lower_scale = uniform(.75, .9)
        lower_z_offset = uniform(-.5, .5)
        return {**params,
            'z_length': z_length,
            'x_mid_first': x_mid_first,
            'x_mid_second': x_mid_second,
            'y_mid_first': y_mid_first,
            'y_mid_second': y_mid_second,
            'y_expand': y_expand,
            'noise_strength': noise_strength,
            'top_shift': top_shift,
            'claw_y_first': claw_y_first,
            'claw_y_second': claw_y_second,
            'claw_x_depth': claw_x_depth,
            'claw_x_turn': claw_x_turn,
            'claw_z_width': claw_z_width,
            'claw_spike_distance': claw_spike_distance,
            'claw_spike_strength': claw_spike_strength,
            'lower_z_scale': lower_z_scale,
            'lower_scale': lower_scale,
            'lower_z_offset': lower_z_offset,
        }


class LobsterClawFactory(CrabClawFactory):
    def sample_params(self):
        y_expand = uniform(1.4, 1.5)
        y_mid_first = uniform(1.5, 2.)
        y_mid_second = y_mid_first * log_uniform(1.2, 1.6)
        claw_y_first = uniform(1.2, 1.5)
        claw_y_second = claw_y_first * uniform(.7, .8)
        noise_strength = uniform(.01, .02)
        claw_spike_strength = uniform(.01, .02)
        return {**super().sample_params(),
            'y_expand': y_expand,
            'y_mid_first': y_mid_first,
            'y_mid_second': y_mid_second,
            'claw_y_first': claw_y_first,
            'claw_y_second': claw_y_second,
            'noise_strength': noise_strength,
            'claw_spike_strength': claw_spike_strength,
        }
