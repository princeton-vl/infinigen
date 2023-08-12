# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np

from infinigen.assets.utils.decorate import displace_vertices
from infinigen.assets.utils.draw import spin
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util import blender as butil


def make_segments(x_cuts, y_cuts, x_anchors, y_anchors, params):
    x_length, y_length, z_length = map(params.get, ['x_length', 'y_length', 'z_length'])
    segments = []
    for i in range(len(x_cuts) - 1):
        x_start, x_end = x_cuts[i], x_cuts[i + 1]
        y_start, y_end = y_cuts[i], y_cuts[i + 1]
        xs = x_anchors(x_start, x_end)
        ys = y_anchors(y_start, y_end)
        obj = spin([np.array([xs[0], *xs, xs[-1]]) * x_length, np.array([0, *ys, 0]) * y_length, .0],
                   [1, len(xs)], axis=(1, 0, 0))

        y_base = y_length * y_start
        displace_vertices(obj, lambda x, y, z: (
            0, 0, -np.clip(z + y_base * params['bottom_cutoff'], None, 0) * (1 - params['bottom_shift'])))
        displace_vertices(obj, lambda x, y, z: (0, 0,
        np.where(z > 0, np.clip(params['top_cutoff'] * y_base - np.abs(y), 0, None) * params['top_shift'], 0)))

        decorate_segment(obj, params, x_start, x_end)
        obj.scale[-1] = params['z_length'] / y_length
        butil.apply_transform(obj)
        segments.append(obj)
    return segments


def decorate_segment(obj, params, x_start, x_end):
    def offset(nw: NodeWrangler, vector):
        noise_texture = nw.new_node(Nodes.NoiseTexture, [vector], input_kwargs={'Scale': params['noise_scale']})
        x = nw.separate(nw.new_node(Nodes.InputPosition))[0]
        ratio = nw.build_float_curve(nw.scalar_divide(x, params['x_length']),
                                     [(x_start, 1), (x_end - .01, 1), (x_end, 0), (x_end + .01, 0)])
        return nw.scale(nw.scalar_multiply(ratio, nw.scalar_multiply(noise_texture, params['noise_strength'])),
                        nw.new_node(Nodes.InputNormal))

    surface.add_geomod(obj, geo_symmetric_texture, input_args=[offset], apply=True)
    butil.modify_mesh(obj, 'WELD', merge_threshold=.001)


def geo_symmetric_texture(nw: NodeWrangler, offset, selection=None):
    geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    pos = nw.new_node(Nodes.InputPosition)
    x, y, z = nw.separate(pos)
    vector = nw.combine(x, nw.math('ABSOLUTE', y), z)
    distance = nw.new_node(Nodes.NamedAttribute, ['distance'])
    geometry = nw.new_node(Nodes.SetPosition, [geometry, surface.eval_argument(nw, selection), None,
        surface.eval_argument(nw, offset, vector=vector, distance=distance)])
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})
