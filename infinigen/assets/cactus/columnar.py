# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import geo_extension
from infinigen.assets.utils.nodegroup import align_tilt
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.assets.cactus.base import BaseCactusFactory
from infinigen.assets.trees.tree import build_radius_tree
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class ColumnarBaseCactusFactory(BaseCactusFactory):
    spike_distance = .08

    @staticmethod
    def radius_fn(base_radius, size, resolution):
        radius_decay = uniform(.5, .8)
        radius_decay_root = uniform(.7, .9)
        leaf_alpha = uniform(2, 3)
        radius = base_radius * radius_decay * np.ones(size * resolution)
        radius[:resolution] *= radius_decay_root ** (1 - np.arange(resolution) / resolution)
        radius[-resolution:] *= (1 - (np.arange(resolution) / resolution) ** leaf_alpha) ** (1 / leaf_alpha)
        return radius

    @property
    def branch_config(self):
        n_major = 16
        n_minor = np.random.randint(10, 14)
        b_minor = np.random.randint(2, 4)
        while True:
            angles = uniform(0, np.pi * 2, b_minor)
            s = np.sort(angles)
            if (np.concatenate([s[1:], [s[0] + np.pi * 2]]) - s > np.pi / 3).all():
                break
        minor_config = {
            'n': b_minor,
            'path_kargs': lambda idx: {
                'n_pts': n_minor,
                'std': .4,
                'momentum': .1,
                'sz': .2,
                'pull_dir': [0, 0, 1],
                'pull_init': 0.,
                'pull_factor': 4.
            },
            'spawn_kargs': lambda idx: {
                'ang_min': np.pi / 2.5,
                'ang_max': np.pi / 2,
                'rng': [.2, .6],
                'axis2': [np.cos(angles[idx]), np.sin(angles[idx]), 0]
            },
            'children': []
        }
        major_config = {
            'n': 1,
            'path_kargs': lambda idx: {'n_pts': n_major, 'std': .4, 'momentum': .99, 'sz': .3},
            'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
            'children': [minor_config]
        }
        return major_config

    def create_asset(self, face_size=.01, **params) -> bpy.types.Object:
        resolution = 16
        base_radius = .25
        obj = build_radius_tree(self.radius_fn, self.branch_config, base_radius, resolution, True)
        surface.add_geomod(obj, self.geo_star, apply=True, input_attributes=[None, 'radius'],
                           attributes=['selection'])
        surface.add_geomod(obj, geo_extension, apply=True, input_kwargs={'musgrave_dimensions': '2D'})
        tag_object(obj, 'columnar_cactus')
        return obj

    @staticmethod
    def geo_star(nw: NodeWrangler):
        perturb = .1
        curve, radius = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Radius', None)]).outputs[:2]
        star_resolution = np.random.randint(5, 8)
        circle = nw.new_node(Nodes.MeshCircle, [star_resolution * 3])
        circle = nw.new_node(Nodes.SetPosition, [circle, None, None, nw.uniform([-perturb] * 3, [perturb] * 3)])
        circle = nw.new_node(Nodes.Transform, [circle], input_kwargs={'Scale': [*uniform(.8, 1., 2), 1]})
        selection = nw.compare('EQUAL', nw.math('MODULO', nw.new_node(Nodes.Index), 2), 0)
        circle, _, selection = nw.new_node(Nodes.CaptureAttribute, [circle, None, selection]).outputs[:3]
        circle = nw.new_node(Nodes.SetPosition, [circle, selection,
            nw.scale(nw.new_node(Nodes.InputPosition), uniform(1.15, 1.25))])
        profile_curve = nw.new_node(Nodes.MeshToCurve, [circle])

        curve = nw.new_node(Nodes.MeshToCurve, [curve])
        curve = align_tilt(nw, curve, noise_strength=uniform(np.pi / 4, np.pi /2))
        curve = nw.new_node(Nodes.SetCurveRadius, [curve, None, radius])
        geometry = nw.curve2mesh(curve, profile_curve)
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry, 'Selection': selection})
