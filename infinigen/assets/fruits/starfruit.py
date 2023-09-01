# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface

from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil
from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.fruits.general_fruit import FruitFactoryGeneralFruit

class FruitFactoryStarfruit(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super().__init__(factory_seed, scale=scale, coarse=coarse)
        self.name = 'starfruit'

    def sample_cross_section_params(self, surface_resolution=256):
        return {
            'cross_section_name': "star_cross_section",
            'cross_section_func_args': {},
            'cross_section_input_args': {'random seed': uniform(-100, 100),
                'radius': normal(1.3, 0.05),
                'Resolution': surface_resolution},
            'cross_section_output_args': {'star parameters': 'noderef-crosssection-curve parameters'}
        }

    def sample_shape_params(self, surface_resolution=256):
        return {
            'shape_name': "shape_quadratic",
            'shape_func_args': {'radius_control_points': [(0.0727, 0.2), (0.2636, 0.6063), (uniform(0.45, 0.65), uniform(0.7, 0.9)), (0.8886, 0.6094), (1.0, 0.0)],},
            'shape_input_args': {'Profile Curve': 'noderef-crosssection-Geometry', 
                'Resolution': surface_resolution,
                'Start': (uniform(-0.3, 0.3), uniform(-0.3, 0.3), uniform(-1.0, -2.0)),
                'End': (0.0, 0.0, 1.0)},
            'shape_output_args': {}
        }

    def sample_surface_params(self):
        base_color = np.array((0.10, 0.999, 0.799))
        base_color[0] += normal(0.0, 0.025)
        base_color[1] += normal(0.0, 0.05)
        base_color[2] += normal(0.0, 0.005)
        base_color_rgba = hsv2rgba(base_color)

        ridge_color = np.copy(base_color)
        ridge_color[0] += normal(0.04, 0.02)
        ridge_color[2] += normal(-0.2, 0.02)
        ridge_color_rgba = hsv2rgba(ridge_color)

        return {
            'surface_name': "starfruit_surface",
            'surface_func_args': {
                'dent_control_points': [(0.0, 0.4219), (0.0977, 0.4469), (0.2273, 0.4844), (0.5568, 0.5125), (1.0, 0.5)],
                'base_color': base_color_rgba, 
                'ridge_color': ridge_color_rgba},
            'surface_input_args': {'Geometry': 'noderef-shapequadratic-Mesh', 
                'spline parameter': 'noderef-shapequadratic-spline parameter', 
                'spline tangent': 'noderef-shapequadratic-spline tangent', 
                'distance to center': 'noderef-shapequadratic-radius to center',
                'dent intensity': normal(1.0, 0.1)
                },
            'surface_output_args': {},
            'surface_resolution': 256,
            'scale_multiplier': 1.0
        }

    def sample_stem_params(self):
        stem_color = np.array((0.10, 0.96, 0.13))
        stem_color[0] += normal(0.0, 0.02)
        stem_color[1] += normal(0.0, 0.05)
        stem_color[2] += normal(0.0, 0.05)
        stem_color_rgba = hsv2rgba(stem_color)

        return {
            'stem_name': "basic_stem",
            'stem_func_args': {'stem_color': stem_color_rgba},
            'stem_input_args': {'quad_mid': (uniform(-0.1, 0.1), uniform(-0.1, 0.1), uniform(0.15, 0.2)),
                'quad_end': (uniform(-0.2, 0.2), uniform(-0.2, 0.2), uniform(0.3, 0.4)),
                'cross_radius': uniform(0.03, 0.05),
                'Translation': (0.0, 0.0, 0.8)},
            'stem_output_args': {}
        }