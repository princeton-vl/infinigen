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

class FruitFactoryApple(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super().__init__(factory_seed, scale=scale, coarse=coarse)
        self.name = 'apple'

    def sample_cross_section_params(self, surface_resolution=256):
        return {
            'cross_section_name': "circle_cross_section",
            'cross_section_func_args': {},
            'cross_section_input_args': {'random seed': uniform(-100, 100),
                'radius': normal(1.5, 0.05),
                'Resolution': surface_resolution},
            'cross_section_output_args': {}
        }

    def sample_shape_params(self, surface_resolution=256):
        return {
            'shape_name': "shape_quadratic",
            'shape_func_args': {'radius_control_points': [(0.0, 0.0), (0.1227, 0.4281), (0.4705, 0.6625), (0.8886, 0.4156), (1.0, 0.0)],},
            'shape_input_args': {'Profile Curve': 'noderef-crosssection-Geometry', 
                'noise amount tilt': 0.0,
                'noise scale pos': 0.5,
                'noise amount pos': 0.1,
                'Resolution': surface_resolution,
                'Start': (uniform(-0.1, 0.1), uniform(-0.1, 0.1), uniform(-0.9, -1.1)),
                'End': (0.0, 0.0, 1.0)},
            'shape_output_args': {}
        }

    def sample_surface_params(self):
        base_color = np.array((uniform(-0.05, 0.1), 0.999, 0.799))
        base_color[1] += normal(0.0, 0.05)
        base_color[2] += normal(0.0, 0.05)
        base_color_rgba = hsv2rgba(base_color)

        alt_color = np.copy(base_color)
        alt_color[0] += normal(0.05, 0.02)
        alt_color[1] += normal(0.0, 0.05)
        alt_color[2] += normal(0.0, 0.05)
        alt_color_rgba = hsv2rgba(alt_color)

        return {
            'surface_name': "apple_surface",
            'surface_func_args': {'color1': base_color_rgba, 
                'color2': alt_color_rgba,
                'random_seed': uniform(-100, 100)},
            'surface_input_args': {'Geometry': 'noderef-shapequadratic-Mesh', 
                'spline parameter': 'noderef-shapequadratic-spline parameter', 
                'spline tangent': 'noderef-shapequadratic-spline tangent', 
                'distance to center': 'noderef-shapequadratic-radius to center'},
            'surface_output_args': {},
            'surface_resolution': 64,
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
                'quad_res': 32, 
                'cross_radius': uniform(0.025, 0.035),
                'cross_res': 32, 
                'Translation': (0.0, 0.0, 0.6)},
            'stem_output_args': {}
        }





