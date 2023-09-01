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

class FruitFactoryBlackberry(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super().__init__(factory_seed, scale=scale, coarse=coarse)
        self.name = 'blackberry'

    def sample_cross_section_params(self, surface_resolution=256):
        return {
            'cross_section_name': "circle_cross_section",
            'cross_section_func_args': {},
            'cross_section_input_args': {'random seed': uniform(-100, 100), 
                'radius': normal(0.9, 0.05),
                'Resolution': surface_resolution},
            'cross_section_output_args': {}
        }

    def sample_shape_params(self, surface_resolution=256):
        return {
            'shape_name': "shape_quadratic",
            'shape_func_args': {'radius_control_points': [(0.0, 0.0), (0.0841, 0.3469), (uniform(0.4, 0.6), 0.8), (0.9432, 0.4781), (1.0, 0.0)]},
            'shape_input_args': {'Profile Curve': 'noderef-crosssection-Geometry', 
                'Start': (uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, -3.0)),
                'End': (0.0, 0.0, 1.0),
                'random seed tilt': uniform(-100, 100),
                'noise amount tilt': 1.0,
                'Resolution': surface_resolution},
            'shape_output_args': {}
        }

    def sample_surface_params(self):
        berry_color = np.array((0.667, 0.254, 0.0))
        berry_color[0] += np.random.normal(0.0, 0.02)
        berry_color[1] += np.random.normal(0.0, 0.05)
        berry_color[2] += np.random.normal(0.0, 0.005)
        berry_color_rgba = hsv2rgba(berry_color)

        return {
            'surface_name': "blackberry_surface",
            'surface_func_args': {'berry_color': berry_color_rgba},
            'surface_input_args': {'Geometry': 'noderef-shapequadratic-Mesh', 
                'spline parameter': 'noderef-shapequadratic-spline parameter'},
            'surface_output_args': {},
            'surface_resolution': 64,
            'scale_multiplier': 0.3
        }

    def sample_stem_params(self):
        stem_color = np.array((0.179, 0.836, 0.318))
        stem_color[0] += np.random.normal(0.0, 0.02)
        stem_color[1] += np.random.normal(0.0, 0.05)
        stem_color[2] += np.random.normal(0.0, 0.05)
        stem_color_rgba = hsv2rgba(stem_color)

        return {
            'stem_name': "basic_stem",
            'stem_func_args': {'stem_color': stem_color_rgba},
            'stem_input_args': {'cross_radius': normal(0.075, 0.005),
                'quad_mid': (uniform(-0.1, 0.1), uniform(-0.1, 0.1), uniform(0.2, 0.3)),
                'quad_end': (uniform(-0.2, 0.2), uniform(-0.2, 0.2), uniform(0.4, 0.6))},
            'stem_output_args': {}
        }