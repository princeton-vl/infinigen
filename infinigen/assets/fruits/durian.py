# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils

import gin
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

@gin.register
class FruitFactoryDurian(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super().__init__(factory_seed, scale=scale, coarse=coarse)
        self.name = 'durian'

    def sample_cross_section_params(self, surface_resolution=256):
        return {
            'cross_section_name': "circle_cross_section",
            'cross_section_func_args': {},
            'cross_section_input_args': {'random seed': uniform(-100, 100),
                'radius': normal(1.2, 0.03),
                'Resolution': surface_resolution},
            'cross_section_output_args': {}
        }

    def sample_shape_params(self, surface_resolution=256):
        return {
            'shape_name': "shape_quadratic",
            'shape_func_args': {'radius_control_points': [(0.0, 0.0031), (0.0841, 0.3469), (uniform(0.4, 0.6), 0.8), (0.8886, 0.6094), (1.0, 0.0)]},
            'shape_input_args': {'Profile Curve': 'noderef-crosssection-Geometry', 
                'noise amount tilt': 5.0,
                'noise scale tilt': 0.5,
                'random seed tilt': uniform(-100, 100),
                'Resolution': surface_resolution,
                'Start': (uniform(-0.3, 0.3), uniform(-0.3, 0.3), uniform(-0.5, -1.5)),
                'End': (0.0, 0.0, 1.0)},
            'shape_output_args': {}
        }

    def sample_surface_params(self):
        base_color = np.array((0.15, 0.74, 0.32))
        base_color[0] += np.random.normal(0.0, 0.02)
        base_color[1] += np.random.normal(0.0, 0.05)
        base_color[2] += np.random.normal(0.0, 0.05)
        base_color_rgba = hsv2rgba(base_color)

        peak_color = np.array((0.09, 0.87, 0.24))
        peak_color[0] += np.random.normal(0.0, 0.025)
        peak_color[1] += np.random.normal(0.0, 0.05)
        peak_color[2] += np.random.normal(0.0, 0.05)
        peak_color_rgba = hsv2rgba(peak_color)

        return {
            'surface_name': "durian_surface",
            'surface_func_args': {'thorn_control_points': [(0.0, 0.0), (0.7318, 0.4344), (1.0, 1.0)],
                'peak_color': peak_color_rgba, 
                'base_color': base_color_rgba
                },
            'surface_input_args': {
                'Geometry': 'noderef-shapequadratic-Mesh', 
                'spline parameter': 'noderef-shapequadratic-spline parameter', 
                'distance Min': uniform(0.07, 0.13),
                'displacement': uniform(0.25, 0.35),
                'noise amount': 0.2
                },
            'surface_output_args': {'durian thorn coordiante': 'noderef-fruitsurface-distance to center'},
            'surface_resolution': 512,
            'scale_multiplier': 2.0
        }

    def sample_stem_params(self):
        stem_color = np.array((0.10, 0.96, 0.13))
        stem_color[0] += np.random.normal(0.0, 0.02)
        stem_color[1] += np.random.normal(0.0, 0.05)
        stem_color[2] += np.random.normal(0.0, 0.05)
        stem_color_rgba = hsv2rgba(stem_color)

        return {
            'stem_name': "basic_stem",
            'stem_func_args': {'stem_color': stem_color_rgba},
            'stem_input_args': {
                'cross_radius': uniform(0.07, 0.09),
                'quad_mid': (uniform(-0.1, 0.1), uniform(-0.1, 0.1), uniform(0.15, 0.2)),
                'quad_end': (uniform(-0.2, 0.2), uniform(-0.2, 0.2), uniform(0.3, 0.4)),
                'Translation': (0.0, 0.0, 0.9)
            },
            'stem_output_args': {}
        }