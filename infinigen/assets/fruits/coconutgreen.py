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

class FruitFactoryCoconutgreen(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super().__init__(factory_seed, scale=scale, coarse=coarse)
        self.name = 'coconut_green'
        
    def sample_cross_section_params(self, surface_resolution=256):
        rad_small = uniform(0.65, 0.75)

        return {
            'cross_section_name': "coconut_cross_section",
            'cross_section_func_args': {'control_points': [(0.0, rad_small), (0.1, rad_small), (1.0, 0.76)]},
            'cross_section_input_args': {'random seed': uniform(-100, 100), 
                'radius': normal(1.8, 0.1),
                'noise scale': 20.0, 
                'noise amount': 0.02,
                'Resolution': surface_resolution},
            'cross_section_output_args': {'crosssection_coordinate': 'noderef-crosssection-curve parameters'}
        }

    def sample_shape_params(self, surface_resolution=256):
        return {
            'shape_name': "shape_quadratic",
            'shape_func_args': {'radius_control_points': [(0.0, 0.0), (0.0591, 0.3156), (uniform(0.2, 0.3), 0.6125), (uniform(0.6, 0.7), 0.675), (0.9636, 0.3625), (1.0, 0.0)]},
            'shape_input_args': {'Profile Curve': 'noderef-crosssection-Geometry', 
                'Start': (uniform(-0.1, 0.1), uniform(-0.1, 0.1), normal(-1.0, 0.1)),
                'End': (0.0, 0.0, 1.0),
                'Resolution': surface_resolution},
            'shape_output_args': {'shape_coordinate': 'noderef-shapequadratic-spline parameter'}
        }

    def sample_surface_params(self):
        bottom_color = np.array((0.282, 0.951, 0.266))
        bottom_color[0] += np.random.normal(0.0, 0.02)
        bottom_color[1] += np.random.normal(0.0, 0.05)
        bottom_color[2] += np.random.normal(0.0, 0.05)
        bottom_color_rgba = hsv2rgba(bottom_color)

        basic_color = np.array((0.235, 0.989, 0.413))
        basic_color[0] += np.random.normal(0.0, 0.025)
        basic_color[1] += np.random.normal(0.0, 0.05)
        basic_color[2] += np.random.normal(0.0, 0.05)
        basic_color_rgba = hsv2rgba(basic_color)

        return {
            'surface_name': "coconutgreen_surface",
            'surface_func_args': {'basic_color': basic_color_rgba, 'bottom_color': bottom_color_rgba},
            'surface_input_args': {'Geometry': 'noderef-shapequadratic-Mesh', 
                'spline parameter': 'noderef-shapequadratic-spline parameter',
                'spline tangent': 'noderef-shapequadratic-spline tangent',
                'distance to center': 'noderef-shapequadratic-radius to center',
                'cross section paramater': 'noderef-crosssection-curve parameters',
                },
            'surface_output_args': {},
            'surface_resolution': 256,
            'scale_multiplier': 1.5
        }

    def sample_stem_params(self):
        bottom_color = np.array((0.282, 0.951, 0.266))
        bottom_color[0] += np.random.normal(0.0, 0.02)
        bottom_color[1] += np.random.normal(0.0, 0.05)
        bottom_color[2] += np.random.normal(0.0, 0.05)
        bottom_color_rgba = hsv2rgba(bottom_color)

        calyx_edge_color = np.array((0.039, 0.96, 0.037))
        calyx_edge_color[0] += np.random.normal(0.0, 0.02)
        calyx_edge_color[1] += np.random.normal(0.0, 0.05)
        calyx_edge_color[2] += np.random.normal(0.0, 0.05)
        calyx_edge_color_rgba = hsv2rgba(calyx_edge_color)

        stem_x = uniform(-0.4, 0.4)
        stem_y = uniform(-0.4, 0.4)

        return {
            'stem_name': "coconut_stem",
            'stem_func_args': {'basic_color': bottom_color_rgba, 'edge_color': calyx_edge_color_rgba},
            'stem_input_args': {
                'Target': 'noderef-fruitsurface-Geometry',
                'radius': 0.001, 
                'calyx width': uniform(0.2, 0.25),
                'Count': randint(4, 6),
                'stem_radius': normal(0.04, 0.005),
                'stem_mid': (stem_x, stem_y, 0.0),
                'stem_end': (2*stem_x, 2*stem_y, uniform(0.3, 0.5)),
            },
            'stem_output_args': {'distance to edge': 'noderef-stem-distance to edge'}
        }