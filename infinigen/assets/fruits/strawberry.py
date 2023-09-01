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

class FruitFactoryStrawberry(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super().__init__(factory_seed, scale=scale, coarse=coarse)
        self.name = 'strawberry'

    def sample_cross_section_params(self, surface_resolution=256):
        return {
            'cross_section_name': "circle_cross_section",
            'cross_section_func_args': {},
            'cross_section_input_args': {'random seed': uniform(-100, 100), 
                'radius': normal(1.0, 0.02),
                'Resolution': surface_resolution},
            'cross_section_output_args': {}
        }

    def sample_shape_params(self, surface_resolution=256):
        return {
            'shape_name': "shape_quadratic",
            'shape_func_args': {'radius_control_points': [(0.0, 0.0), (0.0227, 0.1313), (0.2227, 0.4406), (uniform(0.55, 0.7), uniform(0.7, 0.78)), (0.925, 0.4719), (1.0, 0.0)]},
            'shape_input_args': {'Profile Curve': 'noderef-crosssection-Geometry', 
                'Start': (uniform(-0.2, 0.2), uniform(-0.2, 0.2), uniform(-0.5, -1.0)),
                'End': (0.0, 0.0, 1.0),
                'random seed pos': uniform(-100, 100),
                'Resolution': surface_resolution},
            'shape_output_args': {}
        }

    def sample_surface_params(self):
        main_color = np.array((0.0, 0.995, 0.85))
        main_color[0] += np.random.normal(0.0, 0.02)
        main_color[1] += np.random.normal(0.0, 0.05)
        main_color[2] += np.random.normal(0.0, 0.05)
        main_color_rgba = hsv2rgba(main_color)

        top_color = np.array((0.15, 0.75, 0.75))
        top_color[0] += np.random.normal(0.0, 0.02)
        top_color[1] += np.random.normal(0.0, 0.05)
        top_color[2] += np.random.normal(0.0, 0.05)
        top_color_rgba = hsv2rgba(top_color)

        return {
            'surface_name': "strawberry_surface",
            'surface_func_args': {'top_pos': uniform(0.85, 0.95),
                'main_color': main_color_rgba,
                'top_color': top_color_rgba},
            'surface_input_args': {'Geometry': 'noderef-shapequadratic-Mesh', 
                'spline parameter': 'noderef-shapequadratic-spline parameter',
                'Distance Min': 0.15, 
                'Strength': 1.5,
                'noise random seed': uniform(-100, 100)},
            'surface_output_args': {'strawberry seed height': 'noderef-fruitsurface-curve parameters'},
            'surface_resolution': 64,
            'scale_multiplier': 0.5
        }

    def sample_stem_params(self):
        stem_color = np.array((0.28, 0.91, 0.45))
        stem_color[0] += np.random.normal(0.0, 0.02)
        stem_color[1] += np.random.normal(0.0, 0.05)
        stem_color[2] += np.random.normal(0.0, 0.05)
        stem_color_rgba = hsv2rgba(stem_color)

        stem_color = np.array((0.28, 0.91, 0.45))
        stem_color[0] += np.random.normal(0.0, 0.02)
        stem_color[1] += np.random.normal(0.0, 0.05)
        stem_color[2] += np.random.normal(0.0, 0.05)
        stem_color_rgba = hsv2rgba(stem_color)
        
        return {
            'stem_name': "calyx_stem",
            'stem_func_args': {'stem_color': stem_color_rgba},
            'stem_input_args': {'Geometry': 'noderef-fruitsurface-Geometry',
                'fork number': randint(8, 13),
                'outer radius': uniform(0.7, 0.9),
                'noise random seed': uniform(-100, 100),
                'quad_mid': (uniform(-0.1, 0.1), uniform(-0.1, 0.1), uniform(0.15, 0.2)),
                'quad_end': (uniform(-0.2, 0.2), uniform(-0.2, 0.2), uniform(0.3, 0.4)),
                'cross_radius': uniform(0.035, 0.045),
                'Translation': (0.0, 0.0, 0.97)},
            'stem_output_args': {}
        }