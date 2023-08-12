# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import assign_material, treeify
from infinigen.assets.utils.nodegroup import geo_base_selection, geo_radius
from infinigen.assets.utils.shortest_path import geo_shortest_path
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.assets.utils.misc import build_color_ramp
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util import blender as butil


def shader_mold(nw: NodeWrangler, base_hue):
    bright_color = *colorsys.hsv_to_rgb((base_hue + uniform(-.04, .04)) % 1, uniform(.8, 1.), .8), 1
    dark_color = *colorsys.hsv_to_rgb(base_hue, uniform(.4, .6), .2), 1

    color = build_color_ramp(nw, nw.musgrave(10), [.0, .3, .7, 1.],
                             [dark_color, dark_color, bright_color, bright_color])
    roughness = .8
    bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': color, 'Roughness': roughness})
    return bsdf


class SlimeMold:

    def __init__(self):
        pass

    def apply(self, obj, selection=None):
        scatter_obj = butil.spawn_vert('scatter:' + 'slime_mold')
        surface.add_geomod(scatter_obj, geo_base_selection, apply=True, input_args=[obj, selection])
        if len(scatter_obj.data.vertices) < 5:
            butil.delete(scatter_obj)
            return

        end_index = lambda nw: nw.build_index_case(np.random.randint(0, len(scatter_obj.data.vertices), 40))
        weight = lambda nw: nw.build_float_curve(nw.new_node(Nodes.InputEdgeAngle).outputs['Signed Angle'],
                                                 [(0, .25), (.2, .4)])

        surface.add_geomod(scatter_obj, geo_shortest_path, apply=True, input_args=[end_index, weight, .1, .02])
        treeify(scatter_obj)
        surface.add_geomod(scatter_obj, geo_radius, apply=True, input_args=[
            lambda nw: nw.build_float_curve(nw.new_node(Nodes.NamedAttribute, ['spline_parameter']),
                                            [(0, .008), (1, .015)]), 6])
        base_hue = uniform(.02, .16)
        assign_material(scatter_obj, shaderfunc_to_material(shader_mold, base_hue))

        return scatter_obj
