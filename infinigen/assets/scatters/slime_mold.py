# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.mesh import treeify
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.nodegroup import geo_base_selection, geo_radius
from infinigen.assets.utils.shortest_path import geo_shortest_path
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba


def shader_mold(nw: NodeWrangler, base_hue):
    bright_color = hsv2rgba(
        (base_hue + uniform(-0.04, 0.04)) % 1, uniform(0.8, 1.0), 0.8
    )
    dark_color = hsv2rgba(base_hue, uniform(0.4, 0.6), 0.2)

    color = build_color_ramp(
        nw,
        nw.musgrave(10),
        [0.0, 0.3, 0.7, 1.0],
        [dark_color, dark_color, bright_color, bright_color],
    )
    roughness = 0.8
    bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": color, "Roughness": roughness}
    )
    return bsdf


class SlimeMold:
    def __init__(self):
        pass

    def apply(self, obj, selection=None):
        scatter_obj = butil.spawn_vert("scatter:" + "slime_mold")
        surface.add_geomod(
            scatter_obj, geo_base_selection, apply=True, input_args=[obj, selection]
        )
        if len(scatter_obj.data.vertices) < 5:
            butil.delete(scatter_obj)
            return

        def end_index(nw):
            return nw.build_index_case(
                np.random.randint(0, len(scatter_obj.data.vertices), 40)
            )

        def weight(nw):
            return nw.build_float_curve(
                nw.new_node(Nodes.InputEdgeAngle).outputs["Signed Angle"],
                [(0, 0.25), (0.2, 0.4)],
            )

        surface.add_geomod(
            scatter_obj,
            geo_shortest_path,
            apply=True,
            input_args=[end_index, weight, 0.1, 0.02],
        )
        treeify(scatter_obj)
        surface.add_geomod(
            scatter_obj,
            geo_radius,
            apply=True,
            input_args=[
                lambda nw: nw.build_float_curve(
                    nw.new_node(Nodes.NamedAttribute, ["spline_parameter"]),
                    [(0, 0.008), (1, 0.015)],
                ),
                6,
            ],
        )
        base_hue = uniform(0.02, 0.16)
        assign_material(scatter_obj, shaderfunc_to_material(shader_mold, base_hue))

        return scatter_obj


def apply(obj, selection=None):
    SlimeMold().apply(obj, selection)
