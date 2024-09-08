# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from numpy.random import normal

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.random import random_color_neighbour


def new_whitewater(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": (1.0000, 1.0000, 1.0000, 1.0000),
            "Subsurface Color": random_color_neighbour(
                (0.7147, 0.6062, 0.8000, 1.0000), 0.05, 0.05, 0.05
            ),
            "Specular IOR Level": 0.0886 + 0.01 * normal(),
            "Roughness": 0.1500,
            "Coat Roughness": 0.0000,
            "IOR": 1.1000,
            "Transmission Weight": 0.5000,
        },
        attrs={"distribution": "MULTI_GGX"},
    )

    volume_scatter = nw.new_node(
        "ShaderNodeVolumeScatter",
        input_kwargs={"Color": (0.8856, 0.8594, 1.0000, 1.0000), "Anisotropy": 0.1333},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Volume": volume_scatter},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, new_whitewater, selection=selection)
