# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from numpy.random import normal

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def liquid_particle_material(nw: NodeWrangler):
    # Code generated using version 2.5.1 of the node_transpiler

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": (1.0000, 1.0000, 1.0000, 1.0000),
            "Subsurface Color": (0.7147, 0.6062, 0.8000, 1.0000),
            "Specular IOR Level": 0.0886,
            "Roughness": 0.2705 + (0.1 * normal()),
            "Sheen Tint": 0.0000,
            "Coat Roughness": 0.0000,
            "IOR": 1.2000,
            "Transmission Weight": 0.2818 + (0.1 * normal()),
        },
        attrs={"distribution": "MULTI_GGX"},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf}
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, liquid_particle_material, selection=selection)
