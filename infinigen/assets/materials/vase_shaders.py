# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from numpy.random import uniform

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba


def shader_ceramic(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler
    hsv = (uniform(0.0, 1.0), uniform(0.0, 0.75), uniform(0.0, 0.3))

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = hsv2rgba(hsv)

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": rgb,
            "Subsurface Weight": 0.3,
            "Subsurface Radius": (0.002, 0.002, 0.002),
            "Subsurface Color": rgb,
            "Subsurface Anisotropy": 0.2000,
            "Specular IOR Level": 0.2000,
            "Roughness": 0.0500,
            "Coat Weight": 0.5000,
            "Coat Roughness": 0.0500,
            "IOR": 1.4700,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def shader_glass(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    hsv = (uniform(0.0, 1.0), uniform(0.0, 0.2), 1.0)

    glass_bsdf = nw.new_node(
        Nodes.GlassBSDF,
        input_kwargs={"Color": hsv2rgba(hsv), "Roughness": uniform(0.05, 0.2)},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": glass_bsdf},
        attrs={"is_active_output": True},
    )
