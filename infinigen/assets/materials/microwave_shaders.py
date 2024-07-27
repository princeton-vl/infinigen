# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_black_medal(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    anisotropic_bsdf = nw.new_node(
        "ShaderNodeBsdfAnisotropic",
        input_kwargs={"Color": (0.0167, 0.0167, 0.0167, 1.0000)},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": anisotropic_bsdf},
        attrs={"is_active_output": True},
    )


def shader_black_glass(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    glossy_bsdf = nw.new_node(
        Nodes.GlossyBSDF,
        input_kwargs={"Color": (0.0068, 0.0068, 0.0068, 1.0000), "Roughness": 0.2000},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": glossy_bsdf},
        attrs={"is_active_output": True},
    )


def shader_glass(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    glass_bsdf = nw.new_node(Nodes.GlassBSDF, input_kwargs={"IOR": 1.5000})

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": glass_bsdf},
        attrs={"is_active_output": True},
    )
