# Copyright (C) 2024, Princeton University.
# This source code is licensed under the GPL license found in the LICENSE file in the root directory of this
# source tree.

# Authors: Mingzhe Wang, Lingjie Mei


from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.materials.utils.surface_utils import sample_range
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform


def shader_translucent_plastic(nw: NodeWrangler, clear=False, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    layer_weight = nw.new_node(
        "ShaderNodeLayerWeight", input_kwargs={"Blend": sample_range(0.2, 0.4)}
    )

    rgb = nw.new_node(Nodes.RGB)

    if clear:
        base_color = hsv2rgba(0, 0, log_uniform(0.4, 0.8))
    else:
        base_color = hsv2rgba(uniform(0, 1), uniform(0.5, 0.8), log_uniform(0.4, 0.8))
    rgb.outputs[0].default_value = base_color

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = sample_range(1.2, 1.6)

    glass_bsdf = nw.new_node(
        "ShaderNodeBsdfGlass",
        input_kwargs={"Color": rgb, "Roughness": 0.2, "IOR": value},
    )

    glossy_bsdf = nw.new_node("ShaderNodeBsdfGlossy", input_kwargs={"Roughness": 0.2})

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": layer_weight.outputs["Fresnel"],
            1: glass_bsdf,
            2: glossy_bsdf,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": mix_shader}
    )


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_translucent_plastic, selection, **kwargs)
