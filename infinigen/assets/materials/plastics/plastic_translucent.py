# Copyright (c) Princeton University.
# This source code is licensed under the GPL license found in the LICENSE file in the root directory of this
# source tree.
import colorsys

from infinigen.core.util.color import hsv2rgba
from infinigen.assets.materials import common
from infinigen.core.util.random import log_uniform
from infinigen.assets.materials.utils.surface_utils import sample_range
from numpy.random import uniform
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

    # Code generated using version 2.4.3 of the node_transpiler

    layer_weight = nw.new_node('ShaderNodeLayerWeight', input_kwargs={'Blend': sample_range(0.2, 0.4)})

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = base_color

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = sample_range(1.2, 1.6)

    glass_bsdf = nw.new_node('ShaderNodeBsdfGlass', input_kwargs={'Color': rgb, 'Roughness': 0.2, 'IOR': value})

    glossy_bsdf = nw.new_node('ShaderNodeBsdfGlossy', input_kwargs={'Roughness': 0.2})

    mix_shader = nw.new_node(Nodes.MixShader,
                             input_kwargs={'Fac': layer_weight.outputs["Fresnel"], 1: glass_bsdf, 2: glossy_bsdf
                             })

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': mix_shader})


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_translucent_plastic, selection, **kwargs)