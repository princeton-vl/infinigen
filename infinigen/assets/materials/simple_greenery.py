# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from numpy.random import normal as N
from numpy.random import uniform as U

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import color_category


def shader_simple_greenery(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    fac_color = nw.new_node(Nodes.MapRange, attrs={'interpolation_type': 'SMOOTHSTEP'},
        input_kwargs={'Value': noise(), 4: U(0.1, 1)})
    color = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': fac_color.outputs["Result"], 'Color1': color_category('greenery'), 'Color2': color_category('greenery')})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': color})
    
    rough = nw.new_node(Nodes.MapRange, attrs={'interpolation_type': 'SMOOTHSTEP'},
        input_kwargs={'Value': noise(), 3: U(0.1, 0.8), 4: U(0.1, 0.8)})
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': color, 'Roughness': rough.outputs["Result"]})
    
    fac_translucent = nw.new_node(Nodes.MapRange, attrs={'interpolation_type': 'SMOOTHSTEP'},
        input_kwargs={'Value': noise(), 3: U(0.6, 0.9)})
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': fac_translucent.outputs["Result"], 1: translucent_bsdf, 2: principled_bsdf})
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})
     
class Simple_Greenery():
    def apply(self, obj, selection=None, **kwargs):
        surface.add_material(obj, shader_simple_greenery, selection=selection)
