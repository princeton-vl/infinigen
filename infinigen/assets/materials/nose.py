# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

def shader_nose(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Scale': U(2, 6), 'Detail': 14.699999999999999, 'Dimension': 1.5})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': musgrave_texture})
    colorramp.color_ramp.elements[0].position = U(0.2, 0.6)
    colorramp.color_ramp.elements[0].color = (0.008, 0.0053, 0.0044, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.7068, 0.436, 0.35, 1.0)
    
    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Scale': 10.0})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture_1, 3: N(0.4, 0.1), 4: N(0.7, 0.15)})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Roughness': map_range.outputs["Result"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})



def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_nose, selection=selection)
