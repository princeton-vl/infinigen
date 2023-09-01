# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

def shader_tongue(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Scale': 37.88})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': musgrave_texture})
    colorramp.color_ramp.elements[0].position = 0.24
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.0979, 0.0979, 0.0979, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.8, 0.0605, 0.0437, 1.0), 'Subsurface': 0.0312, 'Subsurface Color': (0.8, 0.0, 0.2679, 1.0), 'Roughness': colorramp.outputs["Color"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_tongue, selection=selection)