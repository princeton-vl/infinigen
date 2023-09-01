# Derived from https://www.blendswap.com/blend/30728
# Original node-graph created by PedroPLopes https://www.blendswap.com/profile/1609866 and licensed CC-0

import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

@node_utils.to_nodegroup('nodegroup_auto_exposure', singleton=False, type='CompositorNodeTree')
def nodegroup_auto_exposure(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketColor', 'Image', (1.0000, 1.0000, 1.0000, 1.0000)),
            ('NodeSocketFloat', 'EV Compensation', 0.0000),
            ('NodeSocketFloat', 'Metering Area', 1.0000)])
    
    divide = nw.new_node('CompositorNodeMath',
        input_kwargs={0: 1.0000, 1: group_input.outputs["Metering Area"]},
        attrs={'operation': 'DIVIDE'})
    
    scale = nw.new_node('CompositorNodeScale', input_kwargs={'Image': group_input.outputs["Image"], 'X': divide, 'Y': divide})
    
    multiply = nw.new_node('CompositorNodeMath',
        input_kwargs={0: group_input.outputs["EV Compensation"], 1: -1.0000},
        attrs={'operation': 'MULTIPLY'})
    
    exposure = nw.new_node(Nodes.Exposure, input_kwargs={'Image': scale, 'Exposure': multiply})
    
    levels = nw.new_node('CompositorNodeLevels', input_kwargs={'Image': exposure}, attrs={'channel': 'LUMINANCE'})
    
    multiply_1 = nw.new_node('CompositorNodeMath',
        input_kwargs={0: levels.outputs["Mean"], 1: 2.0000},
        attrs={'operation': 'MULTIPLY'})
    
    rgb_curves = nw.new_node('CompositorNodeCurveRGB',
        input_kwargs={'Image': group_input.outputs["Image"], 'White Level': multiply_1})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Image': rgb_curves}, attrs={'is_active_output': True})
