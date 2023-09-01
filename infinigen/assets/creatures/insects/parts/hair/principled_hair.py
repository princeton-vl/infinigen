# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.creatures.insects.utils.geom_utils import nodegroup_shape_quadratic, nodegroup_circle_cross_section

@node_utils.to_nodegroup('nodegroup_principled_hair', singleton=False, type='GeometryNodeTree')
def nodegroup_principled_hair(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketIntUnsigned', 'Resolution', 4)])
    
    crosssection = nw.new_node(nodegroup_circle_cross_section().name,
        input_kwargs={'Resolution': group_input.outputs["Resolution"], 'radius': 0.5})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 2.0
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': crosssection, 'Scale': value})
    
    shapequadraticleghair = nw.new_node(nodegroup_shape_quadratic(radius_control_points=[(0.0, 0.1125), (0.625, 0.1), (1.0, 0.0531)]).name,
        input_kwargs={'Profile Curve': transform, 'noise amount tilt': 0.0, 'Resolution': 8, 'Start': (0.0, 0.0, 0.0), 'Middle': (-0.2, 0.0, 1.0), 'End': (0.0, 0.0, 2.66)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': shapequadraticleghair.outputs["Mesh"]})