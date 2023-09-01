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

from infinigen.assets.creatures.insects.utils.geom_utils import nodegroup_simple_tube_v2

@node_utils.to_nodegroup('nodegroup_dragonfly_antenna', singleton=False, type='GeometryNodeTree')
def nodegroup_dragonfly_antenna(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.24, 0.02, 0.01)),
            ('NodeSocketVector', 'angles_deg', (0.0, -63.9, 31.39)),
            ('NodeSocketFloat', 'Carapace Rad Pct', 1.4),
            ('NodeSocketVector', 'spike_length_rad1_rad2', (0.1, 0.025, 0.0))])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'proportions': (0.2533, 0.3333, -0.2267), 'do_bezier': False})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Endpoint': simple_tube_v2.outputs["Endpoint"]})