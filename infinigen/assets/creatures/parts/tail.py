# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import numpy as np
from numpy.random import normal as N

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils

from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle

from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util.part_util import nodegroup_to_part
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_tail', singleton=False, type='GeometryNodeTree')
def nodegroup_tail(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.49, 0.05, 0.02)),
            ('NodeSocketVector', 'angles_deg', (31.39, 65.81, -106.93)),
            ('NodeSocketFloat', 'aspect', 1.0)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"]})

class Tail(PartFactory):

    tags = ['tail']

    def sample_params(self):
        return {
            'length_rad1_rad2': (N(1.5, 0.5), 0.05, 0.02),
            'angles_deg': np.array((31.39, 65.81, -106.93)) * N(1, 0.1),
            'aspect': N(1, 0.05)
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_tail, params)
        part.joints = {
            i: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]]))
            for i in np.linspace(0, 1, 6)
        }
        part.iks = {1.0: IKParams(name='tail', chain_parts=1)}
        tag_object(part.obj, 'tail')
        return part