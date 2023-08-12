# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy

import numpy as np
from numpy.random import uniform, normal

from infinigen.assets.creatures.util.genome import Joint, IKParams

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_attach_part


from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util.part_util import nodegroup_to_part
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_fish_fin', singleton=False, type='GeometryNodeTree')
def nodegroup_fish_fin(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.34, 0.07, 0.04)),
            ('NodeSocketVector', 'angles_deg', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'proportions', (0.3333, 0.3333, 0.3333)),
            ('NodeSocketFloat', 'aspect', 2.65),
            ('NodeSocketFloat', 'fullness', 4.0)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'proportions': group_input.outputs["proportions"], 'aspect': group_input.outputs["aspect"], 'fullness': group_input.outputs["fullness"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Endpoint': simple_tube_v2.outputs["Endpoint"]})

class FishFin(PartFactory):

    tags = ['limb', 'fin']

    def sample_params(self):
        return {
            'length_rad1_rad2': (0.34, 0.07, 0.04),
            'angles_deg': (0.0, 0.0, 0.0),
            'proportions': (0.3333, 0.3333, 0.3333),
            'aspect': 2.65,
            'fullness': 4.
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_fish_fin, params)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # shoulder
            0.6: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])) # elbow
        } 
        tag_object(part.obj, 'fish_fin')
        return part

@node_utils.to_nodegroup('nodegroup_fish_tail', singleton=False, type='GeometryNodeTree')
def nodegroup_fish_tail(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.5, 0.18, 0.04)),
            ('NodeSocketVector', 'angles_deg', (0.0, -4.6, 0.0)),
            ('NodeSocketFloat', 'aspect', 0.46),
            ('NodeSocketFloat', 'fullness', 4.0)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"], 'fullness': group_input.outputs["fullness"], 'Origin': (-0.07, 0.0, 0.0)})
    
    fishfin = nw.new_node(nodegroup_fish_fin().name,
        input_kwargs={'length_rad1_rad2': (0.34, 0.07, 0.11), 'aspect': 4.7})
    
    attach_part = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Geometry': fishfin.outputs["Geometry"], 'Length Fac': 0.775, 'Part Rot': (90.0, -20.7, 0.0)})
    
    fishfin_1 = nw.new_node(nodegroup_fish_fin().name,
        input_kwargs={'length_rad1_rad2': (0.34, 0.07, 0.11), 'aspect': 4.7})
    
    attach_part_1 = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Geometry': fishfin_1.outputs["Geometry"], 'Length Fac': 0.775, 'Part Rot': (90.0, 18.64, 0.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [attach_part.outputs["Geometry"], simple_tube_v2.outputs["Geometry"], attach_part_1.outputs["Geometry"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Endpoint': simple_tube_v2.outputs["Endpoint"]})

class FishTail(PartFactory):

    tags = ['tail']

    def sample_params(self):
        return {
            'length_rad1_rad2': (0.5, 0.18, 0.04),
            'angles_deg': (0.0, -4.6, 0.0),
            'aspect': 0.46,
            'fullness': 4.
        }
    
    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_fish_tail, params)
        part.joints = {
            t: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]]))
            for t in np.linspace(0, 0.7, 4)
        } 
        part.iks = {1.0: IKParams('tail', rotation_weight=0, chain_parts=1)}
        tag_object(part.obj, 'fish_tail')
        return part
