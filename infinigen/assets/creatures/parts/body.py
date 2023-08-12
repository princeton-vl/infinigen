# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import numpy as np
from numpy.random import normal as N, uniform

from infinigen.assets.creatures.util import creature_util as cutil
from infinigen.assets.creatures.util.creature import Part, PartFactory
from infinigen.assets.creatures.util.geometry import nurbs

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_polar_bezier, nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.geometry import nodegroup_symmetric_clone
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle

from infinigen.assets.creatures.util import part_util
from infinigen.assets.creatures.util.geometry import lofting, nurbs
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_quadruped_body', singleton=False, type='GeometryNodeTree')
def nodegroup_quadruped_body(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input_1 = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Pct Ribcage', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Pct Backpart', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Spine StartRad, EndRad, Fullness', (0.05, 0.05, 3.0)),
            ('NodeSocketVector', 'Belly StartRad, EndRad, Fullness', (0.07, 0.15, 2.5)),
            ('NodeSocketVector', 'Belly ProfileHeight, StartTilt, EndTilt', (0.5, 114.0, 114.0)),
            ('NodeSocketVector', 'TopFlank StartRad, EndRad, Fullness', (0.2, 0.28, 2.5)),
            ('NodeSocketVector', 'TopFlank ProfileHeight, StartTilt, EndTilt', (0.6, 72.0, 8.0)),
            ('NodeSocketVector', 'BackFlank StartRad, EndRad, Fullness', (0.15, 0.15, 2.5)),
            ('NodeSocketVector', 'BackFlank ProfileHeight, StartTilt, EndTilt', (0.6, 53.0, 53.0)),
            ('NodeSocketVector', 'BottomFlank StartRad, EndRad, Fullness', (0.14, 0.27, 2.5)),
            ('NodeSocketVector', 'BottomFlank0 ProfileHeight, StartTilt, EndTilt', (0.6, -29.0, 48.0)),
            ('NodeSocketVector', 'BottomFlank1 ProfileHeight, StartTilt, EndTilt', (0.5, -44.0, -17.4)),
            ('NodeSocketFloat', 'aspect', 1.0)])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input_1.outputs["length_rad1_rad2"], 1: group_input_1.outputs["Pct Ribcage"]},
        attrs={'operation': 'MULTIPLY'})
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': multiply.outputs["Vector"], 'angles_deg': (0.0, -1.0, 4.0), 'proportions': (0.3333, 0.45, 0.3), 'aspect': group_input_1.outputs["aspect"], 'fullness': 3.0, 'Origin': (0.48, 0.0, -0.07)})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input_1.outputs["length_rad1_rad2"], 1: group_input_1.outputs["Pct Backpart"]},
        attrs={'operation': 'MULTIPLY'})
    
    vector = nw.new_node(Nodes.Vector)
    vector.vector = (-0.01, 0.0, 0.02)
    
    simple_tube_v2_1 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': multiply_1.outputs["Vector"], 'angles_deg': (0.94, -3.94, 11.66), 'proportions': (0.3, 0.6, 0.2), 'aspect': group_input_1.outputs["aspect"], 'fullness': 7.0, 'Origin': vector})
    
    union = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 2': [simple_tube_v2.outputs["Geometry"], simple_tube_v2_1.outputs["Geometry"]]},
        attrs={'operation': 'UNION'})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Start': vector, 'Middle': simple_tube_v2_1.outputs["Endpoint"], 'End': simple_tube_v2.outputs["Endpoint"]})
    
    bottom_flank_0 = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union, 'Skeleton Curve': quadratic_bezier, 'Coord 0': (0.16, 0.91, 0.66), 'Coord 1': (0.38, 0.37, 1.0), 'Coord 2': (0.67, -0.42, 0.6), 'StartRad, EndRad, Fullness': group_input_1.outputs["BottomFlank StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input_1.outputs["BottomFlank0 ProfileHeight, StartTilt, EndTilt"]},
        label='Bottom Flank 0')
    
    top_flank = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union, 'Skeleton Curve': quadratic_bezier, 'Coord 0': (0.25, 4.91, 0.5), 'Coord 1': (0.65, -0.35, 1.0), 'Coord 2': (0.88, 0.47, 0.7), 'StartRad, EndRad, Fullness': group_input_1.outputs["TopFlank StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input_1.outputs["TopFlank ProfileHeight, StartTilt, EndTilt"]},
        label='Top  Flank')
    
    bottom_flank_1 = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union, 'Skeleton Curve': quadratic_bezier, 'Coord 0': (0.36, 1.03, 0.95), 'Coord 1': (0.6, 0.85, 1.0), 'Coord 2': (0.9, -0.01, 0.71), 'StartRad, EndRad, Fullness': group_input_1.outputs["BottomFlank StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input_1.outputs["BottomFlank1 ProfileHeight, StartTilt, EndTilt"]},
        label='Bottom Flank 1')
    
    back_flank = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union, 'Skeleton Curve': quadratic_bezier, 'Coord 0': (0.02, -0.9, 0.53), 'Coord 1': (0.2, -0.85, 0.85), 'Coord 2': (0.61, -0.99, 0.7), 'StartRad, EndRad, Fullness': group_input_1.outputs["BackFlank StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input_1.outputs["BackFlank ProfileHeight, StartTilt, EndTilt"]},
        label='Back Flank')
    
    belly = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union, 'Skeleton Curve': quadratic_bezier, 'Coord 0': (0.24, 1.52, 0.7), 'Coord 1': (0.48, 1.24, 1.42), 'Coord 2': (0.92, 1.41, 0.97), 'StartRad, EndRad, Fullness': group_input_1.outputs["Belly StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input_1.outputs["Belly ProfileHeight, StartTilt, EndTilt"]},
        label='Belly')
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [bottom_flank_0, top_flank, bottom_flank_1, back_flank, belly]})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': join_geometry_1})
    
    spine = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union, 'Skeleton Curve': quadratic_bezier, 'Coord 0': (0.05, -1.5708, 1.0), 'Coord 1': (0.5, -1.5708, 1.2), 'Coord 2': (0.95, -1.5708, 1.0), 'StartRad, EndRad, Fullness': group_input_1.outputs["Spine StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': (1.0, 0.0, 0.0)},
        label='Spine')
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [union, symmetric_clone.outputs["Both"], spine]})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': quadratic_bezier})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': reroute, 'Base Mesh': union})

class QuadrupedBody(PartFactory):

    tags = ['body', 'head']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((1.7, 0.65, 0.65)) * N(1, 0.15, 3),
            'Pct Ribcage': (0.76, 0.56, 0.56) * N(1, 0.1, 3),
            'Pct Backpart': (0.64, 0.25, 0.4) * N(1, 0.1, 3),
            'Spine StartRad, EndRad, Fullness': np.array((0.05, 0.05, 3.0)) * N(1, 0.1, 3),
            'Belly StartRad, EndRad, Fullness': np.array((0.07, 0.15, 2.5)) * N(1, 0.1, 3),
            'Belly ProfileHeight, StartTilt, EndTilt': (0.5, 114.0, 114.0),
            'TopFlank StartRad, EndRad, Fullness': (0.2, 0.28, 2.5),
            'TopFlank ProfileHeight, StartTilt, EndTilt': (0.6, 72.0, 8.0),
            'BackFlank StartRad, EndRad, Fullness': (0.15, 0.15, 2.5),
            'BackFlank ProfileHeight, StartTilt, EndTilt': (0.6, 53.0, 53.0),
            'BottomFlank StartRad, EndRad, Fullness': (0.14, 0.27, 2.5),
            'BottomFlank0 ProfileHeight, StartTilt, EndTilt': (0.6, -29.0, 48.0),
            'BottomFlank1 ProfileHeight, StartTilt, EndTilt': (0.5, -44.0, -17.4),
            'aspect': N(1, 0.1)
        }

    def make_part(self, params):
        
        part = part_util.nodegroup_to_part(nodegroup_quadruped_body, params)
        part.joints = {
            i: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]]))
            for i in np.linspace(0, 1, 4, endpoint=True)
        }
        part.iks = {
            0.0: IKParams(name='hip', mode='pin', target_size=0.3),
            1.0: IKParams(name='shoulder', rotation_weight=0.1, target_size=0.4)
        }
        tag_object(part.obj, 'quadruped_body')
        return part

@node_utils.to_nodegroup('nodegroup_fish_body', singleton=False, type='GeometryNodeTree')
def nodegroup_fish_body(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.89, 0.2, 0.29)),
            ('NodeSocketVector', 'angles_deg', (7.0, 0.51, -9.02)),
            ('NodeSocketFloat', 'aspect', 0.56),
            ('NodeSocketFloat', 'fullness', 3.43)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"], 'fullness': group_input.outputs["fullness"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Endpoint': simple_tube_v2.outputs["Endpoint"]})

class FishBody(PartFactory):

    tags = ['body']

    def sample_params(self):
        return {}

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_fish_body, params)
        part.joints = {
            i: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]]))
            for i in np.linspace(0, 1, 4, endpoint=True)
        }
        part.iks = {
            0.0: IKParams(name='hip', mode='pin', target_size=0.3),
            1.0: IKParams(name='shoulder', rotation_weight=0.1, target_size=0.4)
        }
        tag_object(part.obj, 'fish_body')
        return part

@node_utils.to_nodegroup('nodegroup_bird_body', singleton=False, type='GeometryNodeTree')
def nodegroup_bird_body(nw: NodeWrangler):
    # Code generated using version 2.5.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.0000, 0.5000, 0.3000)),
                                            ('NodeSocketFloat', 'aspect', 1.0000),
                                            ('NodeSocketFloat', 'fullness', 2.0000)])

    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
                                 input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"],
                                               'proportions': (0.1000, 0.1000, 0.1000),
                                               'aspect': group_input.outputs["aspect"],
                                               'fullness': group_input.outputs["fullness"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"],
                                             'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"]})

class BirdBody(PartFactory):

    tags = ['body']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((0.95, 0.15, 0.2)) * N(1.0, 0.05, size=(3,)),
            'aspect': N(1.2, 0.02),
            'fullness': N(2, 0.1)
        }

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_bird_body, params)
        part.joints = {
            i: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]]))
            for i in np.linspace(0, 1, 4, endpoint=True)
        }
        part.iks = {
            0.0: IKParams(name='hip', mode='pin', target_size=0.3),
            1.0: IKParams(name='shoulder', rotation_weight=0.1, target_size=0.4)
        }
        tag_object(part.obj, 'bird_body')
        return part
        
