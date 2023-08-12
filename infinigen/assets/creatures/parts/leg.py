# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from itertools import chain
import bpy

import numpy as np
from numpy.random import uniform as U, normal as N

from infinigen.core.util.math import clip_gaussian

from infinigen.assets.creatures.util.genome import Joint, IKParams

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube, nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle

from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util.part_util import nodegroup_to_part
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_quadruped_back_leg', singleton=False, type='GeometryNodeTree')
def nodegroup_quadruped_back_leg(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.8, 0.1, 0.05)),
            ('NodeSocketVector', 'angles_deg', (30.0, -100.0, 81.0)),
            ('NodeSocketVector', 'Thigh Rad1 Rad2 Fullness', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Calf Rad1 Rad2 Fullness', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Thigh Height Tilt1 Tilt2', (0.6, 0.0, 0.0)),
            ('NodeSocketVector', 'Calf Height Tilt1 Tilt2', (0.8, 0.0, 0.0)),
            ('NodeSocketFloat', 'fullness', 50.0),
            ('NodeSocketFloat', 'aspect', 1.0)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"], 'fullness': group_input.outputs["fullness"], 'Origin': (-0.05, 0.0, 0.0)})
    
    thigh = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.02, 3.1416, 3.0), 'Coord 1': (0.1, -0.14, 1.47), 'Coord 2': (0.73, 4.71, 1.13), 'StartRad, EndRad, Fullness': group_input.outputs["Thigh Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Thigh Height Tilt1 Tilt2"]},
        label='Thigh')
    
    calf = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.51, 18.91, 0.4), 'Coord 1': (0.69, 0.26, 0.0), 'Coord 2': (0.94, 1.5708, 1.13), 'StartRad, EndRad, Fullness': group_input.outputs["Calf Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Calf Height Tilt1 Tilt2"]},
        label='Calf')
    
    thigh_2 = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.04, 3.1416, 0.0), 'Coord 1': (0.01, 3.46, -0.05), 'Coord 2': (0.73, 4.71, 0.9), 'StartRad, EndRad, Fullness': group_input.outputs["Thigh Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Thigh Height Tilt1 Tilt2"]},
        label='Thigh 2')
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [thigh, calf, thigh_2]})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [join_geometry, simple_tube_v2.outputs["Geometry"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry_1, 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"]})

class QuadrupedBackLeg(PartFactory):

    tags = ['leg']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((1.8, 0.1, 0.05)) * N(1, (0.2, 0, 0), 3),
            'angles_deg': np.array((40.0, -120.0, 100)),
            'fullness': 50.0,
            'aspect': 1.0,
            'Thigh Rad1 Rad2 Fullness':  np.array((0.33, 0.15, 2.5),) * N(1, 0.1, 3),
            'Calf Rad1 Rad2 Fullness':   np.array((0.17, 0.07, 2.5),) * N(1, 0.1, 3),
            'Thigh Height Tilt1 Tilt2':  np.array((0.6, 0.0, 0.0),) + N(0, [0.05, 2, 10]),
            'Calf Height Tilt1 Tilt2':   np.array((0.8, 0.0, 0.0)) + N(0, [0.05, 10, 10])
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_quadruped_back_leg, params)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # shoulder
            0.5: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # elbow
        } 
        tag_object(part.obj, 'quadruped_back_leg')
        return part

@node_utils.to_nodegroup('nodegroup_quadruped_front_leg', singleton=False, type='GeometryNodeTree')
def nodegroup_quadruped_front_leg(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.43, 0.1, 0.1)),
            ('NodeSocketVector', 'angles_deg', (-20.0, 16.0, 9.2)),
            ('NodeSocketFloat', 'aspect', 1.0),
            ('NodeSocketVector', 'Shoulder Rad1 Rad2 Fullness', (0.22, 0.0, 0.0)),
            ('NodeSocketVector', 'Calf Rad1 Rad2 Fullness', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Elbow Rad1 Rad2 Fullness', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Shoulder Height, Tilt1, Tilt2', (0.74, 0.0, 0.0)),
            ('NodeSocketVector', 'Elbow Height, Tilt1, Tilt2', (0.9, 0.0, 0.0)),
            ('NodeSocketVector', 'Calf Height, Tilt1, Tilt2', (0.74, 0.0, 0.0))])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"], 'fullness': 2.5, 'Origin': (-0.15, 0.0, 0.09)})
    
    shoulder = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.0, 0.0, 0.0), 'Coord 1': (0.2, 0.0, 0.0), 'Coord 2': (0.55, 0.0, 0.0), 'StartRad, EndRad, Fullness': group_input.outputs["Shoulder Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Shoulder Height, Tilt1, Tilt2"]},
        label='Shoulder')
    
    elbow_2 = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.53, 1.5708, 1.69), 'Coord 1': (0.57, 0.0, 0.0), 'Coord 2': (0.95, 0.0, 0.0), 'StartRad, EndRad, Fullness': group_input.outputs["Elbow Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Elbow Height, Tilt1, Tilt2"]},
        label='Elbow 2')
    
    elbow_1 = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.22, 1.5708, 1.0), 'Coord 1': (0.4, 0.0, 0.0), 'Coord 2': (0.57, 1.571, 1.7), 'StartRad, EndRad, Fullness': group_input.outputs["Elbow Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Elbow Height, Tilt1, Tilt2"]},
        label='Elbow 1')
    
    forearm = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.41, -1.7008, 0.6), 'Coord 1': (0.57, 0.0, 0.8), 'Coord 2': (0.95, 0.0, 0.0), 'StartRad, EndRad, Fullness': group_input.outputs["Calf Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Calf Height, Tilt1, Tilt2"]},
        label='Forearm')
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [shoulder, elbow_2, elbow_1, forearm, simple_tube_v2.outputs["Geometry"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"]})

class QuadrupedFrontLeg(PartFactory):

    tags = ['leg']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((1.43, 0.1, 0.1)) * N(1, (0.2, 0, 0), 3),
            'angles_deg': np.array((-40.0, 120.0, -100)),
            'aspect': 1.0,
            'Shoulder Rad1 Rad2 Fullness':   np.array((0.22, 0.22, 2.5)) * N(1, 0.1, 3),
            'Calf Rad1 Rad2 Fullness':       np.array((0.08, 0.08, 2.5)) * N(1, 0.1, 3),
            'Elbow Rad1 Rad2 Fullness':      np.array((0.12, 0.1, 2.5) * N(1, 0.1, 3)),
            'Shoulder Height, Tilt1, Tilt2': np.array((0.74, 0.0, 0.0)) + N(0, [0.05, 10, 10]),
            'Elbow Height, Tilt1, Tilt2':    np.array((0.9, 0.0, 0.0)) + N(0, [0.05, 10, 10]),
            'Calf Height, Tilt1, Tilt2':     np.array((0.74, 0.0, 0.0)) + N(0, [0.05, 10, 10]),
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_quadruped_front_leg, params)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # shoulder
            0.6: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])) # elbow
        } 
        tag_object(part.obj, 'quadruped_front_leg')
        return part

@node_utils.to_nodegroup('nodegroup_bird_leg', singleton=False, type='GeometryNodeTree')
def nodegroup_bird_leg(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.0, 0.09, 0.06)),
            ('NodeSocketVector', 'angles_deg', (-70.0, 90.0, -2.0)),
            ('NodeSocketFloat', 'aspect', 1.0),
            ('NodeSocketFloat', 'fullness', 8.0),
            ('NodeSocketVector', 'Thigh Rad1 Rad2 Fullness', (0.18, 0.1, 1.26)),
            ('NodeSocketVector', 'Shin Rad1 Rad2 Fullness', (0.07, 0.06, 5.0))])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"], 'fullness': group_input.outputs["fullness"]})
    
    surface_muscle = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.0, 0.0, 0.0), 'Coord 1': (0.2, 0.0, 0.0), 'Coord 2': (0.4, 1.5708, 1.0), 'StartRad, EndRad, Fullness': group_input.outputs["Thigh Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': (0.72, -21.05, 0.0)})
    
    surface_muscle_1 = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.32, 0.0, 0.0), 'Coord 1': (0.5, 1.5708, 0.0), 'Coord 2': (0.74, 1.32, 0.29), 'StartRad, EndRad, Fullness': group_input.outputs["Shin Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': (0.72, -21.05, 0.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [surface_muscle, surface_muscle_1, simple_tube_v2.outputs["Geometry"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"]})

class BirdLeg(PartFactory):

    tags = ['leg']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((1, 0.09, 0.06)) * np.array((clip_gaussian(1, 0.3, 0.2, 1.5), *N(1, 0.1, 2))),
            'angles_deg': np.array((-70.0, 90.0, -2.0)),
            'aspect': N(1, 0.05),
            'fullness': 8.0 * N(1, 0.1),
            'Thigh Rad1 Rad2 Fullness': np.array((0.18, 0.1, 1.26)) * N(1, 0.1, 3),
            'Shin Rad1 Rad2 Fullness': np.array((0.07, 0.06, 5.0)) * N(1, 0.1, 3)
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_bird_leg, params)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # shoulder
            0.5: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # elbow
        } 
        part.iks = {}
        tag_object(part.obj, 'bird_leg')
        return part

@node_utils.to_nodegroup('nodegroup_insect_leg', singleton=False, type='GeometryNodeTree')
def nodegroup_insect_leg(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.24, 0.02, 0.01)),
            ('NodeSocketVector', 'angles_deg', (0.0, -63.9, 31.39)),
            ('NodeSocketFloat', 'Carapace Rad Pct', 1.4),
            ('NodeSocketVector', 'spike_length_rad1_rad2', (0.1, 0.025, 0.0))])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'proportions': (0.2533, 0.3333, 0.1333), 'do_bezier': False})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["length_rad1_rad2"], 'Scale': group_input.outputs["Carapace Rad Pct"]},
        attrs={'operation': 'SCALE'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': scale.outputs["Vector"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["Y"], 'Y': separate_xyz.outputs["Y"], 'Z': 30.0})
    
    surface_muscle = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.0, 0.0, 0.0), 'Coord 1': (0.01, 0.0, 0.0), 'Coord 2': (0.35, 0.0, 0.0), 'StartRad, EndRad, Fullness': combine_xyz, 'ProfileHeight, StartTilt, EndTilt': (0.73, 0.0, 0.0)})
    
    trim_curve = nw.new_node(Nodes.TrimCurve,
        input_kwargs={'Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Start': 0.4892, 'End': 0.725})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': trim_curve, 'Count': 4})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': resample_curve})
    
    simple_tube_v2_1 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["spike_length_rad1_rad2"], 'angles_deg': (0.0, -40.0, 0.0)})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': curve_to_mesh, 'Instance': simple_tube_v2_1.outputs["Geometry"], 'Rotation': (0.0, 0.1239, 0.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [simple_tube_v2.outputs["Geometry"], surface_muscle, instance_on_points]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Endpoint': simple_tube_v2.outputs["Endpoint"]})

class InsectLeg(PartFactory):

    tags = ['leg', 'rigid']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((1, 0.02, 0.01)) * N(1, 0.25, 3) ,
            'angles_deg': np.array((0.0, -63.9, 31.39)) + N(0, 10, 3),
            'Carapace Rad Pct': 1.4 * U(0.5, 2),
            'spike_length_rad1_rad2': np.array((0.2, 0.025, 0.0)) * N(1, (0.2, 0.1, 0.1), 3),
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_insect_leg, params)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # shoulder
            0.3: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])),
            0.7: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]]))
        } 
        part.iks = {1.0: IKParams('foot', rotation_weight=0.1, chain_parts=1)}
        tag_object(part.obj, 'insect_leg')
        return part